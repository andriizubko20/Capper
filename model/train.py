import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger

from db.session import SessionLocal
from db.models import Match, MatchStats, Odds, Team
from model.features.builder import build_dataset

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    # Форма (загальна) — points враховує нічиї, wins видалено (корелює з points)
    "home_form_points", "home_form_goals_for_avg", "home_form_goals_against_avg",
    "away_form_points", "away_form_goals_for_avg", "away_form_goals_against_avg",

    # xG останні 10 — стабільніше ніж last 5
    # xg_diff видалено — = xg_for - xg_against (математичний дубль)
    "home_xg_for_avg_10", "home_xg_against_avg_10",
    "away_xg_for_avg_10", "away_xg_against_avg_10",

    # xG overperformance — клінічність реалізації
    "home_xg_overperformance", "away_xg_overperformance",

    # Elo diff (мультиколінеарний але дає слабкий сигнал поверх ринку)
    "elo_diff",

    # Дні відпочинку
    "home_rest_days", "away_rest_days",

    # Травми
    "home_injured_count", "away_injured_count",

    # Ринок
    "market_home_prob", "market_away_prob",
]

TARGET_CLASSES = ["away", "draw", "home"]  # алфавітний порядок для LabelEncoder


def train(dataset: pd.DataFrame, version: str = "v1") -> dict:
    """
    Навчає XGBoost на датасеті, калібрує ймовірності.
    dataset — результат build_dataset() з builder.py
    Повертає метрики навчання.
    """
    dataset = dataset.sort_values("date").reset_index(drop=True)

    # Train only on matches with valid market odds (otherwise market features are meaningless)
    if "market_home_prob" in dataset.columns:
        n_before = len(dataset)
        dataset = dataset[dataset["market_home_prob"].notna()].reset_index(drop=True)
        if len(dataset) < n_before:
            logger.info(f"Filtered to {len(dataset)} rows with valid market odds (dropped {n_before - len(dataset)})")

    available_features = [f for f in FEATURE_COLS if f in dataset.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")

    X = dataset[available_features].fillna(0)
    y_raw = dataset["target"]

    le = LabelEncoder()
    le.fit(TARGET_CLASSES)
    y = le.transform(y_raw)

    # Часове розділення: 80% train, 20% validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")

    # Logistic Regression: добре калібрована, інтерпретована коефіцієнти
    # C = сила регуляризації (менше C = більше регуляризації = менше оверфіту)
    n_train = len(X_train)
    C = 0.1 if n_train < 800 else (0.5 if n_train < 1500 else 1.0)

    calibrated = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])
    calibrated.fit(X_train, y_train)

    val_preds = calibrated.predict(X_val)
    accuracy = (val_preds == y_val).mean()
    logger.info(f"Validation accuracy: {accuracy:.3f}")

    # Виводимо топ коефіцієнти по кожному виходу
    lr_model = calibrated.named_steps["lr"]
    scaler = calibrated.named_steps["scaler"]
    classes = le.inverse_transform(range(len(lr_model.classes_)))
    coef_df = pd.DataFrame(
        lr_model.coef_,
        index=classes,
        columns=available_features,
    )
    logger.info("=== Топ-5 факторів для кожного виходу ===")
    for outcome in classes:
        top = coef_df.loc[outcome].abs().nlargest(5).index.tolist()
        vals = {f: round(coef_df.loc[outcome, f], 3) for f in top}
        logger.info(f"  {outcome}: {vals}")

    # Зберігаємо модель і encoder
    model_path = MODEL_DIR / f"model_{version}.pkl"
    encoder_path = MODEL_DIR / f"encoder_{version}.pkl"
    features_path = MODEL_DIR / f"features_{version}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    with open(features_path, "wb") as f:
        pickle.dump(available_features, f)

    logger.info(f"Model saved to {model_path}")

    return {
        "version": version,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_accuracy": accuracy,
        "features_used": len(available_features),
    }


CL_API_ID = 2  # Champions League excluded from training


def load_data_from_db():
    """Завантажує дані з БД у DataFrame для тренування."""
    db = SessionLocal()
    try:
        matches = pd.read_sql("""
            SELECT m.* FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE l.api_id != 2
        """, db.bind)
        stats = pd.read_sql("""
            SELECT ms.*, m.home_team_id, m.away_team_id, m.date,
                   m.home_score, m.away_score
            FROM match_stats ms
            JOIN matches m ON m.id = ms.match_id
        """, db.bind)
        odds = pd.read_sql("SELECT * FROM odds", db.bind)
        teams_raw = pd.read_sql("SELECT id, elo FROM teams", db.bind)
        teams = {row["id"]: {"elo": row["elo"]} for _, row in teams_raw.iterrows()}
        # Травми (якщо таблиця існує)
        try:
            injuries = pd.read_sql("SELECT match_id, team_id, player_api_id FROM injury_reports", db.bind)
        except Exception:
            injuries = pd.DataFrame(columns=["match_id", "team_id", "player_api_id"])
    finally:
        db.close()
    return matches, stats, odds, teams, injuries


if __name__ == "__main__":
    logger.info("Loading data from DB...")
    matches, stats, odds, teams, injuries = load_data_from_db()
    logger.info(f"Loaded {len(matches)} matches, {len(stats)} stats, {len(odds)} odds")

    logger.info("Building features...")
    dataset = build_dataset(matches, stats, odds, teams, injuries_df=injuries)
    logger.info(f"Dataset: {len(dataset)} rows, {len(dataset.columns)} columns")

    if len(dataset) < 100:
        logger.error("Not enough data to train. Need at least 100 matches.")
        exit(1)

    logger.info("Training model...")
    metrics = train(dataset, version="v1-test")
    logger.info(f"Done: {metrics}")
