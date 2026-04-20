"""
experiments/feature_importance.py

Аналіз важливості фіч на повному датасеті (XGBoost gain-based importance).
Показує які з нових і старих фіч реально предиктивні для 1X2.

Запуск: python -m experiments.feature_importance
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from db.session import SessionLocal
from db.models import Match, MatchStats, Odds, Team
from model.features.builder import build_dataset

# Ігноруємо ці не-фічеві колонки
NON_FEATURE_COLS = {"target", "match_id", "date", "league_id"}

# Виключаємо DC odds (не релевантні для 1x2 аналізу)
EXCLUDE_COLS = {"dc_1x_odds", "dc_2x_odds"}


def load_data(db):
    logger.info("Завантаження даних з БД...")

    matches = db.query(Match).filter(Match.status == "Finished").all()
    matches_df = pd.DataFrame([{
        "id": m.id, "date": pd.Timestamp(m.date),
        "home_team_id": m.home_team_id, "away_team_id": m.away_team_id,
        "home_score": m.home_score, "away_score": m.away_score,
        "league_id": m.league_id,
    } for m in matches])
    logger.info(f"Матчів: {len(matches_df)}")

    stats = db.query(MatchStats).join(Match).all()
    stats_df = pd.DataFrame([{
        "match_id": s.match_id,
        "date": pd.Timestamp(s.match.date),
        "home_team_id": s.match.home_team_id,
        "away_team_id": s.match.away_team_id,
        "home_score": s.match.home_score,
        "away_score": s.match.away_score,
        "home_xg": s.home_xg,
        "away_xg": s.away_xg,
        "home_shots":            s.home_shots,
        "away_shots":            s.away_shots,
        "home_shots_on_target":  s.home_shots_on_target,
        "away_shots_on_target":  s.away_shots_on_target,
        "home_shots_inside_box": s.home_shots_inside_box,
        "away_shots_inside_box": s.away_shots_inside_box,
        "home_possession":       s.home_possession,
        "away_possession":       s.away_possession,
        "home_corners":          s.home_corners,
        "away_corners":          s.away_corners,
        "home_gk_saves":         s.home_gk_saves,
        "away_gk_saves":         s.away_gk_saves,
        "home_passes_accurate":  s.home_passes_accurate,
        "away_passes_accurate":  s.away_passes_accurate,
        "home_passes_total":     s.home_passes_total,
        "away_passes_total":     s.away_passes_total,
    } for s in stats])
    logger.info(f"Статистик: {len(stats_df)}, з shots: {stats_df['home_shots_on_target'].notna().sum()}")

    odds = db.query(Odds).filter_by(market="1x2").all()
    odds_df = pd.DataFrame([{
        "match_id": o.match_id, "market": o.market, "bookmaker": o.bookmaker,
        "outcome": o.outcome, "value": o.value, "is_closing": o.is_closing,
    } for o in odds])
    logger.info(f"Odds: {len(odds_df)}")

    teams = {t.id: {"elo": t.elo} for t in db.query(Team).all()}

    return matches_df, stats_df, odds_df, teams


def run():
    db = SessionLocal()
    try:
        matches_df, stats_df, odds_df, teams = load_data(db)
    finally:
        db.close()

    logger.info("Будуємо датасет...")
    dataset = build_dataset(matches_df, stats_df, odds_df, teams)
    logger.info(f"Датасет: {len(dataset)} рядків, {len(dataset.columns)} колонок")

    # Тільки матчі з market odds
    dataset = dataset[dataset["market_home_prob"].notna()].reset_index(drop=True)
    logger.info(f"З market odds: {len(dataset)} рядків")

    # Визначаємо feature columns
    feature_cols = [
        c for c in dataset.columns
        if c not in NON_FEATURE_COLS and c not in EXCLUDE_COLS
        and dataset[c].dtype in [np.float64, np.int64, float, int, bool]
    ]
    logger.info(f"Фіч для аналізу: {len(feature_cols)}")

    X = dataset[feature_cols].fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(dataset["target"])
    classes = le.classes_  # ['away', 'draw', 'home']

    # Часове розділення: 80% train, 20% test (without leakage)
    dataset_sorted = dataset.sort_values("date").reset_index(drop=True)
    X = dataset_sorted[feature_cols].fillna(0)
    y = le.transform(dataset_sorted["target"])

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    test_acc = (model.predict(X_test) == y_test).mean()
    logger.info(f"Test accuracy: {test_acc:.3f}")

    # ─── Feature importance (gain) ───────────────────────────────────────────
    gain_scores = model.get_booster().get_score(importance_type="gain")
    weight_scores = model.get_booster().get_score(importance_type="weight")

    importance = pd.DataFrame([
        {
            "feature": f,
            "gain":   gain_scores.get(f, 0.0),
            "weight": weight_scores.get(f, 0),
        }
        for f in feature_cols
    ]).sort_values("gain", ascending=False)

    # ─── Категорії фіч ───────────────────────────────────────────────────────
    def categorize(name):
        if "market" in name:           return "market_odds"
        if "elo" in name:              return "elo"
        if "xg" in name:               return "xg"
        if any(x in name for x in ["shot_conv", "shot_acc", "shots_box", "save_pct", "pass_acc"]):
            return "efficiency"
        if any(x in name for x in ["shots_ot", "shots_box_avg", "possession", "corners", "gk_saves", "passes_acc_for"]):
            return "match_stats"
        if any(x in name for x in ["clean_sheet", "btts", "failed_to", "streak", "goals_std"]):
            return "form_advanced"
        if "form" in name:             return "form_basic"
        if "table" in name or "standing" in name: return "standings"
        if "rest" in name:             return "rest_days"
        if "injur" in name:            return "injuries"
        if "odds_move" in name or "sharp" in name: return "odds_movement"
        return "other"

    importance["category"] = importance["feature"].apply(categorize)

    # ─── Вивід топ-30 ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"FEATURE IMPORTANCE (XGBoost gain) — топ 40 з {len(feature_cols)}")
    print("="*70)
    top = importance[importance["gain"] > 0].head(40)
    for _, row in top.iterrows():
        bar = "█" * int(row["gain"] / top["gain"].max() * 30)
        print(f"  {row['feature']:<42} {bar:<30} {row['gain']:>8.1f}  [{row['category']}]")

    # ─── Важливість по категоріях ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("ВАЖЛИВІСТЬ ПО КАТЕГОРІЯХ (сума gain)")
    print("="*70)
    by_cat = importance.groupby("category")["gain"].sum().sort_values(ascending=False)
    total = by_cat.sum()
    for cat, val in by_cat.items():
        bar = "█" * int(val / total * 40)
        print(f"  {cat:<20} {bar:<40} {val/total*100:>5.1f}%")

    # ─── Нові фічі окремо ────────────────────────────────────────────────────
    new_feature_keywords = [
        "clean_sheet", "btts", "failed_to", "streak", "goals_for_std",
        "shot_conv", "shot_acc", "shots_box_ratio", "save_pct", "pass_acc",
        "shots_ot", "possession", "corners", "gk_saves", "passes_acc_for",
        "elo_momentum", "market_certainty", "market_home_edge",
    ]
    new_feats = importance[importance["feature"].apply(
        lambda x: any(kw in x for kw in new_feature_keywords)
    )].head(30)

    print("\n" + "="*70)
    print("НОВІ ФІЧІ — важливість")
    print("="*70)
    if new_feats.empty:
        print("  Нові фічі ще не заповнені (бекфіл не завершено)")
    else:
        for _, row in new_feats.iterrows():
            marker = "✓" if row["gain"] > 0 else "○"
            print(f"  {marker} {row['feature']:<42} gain={row['gain']:>8.1f}  [{row['category']}]")

    # ─── Збереження ──────────────────────────────────────────────────────────
    out_path = "experiments/results/feature_importance.csv"
    os.makedirs("experiments/results", exist_ok=True)
    importance.to_csv(out_path, index=False)
    logger.info(f"Збережено: {out_path}")
    print(f"\nПовний список: {out_path}")
    print(f"Test accuracy: {test_acc:.3f}  (baseline ~0.49 random)")


if __name__ == "__main__":
    run()
