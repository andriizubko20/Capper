"""
experiments/backtest_xgb_v3.py

XGBoost OOS backtest з повним feature set після бекфілу shots/possession.
Додано: efficiency, match_stats, raw odds.
Тренується на 2024-25, тестується на 2025-26.

Запуск: python -m experiments.backtest_xgb_v3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from model.train import load_data_from_db
from model.features.builder import build_dataset

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
MIN_EV     = 0.20
MIN_ODDS   = 1.5
MAX_ODDS   = 2.5
MAX_STAKE  = 0.04
FRACTIONAL = 0.25
INITIAL_BR = 1000.0

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}

FEATURE_COLS = [
    # Ринок
    "market_home_prob", "market_away_prob", "market_draw_prob",
    "market_certainty", "market_home_edge",
    "home_odds", "away_odds", "draw_odds",

    # Elo
    "elo_diff", "elo_home_win_prob",
    "home_elo_momentum", "away_elo_momentum", "delta_elo_momentum",

    # Базова форма
    "home_form_points", "home_form_goals_for_avg", "home_form_goals_against_avg",
    "home_form_wins", "home_form_losses",
    "away_form_points", "away_form_goals_for_avg", "away_form_goals_against_avg",
    "away_form_wins", "away_form_losses",
    "home_home_form_wins", "home_home_form_goals_for_avg",
    "away_away_form_wins", "away_away_form_goals_for_avg",

    # Розширена форма
    "home_clean_sheet_rate", "home_btts_rate", "home_failed_to_score_rate",
    "home_win_streak", "home_loss_streak", "home_goals_for_std",
    "away_clean_sheet_rate", "away_btts_rate", "away_failed_to_score_rate",
    "away_win_streak", "away_loss_streak", "away_goals_for_std",
    "delta_clean_sheet_rate", "delta_btts_rate", "delta_failed_to_score_rate",

    # xG
    "home_xg_for_avg", "home_xg_against_avg", "home_xg_ratio", "home_xg_diff_avg",
    "home_xg_for_avg_10", "home_xg_against_avg_10", "home_xg_ratio_10", "home_xg_diff_avg_10",
    "away_xg_for_avg", "away_xg_against_avg", "away_xg_ratio", "away_xg_diff_avg",
    "away_xg_for_avg_10", "away_xg_against_avg_10", "away_xg_ratio_10", "away_xg_diff_avg_10",
    "home_xg_overperformance", "away_xg_overperformance",

    # Match stats rolling avg (shots, possession, corners, saves, passes)
    "home_shots_ot_for_avg", "home_shots_ot_against_avg",
    "home_shots_box_for_avg", "home_shots_box_against_avg",
    "home_possession_avg", "home_corners_for_avg", "home_corners_against_avg",
    "home_gk_saves_avg", "home_passes_acc_for_avg",
    "away_shots_ot_for_avg", "away_shots_ot_against_avg",
    "away_shots_box_for_avg", "away_shots_box_against_avg",
    "away_possession_avg", "away_corners_for_avg", "away_corners_against_avg",
    "away_gk_saves_avg", "away_passes_acc_for_avg",
    "delta_shots_ot_for_avg", "delta_shots_ot_against_avg",
    "delta_shots_box_for_avg", "delta_possession_avg",
    "delta_gk_saves_avg", "delta_passes_acc_for_avg",

    # Efficiency
    "home_shot_conversion_rate", "home_shot_accuracy", "home_shots_box_ratio",
    "home_save_pct", "home_pass_accuracy_pct",
    "away_shot_conversion_rate", "away_shot_accuracy", "away_shots_box_ratio",
    "away_save_pct", "away_pass_accuracy_pct",
    "delta_shot_conversion_rate", "delta_shot_accuracy",
    "delta_shots_box_ratio", "delta_pass_accuracy_pct",

    # Таблиця
    "home_table_position", "home_table_points", "home_table_goal_diff",
    "away_table_position", "away_table_points", "away_table_goal_diff",
    "table_position_diff", "table_points_diff",

    # Дні відпочинку
    "home_rest_days", "away_rest_days", "rest_days_diff",
]


def get_league_map(db):
    from db.models import League
    return {l.id: LEAGUES[l.api_id] for l in db.query(League).all() if l.api_id in LEAGUES}


def train_xgb(X_train, y_train, X_val, y_val):
    base = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def simulate(bets_df: pd.DataFrame) -> dict:
    if bets_df.empty:
        return {"n": 0, "wins": 0, "win_rate": 0, "flat_roi": 0, "kelly_roi": 0, "bankroll": INITIAL_BR}
    bets = bets_df.sort_values("date").copy()
    bankroll, staked, flat_pnl = INITIAL_BR, 0.0, 0.0
    for _, b in bets.iterrows():
        p, odd = b["model_prob"], b["odds"]
        kelly = max(0.0, (p * (odd - 1) - (1 - p)) / (odd - 1)) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * MAX_STAKE)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked += stake
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wins": int(bets["won"].sum()), "win_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(), "avg_ev": bets["ev"].mean(),
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0,
        "bankroll": round(bankroll, 2),
    }


def show_results(label, bets_df):
    if bets_df.empty:
        logger.info(f"{label}: немає ставок")
        return
    r = simulate(bets_df)
    logger.info(
        f"\n{'='*65}\n  {label}\n{'='*65}\n"
        f"  Ставок: {r['n']}  |  WR: {r['win_rate']:.1%}  ({r['wins']}/{r['n']})\n"
        f"  Avg odds: {r['avg_odds']:.2f}  |  Avg EV: {r['avg_ev']:+.1%}\n"
        f"  Flat ROI: {r['flat_roi']:+.1f}%\n"
        f"  Kelly:    ROI {r['kelly_roi']:+.1f}%  |  банкрол ${r['bankroll']:.2f}"
    )
    if "league" in bets_df.columns:
        logger.info(f"\n  {'Ліга':<23} {'N':>6} {'WR%':>7} {'FlatROI':>10}")
        logger.info(f"  {'-'*48}")
        by_l = bets_df.groupby("league").apply(lambda g: pd.Series({
            "n": len(g), "wr": g["won"].mean(),
            "roi": ((g["won"]*(g["odds"]-1)) - (~g["won"])).sum()/len(g)*100,
        })).sort_values("roi", ascending=False)
        for league, row in by_l.iterrows():
            logger.info(f"  {league:<23} {int(row['n']):>6} {row['wr']:>7.1%} {row['roi']:>+10.1f}%")


def collect_bets(model, le, df, available):
    X = df[available].fillna(0)
    probs = model.predict_proba(X)
    cidx = {c: i for i, c in enumerate(le.classes_)}
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        h_odds = row.get("home_odds") or (1/row["market_home_prob"] if row.get("market_home_prob") else None)
        a_odds = row.get("away_odds") or (1/row["market_away_prob"] if row.get("market_away_prob") else None)
        if not h_odds or not a_odds:
            continue
        for side, odds in [("home", h_odds), ("away", a_odds)]:
            if not (MIN_ODDS <= odds <= MAX_ODDS):
                continue
            model_prob = probs[i][cidx[side]]
            ev = model_prob * odds - 1
            if ev < MIN_EV:
                continue
            records.append({
                "date": row["date"], "league": row.get("league", ""),
                "side": side, "actual": row["target"],
                "won": row["target"] == side,
                "odds": round(odds, 2), "model_prob": model_prob, "ev": ev,
            })
    return pd.DataFrame(records)


def fit_model(train_df, available, le):
    X = train_df[available].fillna(0)
    y = le.transform(train_df["target"])
    split = int(len(X) * 0.85)
    model = train_xgb(X.iloc[:split], y[:split], X.iloc[split:], y[split:])
    acc = (model.predict(X.iloc[split:]) == y[split:]).mean()
    logger.info(f"  Val accuracy: {acc:.3f}")
    return model


def run():
    from db.session import SessionLocal
    db = SessionLocal()
    try:
        league_map = get_league_map(db)
    finally:
        db.close()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую датасет...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset[dataset["league_id"].isin(league_map)].copy()
    dataset = dataset[dataset["market_home_prob"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)
    logger.info(f"Датасет: {len(dataset)} матчів")

    available = [f for f in FEATURE_COLS if f in dataset.columns]
    missing   = [f for f in FEATURE_COLS if f not in dataset.columns]
    if missing:
        logger.warning(f"Відсутні фічі: {missing}")
    logger.info(f"Фіч: {len(available)}")

    le = LabelEncoder()
    le.fit(["away", "draw", "home"])

    fold1_train = dataset[dataset["date"] < SPLIT_1].copy()
    fold1_test  = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    fold2_train = dataset[dataset["date"] < SPLIT_2].copy()
    fold2_test  = dataset[dataset["date"] >= SPLIT_2].copy()

    logger.info(f"\nFold 1: train={len(fold1_train)}, test={len(fold1_test)}")
    m1 = fit_model(fold1_train, available, le)
    bets1 = collect_bets(m1, le, fold1_test, available)

    logger.info(f"Fold 2: train={len(fold2_train)}, test={len(fold2_test)}")
    m2 = fit_model(fold2_train, available, le)
    bets2 = collect_bets(m2, le, fold2_test, available)

    show_results("FOLD 1 — 2H сезону 2024-25 (Jan–Jun 2025)", bets1)
    show_results("FOLD 2 — сезон 2025-26 (Jul 2025–зараз)", bets2)

    all_bets = pd.concat([bets1, bets2], ignore_index=True)
    show_results("AGGREGATE OOS", all_bets)

    # Feature importance з fold2 моделі
    try:
        booster = m2.calibrated_classifiers_[0].estimator
        scores = booster.get_booster().get_score(importance_type="gain")
        top = sorted(scores.items(), key=lambda x: -x[1])[:20]
        logger.info("\nТоп-20 фіч (Fold2 модель, gain):")
        for fname, gain in top:
            logger.info(f"  {fname:<42} {gain:.1f}")
    except Exception:
        pass

    os.makedirs("experiments/results", exist_ok=True)
    all_bets.to_csv("experiments/results/xgb_v3_oos_bets.csv", index=False)
    logger.info("\nЗбережено: experiments/results/xgb_v3_oos_bets.csv")


if __name__ == "__main__":
    run()
