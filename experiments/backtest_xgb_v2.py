"""
experiments/backtest_xgb_v2.py

XGBoost OOS backtest з розширеним набором фіч (v2).
Тренується на 2024-25, тестується на 2025-26.

Фічі v2 додають:
  - clean_sheet_rate, btts_rate, failed_to_score_rate, win/loss streak
  - elo_momentum (тренд Elo за 10 матчів)
  - market_certainty, market_home_edge

Виключені (потребують бекфілу, ще не заповнені):
  - shots_on_target, possession, corners, save_pct, pass_accuracy_pct

Запуск: python -m experiments.backtest_xgb_v2
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

# Walk-forward по двох сезонах
SPLIT_1 = pd.Timestamp("2025-01-01")   # середина сезону 2024-25
SPLIT_2 = pd.Timestamp("2025-07-01")   # початок сезону 2025-26
MIN_EV     = 0.20   # оптимальний поріг з ablation (+35% ROI)
MIN_ODDS   = 1.5
MAX_ODDS   = 2.5    # тільки фаворити — де модель реально предиктивна
MAX_STAKE  = 0.04   # макс % банкролю
FRACTIONAL = 0.25
INITIAL_BR = 1000.0

LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    94:  "Primeira Liga",
}

# ─── Оптимальний набір фіч з ablation (form_adv+elo+standings+rest+market_ext) ──
# Видалено: form_basic (-3.41% avg ROI), xg (малий внесок)
# Ablation result: EV≥20%, Odds≤2.5 → 144 bets, 61.8% WR, +35% flat ROI
FEATURE_COLS_V2 = [
    # Розширена форма (clean sheets, streaks, btts)
    "home_clean_sheet_rate", "home_btts_rate", "home_failed_to_score_rate",
    "home_win_streak", "home_loss_streak", "home_goals_for_std",
    "away_clean_sheet_rate", "away_btts_rate", "away_failed_to_score_rate",
    "away_win_streak", "away_loss_streak", "away_goals_for_std",
    "delta_clean_sheet_rate", "delta_btts_rate", "delta_failed_to_score_rate",

    # Elo
    "elo_diff", "elo_home_win_prob",
    "home_elo_momentum", "away_elo_momentum", "delta_elo_momentum",

    # Таблиця
    "home_table_position", "home_table_points", "home_table_goal_diff",
    "away_table_position", "away_table_points", "away_table_goal_diff",
    "table_position_diff", "table_points_diff",

    # Дні відпочинку
    "home_rest_days", "away_rest_days", "rest_days_diff",

    # Ринок (базові + розширені)
    "market_home_prob", "market_away_prob", "market_draw_prob",
    "market_certainty", "market_home_edge",
]


def get_league_db_ids(db):
    from db.models import League
    result = {}
    for api_id, name in LEAGUES.items():
        for l in db.query(League).filter_by(api_id=api_id).all():
            result[l.id] = name
    return result


def train_xgb(X_train, y_train, X_val, y_val):
    base = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,   # сильніша регуляризація
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    base.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    # Ізотонна калібрація — виправляє роздуті ймовірності XGBoost
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def simulate(bets_df: pd.DataFrame) -> dict:
    """Flat + Kelly симуляція."""
    bets = bets_df.sort_values("date").copy()
    bankroll = INITIAL_BR
    flat_pnl = 0.0
    staked   = 0.0

    for _, b in bets.iterrows():
        p   = b["model_prob"]
        odd = b["odds"]
        ev  = p * odd - 1

        # Kelly
        kelly = max(0.0, (p * (odd - 1) - (1 - p)) / (odd - 1)) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * MAX_STAKE)

        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked   += stake

        # Flat
        flat_pnl += (odd - 1) if b["won"] else -1.0

    n = len(bets)
    flat_roi  = flat_pnl / n * 100 if n > 0 else 0.0
    kelly_roi = (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0

    return {
        "n":          n,
        "wins":       int(bets["won"].sum()),
        "win_rate":   bets["won"].mean(),
        "avg_odds":   bets["odds"].mean(),
        "avg_ev":     bets["ev"].mean(),
        "flat_roi":   flat_roi,
        "kelly_roi":  kelly_roi,
        "bankroll":   round(bankroll, 2),
    }


def show_results(label: str, bets_df: pd.DataFrame, league_col: str = "league"):
    if bets_df.empty:
        logger.info(f"{label}: немає ставок")
        return

    r = simulate(bets_df)
    logger.info(
        f"\n{'='*65}\n  {label}\n{'='*65}\n"
        f"  Ставок:   {r['n']}  |  Win rate: {r['win_rate']:.1%}  ({r['wins']}/{r['n']})\n"
        f"  Avg odds: {r['avg_odds']:.2f}  |  Avg EV: {r['avg_ev']:+.1%}\n"
        f"  Flat ROI: {r['flat_roi']:+.1f}%\n"
        f"  Kelly:    ROI {r['kelly_roi']:+.1f}%  |  банкрол ${r['bankroll']:.2f}"
    )

    # По лігах
    if league_col in bets_df.columns:
        logger.info(f"\n  {'Ліга':<23} {'Ставок':>7} {'Win%':>7} {'Flat ROI':>10}")
        logger.info(f"  {'-'*50}")
        by_league = bets_df.groupby(league_col).apply(
            lambda g: pd.Series({
                "n": len(g),
                "win_rate": g["won"].mean(),
                "flat_roi": ((g["won"] * (g["odds"] - 1)) - (~g["won"])).sum() / len(g) * 100,
            })
        ).sort_values("flat_roi", ascending=False)
        for league, row in by_league.iterrows():
            logger.info(f"  {league:<23} {int(row['n']):>7} {row['win_rate']:>7.1%} {row['flat_roi']:>+10.1f}%")


def run():
    from db.session import SessionLocal
    db = SessionLocal()
    try:
        league_map = get_league_db_ids(db)
    finally:
        db.close()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую датасет...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset[dataset["league_id"].isin(league_map.keys())].copy()
    dataset = dataset[dataset["market_home_prob"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)

    logger.info(f"Датасет: {len(dataset)} матчів з odds")

    # Доступні фічі
    available = [f for f in FEATURE_COLS_V2 if f in dataset.columns]
    missing   = [f for f in FEATURE_COLS_V2 if f not in dataset.columns]
    if missing:
        logger.warning(f"Відсутні фічі (пропущені): {missing}")
    logger.info(f"Використовуємо {len(available)} фіч")

    le = LabelEncoder()
    le.fit(["away", "draw", "home"])

    def collect_bets(model, df: pd.DataFrame) -> pd.DataFrame:
        X = df[available].fillna(0)
        probs = model.predict_proba(X)
        class_idx = {c: i for i, c in enumerate(le.classes_)}
        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            h_prob_mkt = row["market_home_prob"]
            a_prob_mkt = row["market_away_prob"]
            if not h_prob_mkt or not a_prob_mkt:
                continue
            h_odds = 1 / h_prob_mkt
            a_odds = 1 / a_prob_mkt
            for side, mkt_prob, odds in [("home", h_prob_mkt, h_odds), ("away", a_prob_mkt, a_odds)]:
                if odds < MIN_ODDS or odds > MAX_ODDS:
                    continue
                model_prob = probs[i][class_idx[side]]
                ev = model_prob * odds - 1
                if ev < MIN_EV:
                    continue
                records.append({
                    "date": row["date"], "league": row.get("league", ""),
                    "side": side, "actual": row["target"],
                    "won": row["target"] == side,
                    "odds": round(odds, 2), "model_prob": model_prob,
                    "mkt_prob": mkt_prob, "ev": ev,
                    "fold": row.get("fold", ""),
                })
        return pd.DataFrame(records)

    def fit_model(train_df):
        X = train_df[available].fillna(0)
        y = le.transform(train_df["target"])
        split = int(len(X) * 0.85)
        model = train_xgb(X.iloc[:split], y[:split], X.iloc[split:], y[split:])
        acc = (model.predict(X.iloc[split:]) == y[split:]).mean()
        logger.info(f"  Val accuracy: {acc:.3f}")
        return model

    # ─── Fold 1: train на першій половині 2024-25, test на другій ────────────
    fold1_train = dataset[dataset["date"] < SPLIT_1].copy()
    fold1_test  = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    fold1_test["fold"] = "Fold1 (2H 2024-25)"

    logger.info(f"\nFold 1: train={len(fold1_train)}, test={len(fold1_test)}")
    model1 = fit_model(fold1_train)
    bets1  = collect_bets(model1, fold1_test)

    # ─── Fold 2: train на всій 2024-25, test на 2025-26 ──────────────────────
    fold2_train = dataset[dataset["date"] < SPLIT_2].copy()
    fold2_test  = dataset[dataset["date"] >= SPLIT_2].copy()
    fold2_test["fold"] = "Fold2 (2025-26)"

    logger.info(f"Fold 2: train={len(fold2_train)}, test={len(fold2_test)}")
    model2 = fit_model(fold2_train)
    bets2  = collect_bets(model2, fold2_test)

    # ─── Результати ──────────────────────────────────────────────────────────
    show_results("FOLD 1 OOS — 2H сезону 2024-25 (Jan–Jun 2025)", bets1)
    show_results("FOLD 2 OOS — сезон 2025-26 (Jul 2025–зараз)",   bets2)

    all_bets = pd.concat([bets1, bets2], ignore_index=True)
    show_results("AGGREGATE OOS — обидва сезони разом", all_bets)

    # Baseline comparison
    n = len(all_bets)
    if n > 0:
        wr  = all_bets["won"].mean()
        roi = ((all_bets["won"] * (all_bets["odds"] - 1)) - (~all_bets["won"])).sum() / n * 100
        logger.info(
            f"\n{'='*65}\n  ПОРІВНЯННЯ З BASELINE (WS Gap OOS)\n{'='*65}"
            f"\n  Baseline WS Gap:  53.5% win rate | -0.6% flat ROI | 535 ставок"
            f"\n  XGBoost v2 aggr:  {wr:.1%} win rate | {roi:+.1f}% flat ROI | {n} ставок"
        )

    # ─── Зберігаємо ──────────────────────────────────────────────────────────
    os.makedirs("experiments/results", exist_ok=True)
    all_bets.to_csv("experiments/results/xgb_v2_oos_bets.csv", index=False)
    logger.info("\nЗбережено: experiments/results/xgb_v2_oos_bets.csv")


if __name__ == "__main__":
    run()
