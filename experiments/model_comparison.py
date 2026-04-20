"""
experiments/model_comparison.py

Порівняння WS Gap vs XGBoost:
  - WS Config A: GAP=80, ODDS=2.2-2.5
  - WS Config B: GAP=100, ODDS=2.2-4.0
  - XGBoost: optimal features з ablation
  - Аналіз перетину: коли обидві моделі погоджуються → чи краще?

Запуск: python -m experiments.model_comparison
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
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0
FRACTIONAL = 0.25
KELLY_CAP  = 0.04

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}

# ─── WS конфіги ───────────────────────────────────────────────────────────────
WS_CONFIGS = {
    "WS_A (GAP=80, 2.2-2.5)": {"gap": 80, "omin": 2.2, "omax": 2.5},
    "WS_B (GAP=100, 2.2-4.0)": {"gap": 100, "omin": 2.2, "omax": 4.0},
}

# ─── XGBoost feature set (slim, optimal з ablation) ───────────────────────────
XGB_FEATURES = [
    "home_clean_sheet_rate", "home_btts_rate", "home_failed_to_score_rate",
    "home_win_streak", "home_loss_streak", "home_goals_for_std",
    "away_clean_sheet_rate", "away_btts_rate", "away_failed_to_score_rate",
    "away_win_streak", "away_loss_streak", "away_goals_for_std",
    "delta_clean_sheet_rate", "delta_btts_rate", "delta_failed_to_score_rate",
    "elo_diff", "elo_home_win_prob",
    "home_elo_momentum", "away_elo_momentum", "delta_elo_momentum",
    "home_table_position", "home_table_points", "home_table_goal_diff",
    "away_table_position", "away_table_points", "away_table_goal_diff",
    "table_position_diff", "table_points_diff",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "market_home_prob", "market_away_prob", "market_draw_prob",
    "market_certainty", "market_home_edge",
]
XGB_MIN_EV   = 0.20
XGB_MIN_ODDS = 1.5
XGB_MAX_ODDS = 2.5


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_ws(features: dict, outcome: str) -> float:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    return sum(weights.get(n, 1) for n, active in _get_factors(features, outcome) if active)


def precompute_ws(dataset: pd.DataFrame) -> pd.DataFrame:
    """Рахуємо ws_home, ws_away, ws_gap один раз для всього датасету."""
    ws_h, ws_a = [], []
    for _, row in dataset.iterrows():
        d = row.to_dict()
        ws_h.append(compute_ws(d, "home"))
        ws_a.append(compute_ws(d, "away"))
    dataset = dataset.copy()
    dataset["ws_home"] = ws_h
    dataset["ws_away"] = ws_a
    dataset["ws_dominant"] = dataset[["ws_home", "ws_away"]].max(axis=1)
    dataset["ws_side"] = (dataset["ws_home"] >= dataset["ws_away"]).map({True: "home", False: "away"})
    dataset["ws_gap"] = dataset["ws_home"] - dataset["ws_away"]
    dataset["ws_gap_abs"] = dataset["ws_gap"].abs()
    dataset["ws_odds"] = dataset.apply(
        lambda r: r["home_odds"] if r["ws_side"] == "home" else r["away_odds"], axis=1
    )
    dataset["ws_p_elo"] = dataset.apply(
        lambda r: r["elo_home_win_prob"] if r["ws_side"] == "home"
                  else 1.0 - r["elo_home_win_prob"], axis=1
    )
    dataset["ws_won"] = dataset["ws_side"] == dataset["target"]
    return dataset


def ws_bets(dataset: pd.DataFrame, gap: float, omin: float, omax: float) -> pd.DataFrame:
    mask = (
        (dataset["ws_gap_abs"] >= gap) &
        (dataset["ws_odds"] >= omin) &
        (dataset["ws_odds"] <= omax) &
        dataset["ws_odds"].notna()
    )
    cols = ["date", "league", "ws_side", "ws_odds", "ws_won",
            "ws_gap_abs", "ws_p_elo", "target", "match_id"]
    return dataset[mask][cols].rename(columns={
        "ws_side": "side", "ws_odds": "odds", "ws_won": "won", "ws_p_elo": "p_elo"
    }).copy()


def simulate(bets: pd.DataFrame, kelly_cap=KELLY_CAP, frac=FRACTIONAL) -> dict:
    if bets.empty:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "flat_roi": 0.0,
                "kelly_roi": 0.0, "bankroll": INITIAL_BR}
    bets = bets.sort_values("date")
    bankroll, staked, flat_pnl = INITIAL_BR, 0.0, 0.0
    for _, b in bets.iterrows():
        p, odd = b["p_elo"], b["odds"]
        kelly = max(0.0, (p * (odd - 1) - (1 - p)) / (odd - 1)) * frac
        stake = min(bankroll * kelly, bankroll * kelly_cap)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked += stake
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {"n": n, "wins": int(bets["won"].sum()), "win_rate": bets["won"].mean(),
            "flat_roi": flat_pnl / n * 100,
            "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0,
            "bankroll": round(bankroll, 2)}


def show_league_table(bets: pd.DataFrame):
    if bets.empty:
        return
    logger.info(f"  {'Ліга':<23} {'N':>5} {'WR%':>7} {'FlatROI':>9}")
    logger.info(f"  {'-'*47}")
    for league, g in bets.groupby("league"):
        n = len(g)
        wr = g["won"].mean()
        roi = ((g["won"] * (g["odds"] - 1)) - (~g["won"])).sum() / n * 100
        logger.info(f"  {league:<23} {n:>5} {wr:>7.1%} {roi:>+9.1f}%")


def train_xgb(X_train, y_train, X_val, y_val):
    base = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                         eval_metric="mlogloss", random_state=42, verbosity=0)
    base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def xgb_bets(model, le, dataset: pd.DataFrame, available: list) -> pd.DataFrame:
    X = dataset[available].fillna(0)
    probs = model.predict_proba(X)
    cidx = {c: i for i, c in enumerate(le.classes_)}
    records = []
    for i, (_, row) in enumerate(dataset.iterrows()):
        h_prob = row.get("market_home_prob")
        a_prob = row.get("market_away_prob")
        if not h_prob or not a_prob:
            continue
        h_odds = row.get("home_odds") or (1 / h_prob)
        a_odds = row.get("away_odds") or (1 / a_prob)
        for side, odds in [("home", h_odds), ("away", a_odds)]:
            if not (XGB_MIN_ODDS <= odds <= XGB_MAX_ODDS):
                continue
            model_prob = probs[i][cidx[side]]
            ev = model_prob * odds - 1
            if ev < XGB_MIN_EV:
                continue
            records.append({
                "date": row["date"], "league": row.get("league", ""),
                "side": side, "odds": round(odds, 2),
                "won": row["target"] == side,
                "p_elo": model_prob,   # використовуємо model prob для Kelly
                "match_id": row.get("match_id"),
                "ev": ev,
            })
    return pd.DataFrame(records)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    from db.session import SessionLocal
    from db.models import League

    db = SessionLocal()
    try:
        league_map = {l.id: LEAGUES[l.api_id] for l in db.query(League).all() if l.api_id in LEAGUES}
    finally:
        db.close()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую датасет...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset[dataset["league_id"].isin(league_map)].copy()
    dataset = dataset[dataset["home_odds"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)
    logger.info(f"Датасет: {len(dataset)} матчів")

    logger.info("Precompute WS scores...")
    dataset = precompute_ws(dataset)

    fold1 = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    fold2 = dataset[dataset["date"] >= SPLIT_2].copy()

    os.makedirs("experiments/results", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. WS конфіги — детальний розріз по лігах
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  WS GAP — ДЕТАЛЬНИЙ РОЗРІЗ ПО ЛІГАХ")
    print(f"{'='*70}")

    ws_bets_store = {}
    for cname, cfg in WS_CONFIGS.items():
        b1 = ws_bets(fold1, cfg["gap"], cfg["omin"], cfg["omax"])
        b2 = ws_bets(fold2, cfg["gap"], cfg["omin"], cfg["omax"])
        ws_bets_store[cname] = {"f1": b1, "f2": b2}

        for label, bets in [("FOLD 1 (Jan–Jun 2025)", b1), ("FOLD 2 (Jul 2025+)", b2)]:
            r = simulate(bets)
            logger.info(f"\n  {cname} | {label}")
            logger.info(f"  {r['n']} ставок | WR {r['win_rate']:.1%} | "
                        f"Flat ROI {r['flat_roi']:>+.1f}% | Kelly ROI {r['kelly_roi']:>+.1f}%")
            show_league_table(bets)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. XGBoost — тренуємо і збираємо беті
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\nXGBoost: тренування...")
    available = [f for f in XGB_FEATURES if f in dataset.columns]
    missing = [f for f in XGB_FEATURES if f not in dataset.columns]
    if missing:
        logger.warning(f"Відсутні XGB фічі: {missing}")

    le = LabelEncoder()
    le.fit(["away", "draw", "home"])

    train1 = dataset[dataset["date"] < SPLIT_1].copy()
    train2 = dataset[dataset["date"] < SPLIT_2].copy()

    def fit(train_df):
        X = train_df[available].fillna(0)
        y = le.transform(train_df["target"])
        split = int(len(X) * 0.85)
        m = train_xgb(X.iloc[:split], y[:split], X.iloc[split:], y[split:])
        logger.info(f"  Val acc: {(m.predict(X.iloc[split:]) == y[split:]).mean():.3f}")
        return m

    m1 = fit(train1)
    m2 = fit(train2)

    xb1 = xgb_bets(m1, le, fold1, available)
    xb2 = xgb_bets(m2, le, fold2, available)

    print(f"\n{'='*70}")
    print("  XGBoost — РОЗРІЗ ПО ЛІГАХ")
    print(f"{'='*70}")
    for label, bets in [("FOLD 1", xb1), ("FOLD 2", xb2)]:
        r = simulate(bets)
        logger.info(f"\n  XGBoost | {label}: {r['n']} ставок | WR {r['win_rate']:.1%} | "
                    f"Flat ROI {r['flat_roi']:>+.1f}% | Kelly ROI {r['kelly_roi']:>+.1f}%")
        show_league_table(bets)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ПОРІВНЯННЯ: WS_A vs XGBoost — overlap аналіз
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  OVERLAP АНАЛІЗ: WS_A vs XGBoost")
    print(f"{'='*70}")

    for label, ws_b, xg_b in [("FOLD 1", ws_bets_store["WS_A (GAP=80, 2.2-2.5)"]["f1"], xb1),
                               ("FOLD 2", ws_bets_store["WS_A (GAP=80, 2.2-2.5)"]["f2"], xb2)]:
        if ws_b.empty or xg_b.empty:
            continue

        ws_ids  = set(zip(ws_b["match_id"], ws_b["side"]))
        xg_ids  = set(zip(xg_b["match_id"], xg_b["side"]))
        both    = ws_ids & xg_ids
        ws_only = ws_ids - xg_ids
        xg_only = xg_ids - ws_ids

        def subset(bets, id_set):
            return bets[bets.apply(lambda r: (r["match_id"], r["side"]) in id_set, axis=1)]

        ws_both = subset(ws_b, both)
        xg_both = subset(xg_b, both)
        ws_solo = subset(ws_b, ws_only)
        xg_solo = subset(xg_b, xg_only)

        logger.info(f"\n  {label}:")
        logger.info(f"  {'Сегмент':<25} {'N':>5} {'WR%':>7} {'FlatROI':>9}")
        logger.info(f"  {'-'*48}")
        for seg, bets in [
            ("WS тільки",       ws_solo),
            ("XGB тільки",      xg_solo),
            ("Обидві (WS)",     ws_both),
            ("Обидві (XGB)",    xg_both),
            ("WS всі",          ws_b),
            ("XGB всі",         xg_b),
        ]:
            if bets.empty:
                logger.info(f"  {seg:<25} {'—':>5}")
                continue
            r = simulate(bets)
            logger.info(f"  {seg:<25} {r['n']:>5} {r['win_rate']:>7.1%} {r['flat_roi']:>+9.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ЗВЕДЕНА ТАБЛИЦЯ
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ЗВЕДЕНА ТАБЛИЦЯ — F1 vs F2 vs Avg")
    print(f"{'='*80}")
    print(f"  {'Модель':<28} {'F1 N':>5} {'F1 WR':>6} {'F1 ROI':>8} | "
          f"{'F2 N':>5} {'F2 WR':>6} {'F2 ROI':>8} | {'Avg':>7}")
    print(f"  {'-'*77}")

    summary = []
    for cname, store in ws_bets_store.items():
        r1, r2 = simulate(store["f1"]), simulate(store["f2"])
        summary.append((cname, r1, r2))
    summary.append(("XGBoost (slim, EV≥20%)", simulate(xb1), simulate(xb2)))

    for name, r1, r2 in summary:
        avg = (r1["flat_roi"] + r2["flat_roi"]) / 2
        print(f"  {name:<28} {r1['n']:>5} {r1['win_rate']:>6.1%} {r1['flat_roi']:>+8.1f}% | "
              f"{r2['n']:>5} {r2['win_rate']:>6.1%} {r2['flat_roi']:>+8.1f}% | {avg:>+7.1f}%")

    logger.info("\nГотово.")


if __name__ == "__main__":
    run()
