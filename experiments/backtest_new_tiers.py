"""
experiments/backtest_new_tiers.py

Тестуємо два нових tier-и:
1. HOME tier: WS dominant=home, GAP≥65, odds 1.90-2.50
2. Strong favorite (AWAY): GAP≥100, odds 1.70-2.20
3. Порівняння з поточною конфігурацією (AWAY, GAP≥80, 2.20-2.60)

Запуск: python -m experiments.backtest_new_tiers
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATABASE_URL', 'postgresql://capper:capper@localhost:5432/capper')

import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0


def compute_ws(features: dict, outcome: str) -> float:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    total = 0.0
    for name, active in _get_factors(features, outcome):
        if active:
            total += weights.get(name, 1)
    return total


def collect_bets(dataset: pd.DataFrame, side_filter: str,
                 gap_min: float, odds_min: float, odds_max: float) -> pd.DataFrame:
    rows = []
    for _, row in dataset.iterrows():
        r = row.to_dict()
        ws_h = compute_ws(r, "home")
        ws_a = compute_ws(r, "away")

        if ws_h >= ws_a:
            dom_side, ws_dom, ws_weak = "home", ws_h, ws_a
            odds_val = r.get("home_odds")
            p_elo = float(r.get("elo_home_win_prob", 0.5))
        else:
            dom_side, ws_dom, ws_weak = "away", ws_a, ws_h
            odds_val = r.get("away_odds")
            p_elo = 1.0 - float(r.get("elo_home_win_prob", 0.5))

        if side_filter != "both" and dom_side != side_filter:
            continue
        if not odds_val:
            continue

        gap = ws_dom - ws_weak
        if gap < gap_min:
            continue
        if not (odds_min <= odds_val <= odds_max):
            continue

        rows.append({
            "side": dom_side,
            "odds": round(odds_val, 2),
            "ws_gap": gap,
            "p_elo": p_elo,
            "won": r["target"] == dom_side,
            "date": r["date"],
            "league": r.get("league", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
        })
    return pd.DataFrame(rows)


def simulate(bets: pd.DataFrame, kelly_cap: float = 0.04, fractional: float = 0.25) -> dict:
    if bets.empty:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "flat_roi": 0.0, "kelly_roi": 0.0, "bankroll": INITIAL_BR}
    bets = bets.sort_values("date")
    bankroll = INITIAL_BR
    staked = 0.0
    flat_pnl = 0.0
    for _, b in bets.iterrows():
        p, odd = b["p_elo"], b["odds"]
        q = 1.0 - p
        kelly = max(0.0, (p * (odd - 1) - q) / (odd - 1)) * fractional
        stake = min(bankroll * kelly, bankroll * kelly_cap)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked += stake
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wins": int(bets["won"].sum()),
        "win_rate": bets["won"].mean(),
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0,
        "bankroll": round(bankroll, 2),
    }


def print_tier(label, bets, f1, f2):
    s_all = simulate(bets)
    s_f1  = simulate(f1)
    s_f2  = simulate(f2)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Всього: {s_all['n']} бетів  WR={s_all['win_rate']:.1%}  FlatROI={s_all['flat_roi']:+.1f}%  BK=${s_all['bankroll']:.0f}")
    print(f"  F1 ({s_f1['n']:3d} bets): WR={s_f1['win_rate']:.1%}  FlatROI={s_f1['flat_roi']:+.1f}%  KellyROI={s_f1['kelly_roi']:+.1f}%  BK=${s_f1['bankroll']:.0f}")
    print(f"  F2 ({s_f2['n']:3d} bets): WR={s_f2['win_rate']:.1%}  FlatROI={s_f2['flat_roi']:+.1f}%  KellyROI={s_f2['kelly_roi']:+.1f}%  BK=${s_f2['bankroll']:.0f}")
    if not bets.empty:
        print(f"  По лігах:")
        for lg, g in bets.groupby("league"):
            n = len(g)
            wr = g["won"].mean()
            roi = ((g["won"] * (g["odds"] - 1)) - (~g["won"])).sum() / n * 100
            print(f"    {lg:<25} N={n:3d}  WR={wr:.1%}  FlatROI={roi:+.1f}%")


def run():
    logger.info("Loading data...")
    matches_df, stats_df, odds_df, teams, injuries_df = load_data_from_db()
    logger.info("Building dataset...")
    df = build_dataset(matches_df, stats_df, odds_df, teams, injuries_df)
    logger.info(f"Dataset: {len(df)} rows")

    df["date"] = pd.to_datetime(df["date"])
    f1_df = df[(df["date"] >= SPLIT_1) & (df["date"] < SPLIT_2)]
    f2_df = df[df["date"] >= SPLIT_2]

    configs = [
        # label, side, gap_min, odds_min, odds_max
        ("ПОТОЧНИЙ: AWAY, GAP≥80, 2.20-2.60",    "away", 80,  2.20, 2.60),
        ("ПОТОЧНИЙ: AWAY, GAP≥70, 2.00-inf",      "away", 70,  2.00, 9.9),
        # --- нові ---
        ("НОВИЙ: HOME, GAP≥65, 1.90-2.50",         "home", 65,  1.90, 2.50),
        ("НОВИЙ: HOME, GAP≥70, 1.90-2.50",         "home", 70,  1.90, 2.50),
        ("НОВИЙ: HOME, GAP≥80, 1.90-2.50",         "home", 80,  1.90, 2.50),
        ("НОВИЙ: AWAY GAP≥100, 1.70-2.20",         "away", 100, 1.70, 2.20),
        ("НОВИЙ: AWAY GAP≥100, 1.75-2.20",         "away", 100, 1.75, 2.20),
        ("НОВИЙ: AWAY GAP≥100, 1.80-2.10",         "away", 100, 1.80, 2.10),
        # --- комбо (обидві сторони) ---
        ("КОМБО: BOTH, GAP≥80, 1.90-2.60",         "both", 80,  1.90, 2.60),
        ("КОМБО: BOTH, GAP≥80, 2.00-2.60",         "both", 80,  2.00, 2.60),
    ]

    for label, side, gap, omin, omax in configs:
        bets = collect_bets(df,    side, gap, omin, omax)
        f1   = collect_bets(f1_df, side, gap, omin, omax)
        f2   = collect_bets(f2_df, side, gap, omin, omax)
        print_tier(label, bets, f1, f2)


if __name__ == "__main__":
    run()
