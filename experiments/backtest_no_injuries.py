"""
experiments/backtest_no_injuries.py

Порівнює результати з injury-факторами і без них.
Мета: перевірити чи injuries — реальний сигнал або шум/нестабільні дані.

Injury-фактори:
  AWAY: injury_adv_away (+1), big_injury_adv_away (+10)
  HOME: injury_advantage (+1), big_injury_advantage (+1)

Запуск: python -m experiments.backtest_no_injuries
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATABASE_URL', 'postgresql://capper:capper@localhost:5432/capper')

import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, AWAY_WEIGHTS, HOME_WEIGHTS

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0
FRAC       = 0.25
CAP        = 0.04
ODDS_MAX   = 3.0

TOP5 = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

INJURY_FACTORS = {
    "injury_adv_away", "big_injury_adv_away",   # AWAY
    "injury_advantage", "big_injury_advantage",  # HOME
}

# Модифіковані ваги: injuries = 0
AWAY_NO_INJ = {k: (0 if k in INJURY_FACTORS else v) for k, v in AWAY_WEIGHTS.items()}
HOME_NO_INJ = {k: (0 if k in INJURY_FACTORS else v) for k, v in HOME_WEIGHTS.items()}

GRID_CONFIGS = [
    # (label, odds_min, gap_min)
    ("ODDS≥2.20 GAP≥90",  2.20, 90),
    ("ODDS≥2.20 GAP≥80",  2.20, 80),
    ("ODDS≥2.10 GAP≥90",  2.10, 90),
    ("ODDS≥2.10 GAP≥110", 2.10, 110),
    ("ODDS≥2.00 GAP≥110", 2.00, 110),
]


def compute_ws(features: dict, outcome: str, no_injuries: bool = False) -> float:
    if no_injuries:
        weights = AWAY_NO_INJ if outcome == "away" else HOME_NO_INJ
    else:
        weights = AWAY_WEIGHTS if outcome == "away" else HOME_WEIGHTS
    return sum(weights.get(n, 1) for n, active in _get_factors(features, outcome) if active)


def simulate(bets: list) -> dict:
    if not bets:
        return {"n": 0, "wr": 0.0, "flat_roi": 0.0,
                "kelly_roi": 0.0, "bankroll": INITIAL_BR, "max_dd": 0.0}
    bankroll = INITIAL_BR
    peak = INITIAL_BR
    max_dd = 0.0
    staked = 0.0
    flat_pnl = 0.0
    wins = 0
    for b in sorted(bets, key=lambda x: x["date"]):
        p, odd = b["p_elo"], b["odds"]
        q = 1 - p
        denom = odd - 1
        kelly = max(0.0, (p * denom - q) / denom) * FRAC if denom > 0 else 0
        stake = min(bankroll * kelly, bankroll * CAP)
        if stake > 0 and bankroll > 0:
            if b["won"]:
                bankroll += stake * (odd - 1); wins += 1
            else:
                bankroll -= stake
            staked += stake
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak * 100)
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wr": wins / n,
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0,
        "bankroll": round(bankroll, 2),
        "max_dd": round(max_dd, 1),
    }


def run():
    logger.info("Loading data...")
    matches_df, stats_df, odds_df, teams, injuries_df = load_data_from_db()
    logger.info("Building dataset...")
    df = build_dataset(matches_df, stats_df, odds_df, teams, injuries_df)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Dataset: {len(df)} rows")

    from db.session import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT m.id, l.name FROM matches m JOIN leagues l ON l.id = m.league_id"
        )).fetchall()
    league_map = {r[0]: r[1] for r in rows}

    logger.info("Pre-computing WS (with and without injuries)...")
    records = []
    injury_contributed = 0  # бети де injury підняли GAP вище порогу

    for _, row in df.iterrows():
        r = row.to_dict()

        # З injuries
        ws_h = compute_ws(r, "home", no_injuries=False)
        ws_a = compute_ws(r, "away", no_injuries=False)

        # Без injuries
        ws_h_ni = compute_ws(r, "home", no_injuries=True)
        ws_a_ni = compute_ws(r, "away", no_injuries=True)

        # Базовий фільтр: AWAY dominant з injuries
        if ws_a <= ws_h:
            continue

        gap = ws_a - ws_h
        gap_ni = ws_a_ni - ws_h_ni  # gap без injuries

        a_odds = r.get("away_odds")
        if not a_odds or a_odds > ODDS_MAX:
            continue

        mid = int(r.get("match_id", 0))
        league = league_map.get(mid, "")
        if league not in TOP5:
            continue

        p_elo = 1.0 - float(r.get("elo_home_win_prob", 0.5))

        records.append({
            "date":    row["date"],
            "odds":    round(a_odds, 2),
            "gap":     gap,
            "gap_ni":  gap_ni,          # gap without injuries
            "away_dominant_ni": ws_a_ni > ws_h_ni,  # чи AWAY все ще dominant без injuries
            "p_elo":   p_elo,
            "won":     r.get("target") == "away",
            "inj_boost": gap - gap_ni,  # скільки додали injuries
        })

    all_df = pd.DataFrame(records).sort_values("date")
    f1_df  = all_df[(all_df["date"] >= SPLIT_1) & (all_df["date"] < SPLIT_2)]
    f2_df  = all_df[all_df["date"] >= SPLIT_2]
    logger.info(f"Base pool (AWAY dominant with injuries): {len(all_df)}")

    # Статистика впливу injuries
    inj_10plus = (all_df["inj_boost"] >= 10).sum()
    flipped = (~all_df["away_dominant_ni"]).sum()
    logger.info(f"Injury boost ≥10 pts: {inj_10plus} ({inj_10plus/len(all_df)*100:.1f}%)")
    logger.info(f"AWAY dominant only because of injuries (flipped): {flipped} ({flipped/len(all_df)*100:.1f}%)")

    print(f"\n{'='*95}")
    print(f"  INJURY IMPACT STATS (base pool: {len(all_df)} records)")
    print(f"  Injury boost distribution:")
    for threshold in [0, 1, 5, 10, 11, 20]:
        n = (all_df["inj_boost"] >= threshold).sum()
        print(f"    boost >= {threshold:3d}: {n:5d} ({n/len(all_df)*100:.1f}%)")
    print(f"  Flipped to AWAY by injuries only: {flipped} ({flipped/len(all_df)*100:.1f}%)")
    print(f"  Win rate of 'flipped' bets: {all_df[~all_df['away_dominant_ni']]['won'].mean():.1%}")

    # --- GRID: WITH vs WITHOUT injuries ---
    print(f"\n{'='*95}")
    print(f"  GRID: WITH injuries vs WITHOUT injuries")
    print(f"  {'Config':<22} {'Mode':<10} | {'N_all':>6} {'N_F1':>5} {'N_F2':>5} | "
          f"{'F1_ROI':>8} {'F2_ROI':>8} {'Avg':>8} | {'KellyF2':>8} {'MaxDD':>7}")
    print(f"  {'-'*90}")

    results = []
    for label, odds_min, gap_min in GRID_CONFIGS:
        for mode, use_df, use_f1, use_f2 in [
            ("WITH inj",    all_df, f1_df, f2_df),
            ("NO  inj",
             all_df[all_df["away_dominant_ni"] & (all_df["gap_ni"] >= gap_min)],
             f1_df[f1_df["away_dominant_ni"] & (f1_df["gap_ni"] >= gap_min)],
             f2_df[f2_df["away_dominant_ni"] & (f2_df["gap_ni"] >= gap_min)]),
        ]:
            def filt(d, gmin=gap_min, omin=odds_min):
                col = "gap" if mode == "WITH inj" else "gap_ni"
                return d[(d[col] >= gmin) & (d["odds"] >= omin)].to_dict("records")

            bets_all = filt(use_df)
            bets_f1  = filt(use_f1)
            bets_f2  = filt(use_f2)

            s_all = simulate(bets_all)
            s_f1  = simulate(bets_f1)
            s_f2  = simulate(bets_f2)
            avg   = (s_f1["flat_roi"] + s_f2["flat_roi"]) / 2

            marker = " ✓" if s_f2["flat_roi"] >= 10 and s_f2["n"] >= 15 else ""
            print(f"  {label:<22} {mode:<10} | "
                  f"{s_all['n']:>6} {s_f1['n']:>5} {s_f2['n']:>5} | "
                  f"{s_f1['flat_roi']:>+8.1f}% {s_f2['flat_roi']:>+8.1f}% {avg:>+8.1f}% | "
                  f"{s_f2['kelly_roi']:>+8.1f}% {s_f2['max_dd']:>6.1f}%{marker}")

            results.append({
                "config": label, "mode": mode,
                "N_all": s_all["n"], "N_F1": s_f1["n"], "N_F2": s_f2["n"],
                "F1_%": round(s_f1["flat_roi"], 1), "F2_%": round(s_f2["flat_roi"], 1),
                "Avg_%": round(avg, 1), "KellyF2_%": round(s_f2["kelly_roi"], 1),
                "MaxDD_%": s_f2["max_dd"],
            })
        print()

    pd.DataFrame(results).to_csv("experiments/results/no_injuries_grid.csv", index=False)
    logger.info("Saved: experiments/results/no_injuries_grid.csv")


if __name__ == "__main__":
    run()
