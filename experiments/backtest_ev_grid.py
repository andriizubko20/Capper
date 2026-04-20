"""
experiments/backtest_ev_grid.py

Грід по EV порогах + розширена лінія коефіцієнтів 1.65-2.60.
EV = p_elo * odds - 1

Запуск: python -m experiments.backtest_ev_grid
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

TOP5 = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

# Грід параметрів
GAP_VALS     = [70, 80, 90]
ODDS_MIN_VALS = [1.65, 1.75, 1.90, 2.10, 2.20]
ODDS_MAX      = 2.60
EV_MIN_VALS   = [None, 0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]


def compute_ws(features: dict, outcome: str) -> float:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    return sum(weights.get(n, 1) for n, active in _get_factors(features, outcome) if active)


def simulate(bets: list, kelly_cap=CAP, frac=FRAC) -> dict:
    if not bets:
        return {"n": 0, "wins": 0, "wr": 0.0, "flat_roi": 0.0,
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
        kelly = max(0.0, (p * (odd - 1) - q) / (odd - 1)) * frac
        stake = min(bankroll * kelly, bankroll * kelly_cap)
        if stake > 0 and bankroll > 0:
            if b["won"]:
                bankroll += stake * (odd - 1); wins += 1
            else:
                bankroll -= stake
            staked += stake
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak * 100
            max_dd = max(max_dd, dd)
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wins": wins, "wr": wins / n,
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

    # Load match info for league names
    from db.session import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT m.id, l.name FROM matches m
            JOIN leagues l ON l.id = m.league_id
        """)).fetchall()
    league_map = {r[0]: r[1] for r in rows}

    # Pre-compute WS and p_elo for all rows
    logger.info("Pre-computing WS scores...")
    records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        ws_h = compute_ws(r, "home")
        ws_a = compute_ws(r, "away")
        if ws_a <= ws_h:
            continue  # only AWAY dominant

        gap = ws_a - ws_h
        a_odds = r.get("away_odds")
        if not a_odds:
            continue

        p_elo = 1.0 - float(r.get("elo_home_win_prob", 0.5))
        ev = p_elo * a_odds - 1.0

        mid = int(r.get("match_id", 0))
        league = league_map.get(mid, "")
        if league not in TOP5:
            continue

        records.append({
            "date":   row["date"],
            "league": league,
            "odds":   round(a_odds, 2),
            "ws_gap": gap,
            "p_elo":  round(p_elo, 4),
            "ev":     round(ev, 4),
            "won":    r.get("target") == "away",
        })

    all_df = pd.DataFrame(records)
    f1_df  = all_df[(all_df["date"] >= SPLIT_1) & (all_df["date"] < SPLIT_2)]
    f2_df  = all_df[all_df["date"] >= SPLIT_2]
    logger.info(f"Total AWAY dominant Top-5 records: {len(all_df)}")

    # --- GRID ---
    results = []
    for gap_min in GAP_VALS:
        for odds_min in ODDS_MIN_VALS:
            for ev_min in EV_MIN_VALS:
                def filt(d):
                    m = d[(d["ws_gap"] >= gap_min) & (d["odds"] >= odds_min) & (d["odds"] <= ODDS_MAX)]
                    if ev_min is not None:
                        m = m[m["ev"] >= ev_min]
                    return m.to_dict("records")

                bets_all = filt(all_df)
                bets_f1  = filt(f1_df)
                bets_f2  = filt(f2_df)

                s_all = simulate(bets_all)
                s_f1  = simulate(bets_f1)
                s_f2  = simulate(bets_f2)

                if s_all["n"] < 10:
                    continue

                avg_roi = (s_f1["flat_roi"] + s_f2["flat_roi"]) / 2
                results.append({
                    "gap":       gap_min,
                    "odds_min":  odds_min,
                    "ev_min":    ev_min if ev_min is not None else -99,
                    "n_all":     s_all["n"],
                    "wr_all":    s_all["wr"],
                    "flat_all":  s_all["flat_roi"],
                    "bk_all":    s_all["bankroll"],
                    "n_f1":      s_f1["n"],
                    "wr_f1":     s_f1["wr"],
                    "flat_f1":   s_f1["flat_roi"],
                    "n_f2":      s_f2["n"],
                    "wr_f2":     s_f2["wr"],
                    "flat_f2":   s_f2["flat_roi"],
                    "kelly_f1":  s_f1["kelly_roi"],
                    "kelly_f2":  s_f2["kelly_roi"],
                    "avg_roi":   avg_roi,
                    "dd_f2":     s_f2["max_dd"],
                })

    res_df = pd.DataFrame(results).sort_values("avg_roi", ascending=False)

    # --- Print top results ---
    print(f"\n{'='*110}")
    print(f"  EV GRID — GAP×ODDS_MIN×EV_MIN | AWAY, Top-5, 2.10-2.60 ODDS_MAX")
    print(f"{'='*110}")
    print(f"  {'GAP':>5} {'OMin':>6} {'EVMin':>7} | "
          f"{'All N':>6} {'AllWR':>6} {'AllROI':>8} | "
          f"{'F1 N':>5} {'F1WR':>6} {'F1ROI':>8} | "
          f"{'F2 N':>5} {'F2WR':>6} {'F2ROI':>8} | "
          f"{'AvgROI':>8} {'F2DD':>6} {'BK$':>7}")
    print(f"  {'-'*106}")

    for _, r in res_df.head(40).iterrows():
        ev_str = f"{r['ev_min']:.2f}" if r['ev_min'] != -99 else " none"
        print(f"  {int(r['gap']):>5} {r['odds_min']:>6.2f} {ev_str:>7} | "
              f"  {int(r['n_all']):>4} {r['wr_all']:>6.1%} {r['flat_all']:>+8.1f}% | "
              f"  {int(r['n_f1']):>3} {r['wr_f1']:>6.1%} {r['flat_f1']:>+8.1f}% | "
              f"  {int(r['n_f2']):>3} {r['wr_f2']:>6.1%} {r['flat_f2']:>+8.1f}% | "
              f"  {r['avg_roi']:>+7.1f}% {r['dd_f2']:>5.1f}% ${r['bk_all']:>6.0f}")

    # --- Best configs per odds_min ---
    print(f"\n{'='*75}")
    print(f"  BEST CONFIG PER ODDS_MIN (за avg F1+F2 ROI)")
    print(f"{'='*75}")
    for omin, grp in res_df.groupby("odds_min"):
        best = grp.iloc[0]
        ev_str = f"{best['ev_min']:.2f}" if best['ev_min'] != -99 else "none"
        print(f"  ODDS≥{omin:.2f}  GAP≥{int(best['gap'])}  EV≥{ev_str}  "
              f"N={int(best['n_all'])}  F1={best['flat_f1']:+.1f}%  F2={best['flat_f2']:+.1f}%  "
              f"avg={best['avg_roi']:+.1f}%  BK=${best['bk_all']:.0f}")

    # Save CSV
    res_df.to_csv("experiments/results/ev_grid_results.csv", index=False)
    logger.info("Saved: experiments/results/ev_grid_results.csv")


if __name__ == "__main__":
    run()
