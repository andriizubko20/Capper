"""
model/gem/threshold_sweep.py

Sweep Gem filter thresholds on saved OOF predictions to find the optimal
yield × ROI tradeoff WITHOUT retraining.

Reads:  model/gem/artifacts/oof.npz   (oof_ensemble_calibrated, y, covered)
        model/gem/artifacts/info.parquet (per-match metadata + odds)

Outputs:
  reports/gem_threshold_sweep.csv
  console table sorted by efficiency_score = bets/year × roi × stable_factor
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ARTIFACTS = Path(__file__).parent / "artifacts"
REPORTS   = Path(__file__).parent / "reports"

# Sweep grid
MAX_DRAW_GRID = [0.25, 0.28, 0.30, 0.32, 0.35]
MIN_BET_GRID  = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]
MIN_GEM_GRID  = [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
ODDS_LO = 1.50
ODDS_HI = 3.00

HISTORY_YEARS = 32 / 12.0  # data spans ~32 months


def market_probs_devig(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float] | None:
    if not (home_odds and draw_odds and away_odds):
        return None
    raw = 1 / home_odds + 1 / draw_odds + 1 / away_odds
    if raw <= 0:
        return None
    return (1 / home_odds) / raw, (1 / draw_odds) / raw, (1 / away_odds) / raw


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def evaluate_combo(
    proba: np.ndarray, y: np.ndarray, covered: np.ndarray,
    home_odds: np.ndarray, draw_odds: np.ndarray, away_odds: np.ndarray,
    max_draw: float, min_bet: float, min_gem: float,
) -> dict:
    """Apply filter combo to OOF, return aggregate stats."""
    mask = covered & ~np.isnan(home_odds) & ~np.isnan(draw_odds) & ~np.isnan(away_odds)
    if not mask.any():
        return {"n_bets": 0, "wins": 0, "wr": 0, "roi": 0, "lo95": 0, "ev_avg": 0}

    pH = proba[:, 0]
    pD = proba[:, 1]
    pA = proba[:, 2]

    # Vectorized de-vig market probs
    raw_sum = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds
    mp_h = (1.0 / home_odds) / raw_sum
    mp_a = (1.0 / away_odds) / raw_sum

    # Pick side: argmax(P_H, P_A); default home on tie
    pick_home = pH >= pA
    p_side    = np.where(pick_home, pH, pA)
    odds_side = np.where(pick_home, home_odds, away_odds)
    mp_side   = np.where(pick_home, mp_h, mp_a)
    won_side  = np.where(pick_home, y == 0, y == 2)

    # Apply filter
    sel = (
        mask
        & (pD < max_draw)
        & (p_side > min_bet)
        & (odds_side >= ODDS_LO) & (odds_side <= ODDS_HI)
        & (p_side - mp_side > min_gem)
    )
    n = int(sel.sum())
    if n == 0:
        return {"n_bets": 0, "wins": 0, "wr": None, "roi": None, "lo95": 0, "ev_avg": None}

    won = won_side[sel].astype(int)
    odds = odds_side[sel]
    wins = int(won.sum())
    wr = wins / n
    pnl = (odds - 1) * won - (1 - won)
    roi = float(pnl.mean())
    ev_avg = float((p_side[sel] * odds - 1).mean())

    return {
        "n_bets":  n,
        "wins":    wins,
        "wr":      float(wr),
        "lo95":    wilson_lower(wins, n),
        "roi":     roi,
        "ev_avg":  ev_avg,
        "avg_odds": float(odds.mean()),
    }


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logger.info("Loading OOF + info …")
    npz = np.load(ARTIFACTS / "oof.npz")
    info = pd.read_parquet(ARTIFACTS / "info.parquet")

    proba   = npz["oof_ensemble_calibrated"]
    y       = npz["y"]
    covered = npz["covered"]

    home_odds = info["home_odds"].astype(float).to_numpy()
    draw_odds = info["draw_odds"].astype(float).to_numpy()
    away_odds = info["away_odds"].astype(float).to_numpy()

    logger.info(f"OOF: {covered.sum():,} covered rows, {len(home_odds):,} total")
    logger.info(f"Sweeping {len(MAX_DRAW_GRID)}×{len(MIN_BET_GRID)}×{len(MIN_GEM_GRID)} = "
                f"{len(MAX_DRAW_GRID)*len(MIN_BET_GRID)*len(MIN_GEM_GRID)} combos …")

    rows = []
    for max_draw in MAX_DRAW_GRID:
        for min_bet in MIN_BET_GRID:
            for min_gem in MIN_GEM_GRID:
                r = evaluate_combo(
                    proba, y, covered, home_odds, draw_odds, away_odds,
                    max_draw, min_bet, min_gem,
                )
                rows.append({
                    "max_draw": max_draw,
                    "min_bet":  min_bet,
                    "min_gem":  min_gem,
                    **r,
                })

    df = pd.DataFrame(rows)
    df = df[df["n_bets"] >= 10].copy()  # require min sample size
    df["bets_per_year"] = df["n_bets"] / HISTORY_YEARS
    df["annual_units"] = df["bets_per_year"] * df["roi"]

    # Score: prioritize HIGH wilson lower bound × volume × roi
    df["efficiency"] = df["lo95"] * df["bets_per_year"] * df["roi"].clip(lower=0)

    # Sort by efficiency (descending)
    df = df.sort_values("efficiency", ascending=False).reset_index(drop=True)

    df.to_csv(REPORTS / "gem_threshold_sweep.csv", index=False)

    print("\n" + "=" * 110)
    print("GEM THRESHOLD SWEEP — top 25 by efficiency_score")
    print("=" * 110)
    print(f"{'max_draw':>9s} {'min_bet':>8s} {'min_gem':>8s} | {'n':>5s} "
          f"{'WR':>6s} {'lo95':>6s} {'ROI':>7s} {'odds':>5s} {'/year':>6s} {'units/yr':>9s} {'eff':>7s}")
    print("-" * 110)
    for r in df.head(25).itertuples():
        print(
            f"{r.max_draw:>9.2f} {r.min_bet:>8.2f} {r.min_gem:>8.2f} | "
            f"{r.n_bets:>5d} {r.wr:>5.1%} {r.lo95:>5.1%} {r.roi:>+6.1%} "
            f"{r.avg_odds:>5.2f} {r.bets_per_year:>5.1f} {r.annual_units:>+8.2f} {r.efficiency:>+6.2f}"
        )

    # Show also: BEST yield with WR > 65% and ROI > +10%
    safe = df[(df["wr"] >= 0.65) & (df["roi"] >= 0.10)].copy()
    safe = safe.sort_values("n_bets", ascending=False).head(10)
    if not safe.empty:
        print("\n" + "=" * 110)
        print("HIGH-CONFIDENCE COMBOS (WR≥65%, ROI≥+10%) — sorted by yield (n)")
        print("=" * 110)
        print(f"{'max_draw':>9s} {'min_bet':>8s} {'min_gem':>8s} | {'n':>5s} "
              f"{'WR':>6s} {'lo95':>6s} {'ROI':>7s} {'/year':>6s}")
        for r in safe.itertuples():
            print(
                f"{r.max_draw:>9.2f} {r.min_bet:>8.2f} {r.min_gem:>8.2f} | "
                f"{r.n_bets:>5d} {r.wr:>5.1%} {r.lo95:>5.1%} {r.roi:>+6.1%} "
                f"{r.bets_per_year:>5.1f}"
            )

    # Current production thresholds (max_draw=0.32, min_bet=0.60, min_gem=0.05)
    cur = df[(df["max_draw"] == 0.32) & (df["min_bet"] == 0.60) & (df["min_gem"] == 0.05)]
    if not cur.empty:
        c = cur.iloc[0]
        print("\n" + "=" * 110)
        print(f"CURRENT PROD (0.32 / 0.60 / 0.05): "
              f"n={int(c.n_bets)} WR={c.wr:.1%} lo95={c.lo95:.1%} ROI={c.roi:+.1%} "
              f"yield={c.bets_per_year:.1f}/yr units/yr={c.annual_units:+.2f}")

    logger.info(f"Saved → {REPORTS}/gem_threshold_sweep.csv ({len(df)} rows)")


if __name__ == "__main__":
    run()
