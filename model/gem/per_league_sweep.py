"""
model/gem/per_league_sweep.py

Per-league threshold sweep on Gem OOF predictions.

For each league with enough data:
  Sweep (MAX_DRAW_PROB, MIN_BET_PROB, MIN_GEM_SCORE) × (proportional, shin)
  Pick the combo that maximises annual_units (= bets/year × ROI)
  subject to:
    - n_bets >= MIN_N_LEAGUE (sample-size guard, default 12)
    - WR >= 0.55 (don't accept high-volume losing combos)
    - ROI >= +5% (positive EV after closing-odds bias)

Saves: model/gem/artifacts/per_league_thresholds.json
       reports/per_league_sweep.csv

Each entry:
  {
    "League Name": {
      "devig":          "proportional" | "shin",
      "max_draw_prob":  float,
      "min_bet_prob":   float,
      "min_gem_score":  float,
      "n_bets_24mo":    int,
      "wr":             float,
      "roi":            float,
      "annual_units":   float,
      "wilson_lower_95": float
    }
  }
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ARTIFACTS = Path(__file__).parent / "artifacts"
REPORTS   = Path(__file__).parent / "reports"

# Same grid as global sweep
MAX_DRAW_GRID = [0.25, 0.28, 0.30, 0.32, 0.35]
MIN_BET_GRID  = [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72]
MIN_GEM_GRID  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
# Per-league odds bounds: each league has its own "sweet spot" — model is strong
# on mid-range odds but unreliable on extremes (heavy favs / longshots).
ODDS_LO_GRID  = [1.40, 1.50, 1.60]
ODDS_HI_GRID  = [2.50, 3.00, 3.50]

HISTORY_YEARS = 32 / 12.0

# Per-league constraints
MIN_N_LEAGUE = 12       # league must produce at least this many picks for sweep to be valid
MIN_WR       = 0.55     # reject combos with WR below this
MIN_ROI      = 0.05     # reject combos with ROI below +5%
MIN_LO95     = 0.40     # reject thin-sample fits (Wilson lower bound on WR)


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def _shin_devig_vector(
    home_odds: np.ndarray, draw_odds: np.ndarray, away_odds: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Shin de-vig on 3-outcome odds via per-row binary search.
    Returns (p_home, p_away). Falls back to proportional on rows that don't converge.
    """
    n = len(home_odds)
    p_h = np.full(n, np.nan)
    p_a = np.full(n, np.nan)
    for i in np.where(mask)[0]:
        h, d, a = home_odds[i], draw_odds[i], away_odds[i]
        if not (h > 0 and d > 0 and a > 0):
            continue
        qs = (1.0/h, 1.0/d, 1.0/a)
        Q = qs[0] + qs[1] + qs[2]
        if Q <= 1.0:
            p_h[i] = qs[0] / Q
            p_a[i] = qs[2] / Q
            continue
        def sum_p(z):
            return sum(
                (math.sqrt(z*z + 4*(1-z) * q*q / Q) - z) / (2*(1-z))
                for q in qs
            )
        lo, hi = 1e-6, 0.5
        if sum_p(lo) <= 1.0 or sum_p(hi) >= 1.0:
            p_h[i] = qs[0] / Q
            p_a[i] = qs[2] / Q
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            if sum_p(mid) > 1.0:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2
        p_h[i] = (math.sqrt(z*z + 4*(1-z) * qs[0]*qs[0] / Q) - z) / (2*(1-z))
        p_a[i] = (math.sqrt(z*z + 4*(1-z) * qs[2]*qs[2] / Q) - z) / (2*(1-z))
    return p_h, p_a


def evaluate_league_combo(
    proba: np.ndarray, y: np.ndarray, mask: np.ndarray,
    home_odds: np.ndarray, draw_odds: np.ndarray, away_odds: np.ndarray,
    mp_h: np.ndarray, mp_a: np.ndarray,
    max_draw: float, min_bet: float, min_gem: float,
    odds_lo: float, odds_hi: float,
) -> dict | None:
    if not mask.any():
        return None

    pH = proba[:, 0]; pD = proba[:, 1]; pA = proba[:, 2]

    pick_home = pH >= pA
    p_side    = np.where(pick_home, pH, pA)
    odds_side = np.where(pick_home, home_odds, away_odds)
    mp_side   = np.where(pick_home, mp_h, mp_a)
    won_side  = np.where(pick_home, y == 0, y == 2)

    sel = (
        mask
        & (pD < max_draw)
        & (p_side > min_bet)
        & (odds_side >= odds_lo) & (odds_side <= odds_hi)
        & (p_side - mp_side > min_gem)
    )
    n = int(sel.sum())
    if n == 0:
        return None
    won = won_side[sel].astype(int)
    odds = odds_side[sel]
    wins = int(won.sum())
    pnl = (odds - 1) * won - (1 - won)
    return {
        "n_bets": n,
        "wins":   wins,
        "wr":     float(wins / n),
        "roi":    float(pnl.mean()),
        "lo95":   wilson_lower(wins, n),
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
    leagues = info["league_name"].astype(str).to_numpy()

    odds_valid = ~(np.isnan(home_odds) | np.isnan(draw_odds) | np.isnan(away_odds))

    league_counts = pd.Series(leagues[covered & odds_valid]).value_counts()
    logger.info(f"Leagues with covered+odds data:\n{league_counts.to_string()}")

    # Pre-compute proportional + Shin devig vectors on the covered+valid mask
    base_mask = covered & odds_valid
    raw_sum = 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds
    mp_h_prop = (1.0 / home_odds) / raw_sum
    mp_a_prop = (1.0 / away_odds) / raw_sum
    logger.info("Computing Shin devig (binary search per row)…")
    mp_h_shin, mp_a_shin = _shin_devig_vector(home_odds, draw_odds, away_odds, base_mask)

    DEVIG_METHODS = {
        "proportional": (mp_h_prop, mp_a_prop),
        "shin":         (mp_h_shin, mp_a_shin),
    }

    all_rows = []
    best_per_league: dict[str, dict] = {}

    for league, total in league_counts.items():
        if total < 100:  # skip leagues with too little OOF data
            logger.warning(f"  skip {league}: {total} matches (< 100)")
            continue

        league_mask = base_mask & (leagues == league)

        candidates = []
        for devig_name, (mp_h, mp_a) in DEVIG_METHODS.items():
            for odds_lo in ODDS_LO_GRID:
                for odds_hi in ODDS_HI_GRID:
                    for max_draw in MAX_DRAW_GRID:
                        for min_bet in MIN_BET_GRID:
                            for min_gem in MIN_GEM_GRID:
                                r = evaluate_league_combo(
                                    proba, y, league_mask,
                                    home_odds, draw_odds, away_odds,
                                    mp_h, mp_a,
                                    max_draw, min_bet, min_gem,
                                    odds_lo, odds_hi,
                                )
                                if r is None:
                                    continue
                                r["devig"]    = devig_name
                                r["odds_lo"]  = odds_lo
                                r["odds_hi"]  = odds_hi
                                r["max_draw"] = max_draw
                                r["min_bet"]  = min_bet
                                r["min_gem"]  = min_gem
                                r["league"]   = league
                                r["bets_per_year"] = r["n_bets"] / HISTORY_YEARS
                                r["annual_units"]  = r["bets_per_year"] * r["roi"]
                                all_rows.append(r)
                                candidates.append(r)

        # Filter viable combos
        viable = [
            c for c in candidates
            if c["n_bets"] >= MIN_N_LEAGUE
            and c["wr"]   >= MIN_WR
            and c["roi"]  >= MIN_ROI
            and c["lo95"] >= MIN_LO95
        ]
        if not viable:
            logger.warning(
                f"  ❌ {league}: no viable combo "
                f"(n>={MIN_N_LEAGUE}, WR>={MIN_WR}, ROI>={MIN_ROI}, lo95>={MIN_LO95})"
            )
            continue

        # Pick best by annual_units (then by lo95 as tiebreaker)
        best = sorted(viable, key=lambda c: (c["annual_units"], c["lo95"]), reverse=True)[0]
        best_per_league[league] = {
            "devig":          best["devig"],
            "min_odds":       best["odds_lo"],
            "max_odds":       best["odds_hi"],
            "max_draw_prob":  best["max_draw"],
            "min_bet_prob":   best["min_bet"],
            "min_gem_score":  best["min_gem"],
            "n_bets_24mo":    best["n_bets"],
            "wr":             round(best["wr"], 4),
            "roi":            round(best["roi"], 4),
            "annual_units":   round(best["annual_units"], 3),
            "wilson_lower_95": round(best["lo95"], 4),
            "avg_odds":       round(best["avg_odds"], 2),
        }

    df = pd.DataFrame(all_rows)
    df.to_csv(REPORTS / "per_league_sweep.csv", index=False)

    out_path = ARTIFACTS / "per_league_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(best_per_league, f, indent=2)
    logger.info(f"Saved → {out_path}")

    # Console
    print("\n" + "=" * 120)
    print("PER-LEAGUE OPTIMAL GEM THRESHOLDS (cherry-pick: best of {proportional, shin})")
    print("=" * 120)
    print(f"{'League':>22s} | {'devig':>5s} {'oLo':>4s} {'oHi':>4s} {'mxD':>5s} {'mBet':>5s} {'mGem':>5s} | "
          f"{'n':>4s} {'WR':>6s} {'lo95':>6s} {'ROI':>7s} {'odds':>5s} {'/yr':>5s} {'units/yr':>9s}")
    print("-" * 130)
    rows_sorted = sorted(best_per_league.items(), key=lambda kv: kv[1]["annual_units"], reverse=True)
    total_yr = 0.0
    total_n  = 0
    for league, t in rows_sorted:
        units = t["annual_units"]
        total_yr += units
        total_n  += t["n_bets_24mo"]
        print(
            f"{league[:22]:>22s} | {t['devig'][:5]:>5s} {t['min_odds']:>4.2f} {t['max_odds']:>4.2f} "
            f"{t['max_draw_prob']:>5.2f} {t['min_bet_prob']:>5.2f} {t['min_gem_score']:>5.2f} | "
            f"{t['n_bets_24mo']:>4d} {t['wr']*100:>5.1f}% {t['wilson_lower_95']*100:>5.1f}% "
            f"{t['roi']*100:>+6.1f}% {t['avg_odds']:>5.2f} {t['n_bets_24mo']/HISTORY_YEARS:>5.1f} "
            f"{units:>+8.2f}"
        )
    print("-" * 130)
    print(f"{'TOTAL':>22s} | {' '*33} | {total_n:>4d} {' '*38} {total_n/HISTORY_YEARS:>5.1f} {total_yr:>+8.2f}")

    print(f"\n  vs current PROD universal (0.62/0.15): n=60, +4.54 units/yr")


if __name__ == "__main__":
    run()
