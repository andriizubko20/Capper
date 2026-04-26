"""
model/pure/recency_check.py

Re-evaluate user's curated niches on RECENT data only (e.g. last 12 months)
and compare to full-history p_is. Flag niches whose performance has decayed.

Output:
  console table with: niche_id | full_p_is | recent_p_is | drift | verdict
  reports/recency_check.csv
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.compute_pis import evaluate_niche
from model.pure.selected_niches import parse_all

REPORTS = Path(__file__).parent / "reports"


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def evaluate_filtered(df: pd.DataFrame, niche: dict) -> dict:
    """Same as compute_pis.evaluate_niche but uses pre-filtered df.

    Recency check needs raw "all rows in df" semantics (df_full vs df_recent
    are already pre-filtered by date), so we explicitly disable the OOS split
    inside evaluate_niche by passing oos_start=None.
    """
    return evaluate_niche(df, niche, oos_start=None)


def positive_ev_at_lower_bound(p_lower_95: float, avg_odds: float | None) -> bool:
    """Methodology guard: a niche should only be admitted when its lower-95
    Wilson bound × avg_odds yields positive EV. Without this guard, a thin
    sample could produce a high *point-estimate* p_is that admits noise.

    Formula:  wilson_lower_95 * avg_odds - 1 > 0
    Threshold: > 0  (strict positive EV at the lower confidence bound)
    """
    if avg_odds is None or p_lower_95 is None:
        return False
    return (p_lower_95 * avg_odds - 1.0) > 0.0


def run(cutoff_date: str = "2025-01-01") -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logger.info(f"Recency cutoff: {cutoff_date}")

    df_full = pd.read_parquet(REPORTS / "match_factors.parquet")
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_recent = df_full[df_full["date"] >= cutoff_date].reset_index(drop=True)
    logger.info(f"Full: {len(df_full):,} matches, Recent: {len(df_recent):,} matches")

    niches_by_league = parse_all()
    rows = []
    for league, niches in niches_by_league.items():
        for n in niches:
            n["_league"] = league
            # Pass oos_start=None: recency_check splits df by date itself
            # (df_full / df_recent), and we want the raw stats on each slice
            # without an additional internal IS/OOS split.
            full_stats   = evaluate_niche(df_full,   n, oos_start=None)
            recent_stats = evaluate_niche(df_recent, n, oos_start=None)
            del n["_league"]

            full_p   = full_stats.get("p_is")
            recent_p = recent_stats.get("p_is")
            drift = (recent_p - full_p) if (full_p and recent_p) else None

            # Methodology guard: ev_at_lower_95 must be > 0.
            # This protects against thin-sample point estimates passing through
            # while the lower-95 Wilson bound is actually break-even / negative.
            full_ev_guard = positive_ev_at_lower_bound(
                full_stats.get("p_is_lower_95"), full_stats.get("avg_odds")
            )

            rows.append({
                "league":      league,
                "niche":       n["niche_id"],
                "full_n":      full_stats["n"],
                "full_p_is":   full_p,
                "full_lo95":   full_stats["p_is_lower_95"],
                "recent_n":    recent_stats["n"],
                "recent_p_is": recent_p,
                "recent_lo95": recent_stats["p_is_lower_95"],
                "drift":       drift,
                "avg_odds":    full_stats.get("avg_odds"),
                "ev_lower_95_pos": full_ev_guard,
            })

    df_out = pd.DataFrame(rows)

    def verdict(r):
        if r.recent_n < 5:
            return "small n"
        if r.recent_p_is is None:
            return "no data"
        if r.drift is None:
            return "?"
        # Methodology guard: full-history lower-95 EV must be positive.
        # Without it, point-estimate p_is alone could admit thin-sample noise.
        if not r.ev_lower_95_pos:
            return "DROP - lo95 EV<=0"
        # Strong decay = drift < -0.10 AND recent_lo95 < 0.55
        if r.drift < -0.15 and r.recent_lo95 < 0.55:
            return "DROP - decayed"
        if r.drift < -0.05 and r.recent_lo95 < 0.55:
            return "watch - weakening"
        if r.recent_p_is >= r.full_p_is - 0.03:
            return "stable"
        return "mild drift"

    df_out["verdict"] = df_out.apply(verdict, axis=1)
    df_out.to_csv(REPORTS / "recency_check.csv", index=False)

    # Console
    print("\n" + "=" * 130)
    print(f"RECENCY CHECK — niches re-evaluated on data since {cutoff_date}")
    print("=" * 130)
    print(f"{'league':>20s} {'niche':>50s}  {'full':>11s}  {'recent':>11s}  {'drift':>7s}  verdict")
    print("-" * 130)
    by_league = df_out.groupby("league", sort=False)
    for lg, sub in by_league:
        for r in sub.itertuples():
            full_str   = f"{r.full_p_is*100:>5.1f}% n={r.full_n:>3d}"   if r.full_p_is   else "    n/a    "
            recent_str = f"{r.recent_p_is*100:>5.1f}% n={r.recent_n:>3d}" if r.recent_p_is else "    n/a    "
            drift_str  = f"{r.drift*100:+5.1f}%" if r.drift is not None else "  ---"
            print(f"{r.league:>20s} {r.niche:>50s}  {full_str:>11s}  {recent_str:>11s}  {drift_str:>7s}  {r.verdict}")

    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df_out["verdict"].value_counts().to_string())
    logger.info(f"Saved → {REPORTS / 'recency_check.csv'}")


if __name__ == "__main__":
    import sys
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    run(cutoff)
