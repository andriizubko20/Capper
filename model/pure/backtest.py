"""
model/pure/backtest.py

Portfolio backtest of discovered Pure niches.

For each historical match:
  1. Find all matching niches (across both sides) for its league.
  2. Pick the niche with HIGHEST p_is_lower_95 (most confident).
  3. Compute EV = p_is * odds - 1. Skip if EV <= 0 (shouldn't happen since
     p_lower-based niches are conservative, but defensive).
  4. Stake = flat 4% of current bankroll.
  5. Settle with actual result.

Reports: chronological bankroll, WR, ROI, drawdown, Sharpe, per-league + per-niche.

Usage:
  python -m model.pure.backtest [--bank 1000]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.niches import features_for_side, matches_niche

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
REPORTS   = ROOT / "reports"

FLAT_STAKE_FRAC = 0.04


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def find_best_niche(row: dict, league_niches: dict) -> dict | None:
    """Iterate all niches; return the one with highest p_is_lower that matches."""
    best = None
    for side in ("home", "away"):
        feats = features_for_side(row, side)
        if feats.get("odds") is None:
            continue
        for niche in league_niches.get(side, []):
            if matches_niche(feats, niche, side):
                if best is None or niche["p_is_lower_95"] > best["p_is_lower_95"]:
                    best = {**niche, "_side": side, "_odds": feats["odds"]}
    return best


def run(starting_bank: float = 1000.0) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "niches.json") as f:
        niches_by_league = json.load(f)

    df = pd.read_parquet(REPORTS / "match_factors.parquet")
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} matches; niches in {len(niches_by_league)} leagues")

    bets: list[dict] = []
    bank = starting_bank

    for row in df.itertuples(index=False):
        league = row.league_name
        if league not in niches_by_league:
            continue
        row_dict = row._asdict()
        best = find_best_niche(row_dict, niches_by_league[league])
        if best is None:
            continue

        side = best["_side"]
        odds = best["_odds"]
        p_is = best["p_is"]
        ev = p_is * odds - 1
        if ev <= 0:
            continue

        won = (side == "home" and row.result == "H") or (side == "away" and row.result == "A")
        stake = FLAT_STAKE_FRAC * bank
        pnl = stake * (odds - 1) if won else -stake
        bank += pnl

        bets.append({
            "date": row.date, "league": league, "niche_id": best["niche_id"],
            "side": side, "odds": odds, "p_is": p_is,
            "p_is_lower_95": best["p_is_lower_95"],
            "ev": ev, "won": int(won), "stake": stake, "pnl": pnl, "bank_after": bank,
        })

    if not bets:
        print("No bets generated.")
        return

    bdf = pd.DataFrame(bets)
    bdf.to_csv(REPORTS / "pure_portfolio_bets.csv", index=False)

    n = len(bdf)
    wins = int(bdf["won"].sum())
    wr = wins / n
    roi = float(((bdf["odds"] - 1) * bdf["won"] - (1 - bdf["won"])).mean())
    final = float(bdf["bank_after"].iloc[-1])
    peak = bdf["bank_after"].cummax()
    max_dd = float(((bdf["bank_after"] - peak) / peak).min())

    bdf["week"] = pd.to_datetime(bdf["date"]).dt.to_period("W").astype(str)
    weekly_pnl = bdf.groupby("week")["pnl"].sum()
    sharpe = float(weekly_pnl.mean() / weekly_pnl.std() * math.sqrt(52)) if weekly_pnl.std() > 0 else None

    print("\n" + "=" * 80)
    print(f"PURE PORTFOLIO BACKTEST  (flat 4%, {len(niches_by_league)} leagues)")
    print("=" * 80)
    print(f"  N bets               : {n:,}")
    print(f"  Win rate             : {wr:.1%}  (Wilson lo95: {wilson_lower(wins, n):.1%})")
    print(f"  ROI per stake        : {roi:+.2%}")
    print(f"  Avg odds             : {bdf['odds'].mean():.2f}")
    print(f"  Bets / week          : {n / max(bdf['week'].nunique(), 1):.1f}")
    print(f"  Final bank           : ${final:,.0f} (from ${starting_bank:,.0f})")
    print(f"  Max drawdown         : {max_dd:.1%}")
    print(f"  Sharpe (weekly)      : {sharpe:.2f}" if sharpe else "  Sharpe : n/a")

    # Per-league
    print("\n  PER-LEAGUE:")
    league_stats = bdf.groupby("league").agg(
        n=("won", "size"),
        wins=("won", "sum"),
        wr=("won", "mean"),
        avg_odds=("odds", "mean"),
        roi=("pnl", lambda p: p.sum() / bdf.loc[p.index, "stake"].sum()),
    ).reset_index().sort_values("n", ascending=False)
    for r in league_stats.itertuples():
        lo = wilson_lower(int(r.wins), r.n)
        print(f"    {r.league:>22s}: n={r.n:>3d}  WR={r.wr:5.1%}  lo95={lo:5.1%}  ROI={r.roi:+6.1%}  avg_odds={r.avg_odds:.2f}")

    # Per-niche (top 10 by frequency)
    print("\n  PER-NICHE (top 10 by freq):")
    niche_stats = bdf.groupby("niche_id").agg(
        n=("won", "size"),
        wins=("won", "sum"),
        wr=("won", "mean"),
        roi=("pnl", lambda p: p.sum() / bdf.loc[p.index, "stake"].sum()),
    ).reset_index().sort_values("n", ascending=False).head(10)
    for r in niche_stats.itertuples():
        print(f"    {r.niche_id:>50s}  n={r.n:>3d}  WR={r.wr:5.1%}  ROI={r.roi:+6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=float, default=1000.0)
    args = ap.parse_args()
    run(starting_bank=args.bank)
