"""
model/pure/apply_recent.py

Apply user's curated 41 niches to recent finished matches.

For each match in date range:
  - Build per-side feature dict
  - Test ALL niches for the match's league
  - If any matches → record pick + actual result

Output: console report + CSV with all picks.

Usage:
  python -m model.pure.apply_recent [--days 7] [--from 2026-04-18] [--to 2026-04-25]
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.selected_niches import parse_all

REPORTS = Path(__file__).parent / "reports"


def matches_niche(features: dict, niche: dict) -> bool:
    """Check if features dict (per-side) matches niche conditions."""
    odds = features.get("odds")
    if odds is None or pd.isna(odds):
        return False
    lo, hi = niche["odds_range"]
    if not (lo <= odds <= hi):
        return False

    checks = [
        ("glicko_gap",       "min_glicko_gap",      ">="),
        ("glicko_prob",      "min_glicko_prob",     ">="),
        ("xg_diff",          "min_xg_diff",         ">="),
        ("xg_quality_gap",   "min_xg_quality_gap",  ">="),
        ("attack_vs_def",    "min_attack_vs_def",   ">="),
        ("form_advantage",   "min_form_advantage",  ">="),
        ("ppg",              "min_ppg",             ">="),
        ("xg_trend",         "min_xg_trend",        ">="),
        ("glicko_momentum",  "min_glicko_momentum", ">="),
        ("win_streak",       "min_win_streak",      ">="),
        ("opp_lose_streak",  "min_opp_lose_streak", ">="),
        ("possession_10",    "min_possession_10",   ">="),
        ("sot_10",           "min_sot_10",          ">="),
        ("pass_acc_10",      "min_pass_acc_10",     ">="),
        ("rest_advantage",   "min_rest_advantage",  ">="),
        ("h2h_wr",           "min_h2h_wr",          ">="),
        ("market_prob",      "max_market_prob",     "<="),
    ]
    for feat_key, niche_key, op in checks:
        thr = niche.get(niche_key)
        if thr is None:
            continue
        v = features.get(feat_key)
        if v is None or pd.isna(v):
            return False
        if op == ">=" and v < thr:
            return False
        if op == "<=" and v > thr:
            return False
    return True


def build_side_features(row: pd.Series, side: str) -> dict:
    """
    Convert a row from pure_features.parquet to a flat features dict
    aligned with niche grammar.
    """
    if side != row["side"]:
        return {}
    return {
        "odds":            row["odds"],
        "glicko_gap":      row.get("glicko_gap"),
        "glicko_prob":     row.get("glicko_prob"),
        "market_prob":     row.get("market_prob"),
        "xg_diff":         row.get("xg_diff"),
        "xg_quality_gap":  row.get("xg_quality_gap"),
        "attack_vs_def":   row.get("attack_vs_def"),
        "form_advantage":  row.get("form_advantage"),
        "ppg":             row.get("ppg_10"),
        "xg_trend":        row.get("xg_trend"),
        "glicko_momentum": row.get("glicko_momentum"),
        "win_streak":      row.get("win_streak"),
        "opp_lose_streak": row.get("opp_lose_streak"),  # added below
        "possession_10":   row.get("possession_10"),
        "sot_10":          row.get("sot_10"),
        "pass_acc_10":     row.get("pass_acc_10"),
        "rest_advantage":  row.get("rest_advantage"),
        "h2h_wr":          row.get("h2h_wr"),
    }


def run(date_from: datetime, date_to: datetime) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    niches_by_league = parse_all()
    total_niches = sum(len(v) for v in niches_by_league.values())
    logger.info(f"Loaded {total_niches} niches across {len(niches_by_league)} leagues")
    logger.info(f"Date range: {date_from.date()} → {date_to.date()}")

    pf = pd.read_parquet(REPORTS / "pure_features.parquet")
    pf["date"] = pd.to_datetime(pf["date"])

    # Pull opp_lose_streak: each row is one (match,side). Need OTHER side's lose_streak.
    other = pf[["match_id", "side", "lose_streak"]].copy()
    other["other_side"] = other["side"].map({"home": "away", "away": "home"})
    other = other.rename(columns={"lose_streak": "opp_lose_streak"})
    pf = pf.merge(
        other[["match_id", "other_side", "opp_lose_streak"]].rename(columns={"other_side": "side"}),
        on=["match_id", "side"], how="left",
    )

    # Filter to date range
    sub = pf[(pf["date"] >= pd.Timestamp(date_from)) & (pf["date"] <= pd.Timestamp(date_to))].copy()
    logger.info(f"{len(sub):,} per-side rows in window")

    picks: list[dict] = []
    for r in sub.itertuples(index=False):
        row_dict = r._asdict()
        league = row_dict["league_name"]
        if league not in niches_by_league:
            continue
        side = row_dict["side"]
        feats = build_side_features(pd.Series(row_dict), side)

        for niche in niches_by_league[league]:
            if niche["side"] != side:
                continue
            if matches_niche(feats, niche):
                picks.append({
                    "date":      row_dict["date"],
                    "league":    league,
                    "niche_id":  niche["niche_id"],
                    "side":      side,
                    "match_id":  row_dict["match_id"],
                    "result":    row_dict["result"],
                    "odds":      row_dict["odds"],
                    "won":       int(row_dict["won"]),
                })

    if not picks:
        print(f"\nNo niches matched in window {date_from.date()} → {date_to.date()}.")
        return

    df = pd.DataFrame(picks)
    df = df.sort_values("date")

    # Get team names for display
    from sqlalchemy import text
    from db.session import SessionLocal
    db = SessionLocal()
    try:
        ids = [int(x) for x in df["match_id"].unique()]
        names = pd.DataFrame(db.execute(
            text("""
                SELECT m.id AS match_id, th.name AS home, ta.name AS away,
                       m.home_score, m.away_score
                FROM matches m
                JOIN teams th ON th.id = m.home_team_id
                JOIN teams ta ON ta.id = m.away_team_id
                WHERE m.id = ANY(:ids)
            """), {"ids": ids}).fetchall(),
            columns=["match_id", "home", "away", "home_score", "away_score"],
        )
    finally:
        db.close()
    df = df.merge(names, on="match_id", how="left")

    # Console report
    print("\n" + "=" * 110)
    print(f"PURE PICKS — {len(df)} picks in {(date_to - date_from).days + 1} days "
          f"({date_from.date()} → {date_to.date()})")
    print("=" * 110)
    print(f"  {'date':>10s}  {'league':>16s}  {'match':>40s}  {'side':>4s}  "
          f"{'odds':>5s}  {'res':>4s}  {'won':>4s}  niche")
    print("  " + "-" * 105)
    for r in df.itertuples():
        match_str = f"{r.home[:18]} {r.home_score}-{r.away_score} {r.away[:18]}"
        won_str = "✅" if r.won else "❌"
        print(
            f"  {r.date.strftime('%Y-%m-%d')}  {r.league:>16s}  {match_str:>40s}  "
            f"{r.side:>4s}  {r.odds:>5.2f}  {r.result:>4s}  {won_str:>4s}  {r.niche_id}"
        )

    # Aggregate
    n = len(df)
    wins = int(df["won"].sum())
    wr = wins / n
    pnl = ((df["odds"] - 1) * df["won"] - (1 - df["won"])).sum()
    roi = pnl / n
    print("\n" + "-" * 110)
    print(f"  TOTAL: {n} picks | WR {wr:.1%} ({wins}W-{n - wins}L) | "
          f"ROI {roi:+.1%} | Cumulative PnL {pnl:+.2f}u (1u flat stake)")

    out = REPORTS / "pure_picks_recent.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",  type=int, default=7)
    ap.add_argument("--from",  dest="d_from", type=str, default=None)
    ap.add_argument("--to",    dest="d_to", type=str, default=None)
    args = ap.parse_args()

    if args.d_to:
        date_to = datetime.strptime(args.d_to, "%Y-%m-%d")
    else:
        date_to = datetime(2026, 4, 25)
    if args.d_from:
        date_from = datetime.strptime(args.d_from, "%Y-%m-%d")
    else:
        date_from = date_to - timedelta(days=args.days)

    run(date_from, date_to)
