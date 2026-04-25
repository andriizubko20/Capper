"""
model/pure/mass_search.py

Brute-force per-league niche search across all variables in inventory.

For each (league, side):
  - Build per-side feature DataFrame (already computed).
  - Enumerate niches up to N_FACTORS (default 3) factors + odds_range.
  - Evaluate each: n, wins, WR, ROI, Wilson lower-95, EV (point + conservative).
  - Filter: n >= MIN_N, WR >= MIN_WR, ROI >= MIN_ROI.
  - Stability check on top (chronological halves).
  - Deduplicate overlapping niches (Jaccard > 0.85).

Output: model/pure/reports/profitable_niches.xlsx
  Sheets:
    Summary             — top niches across all leagues
    <League>            — one sheet per league with all surviving niches

Usage:
  python -m model.pure.mass_search [--min-n 25] [--min-wr 0.60] [--min-roi 0.20] [--max-factors 3]
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPORTS = Path(__file__).parent / "reports"

LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga", "Eredivisie", "Jupiler Pro League", "Serie B",
]

# Each entry: (column_name, comparison, threshold, label)
# comparison: ">=" or "<="
FACTOR_GRID = [
    # ── STRENGTH ──
    ("glicko_gap_side", ">=",  50,    "g>=50"),
    ("glicko_gap_side", ">=", 100,    "g>=100"),
    ("glicko_gap_side", ">=", 150,    "g>=150"),
    ("glicko_prob",     ">=",  0.55,  "p>=0.55"),
    ("glicko_prob",     ">=",  0.60,  "p>=0.60"),
    ("glicko_prob",     ">=",  0.65,  "p>=0.65"),
    ("glicko_prob",     ">=",  0.70,  "p>=0.70"),
    # ── XG ──
    ("xg_diff_side",    ">=",  0.20,  "xd>=0.2"),
    ("xg_diff_side",    ">=",  0.40,  "xd>=0.4"),
    ("xg_diff_side",    ">=",  0.60,  "xd>=0.6"),
    ("xg_quality_gap_side", ">=", 0.30, "xq>=0.3"),
    ("xg_quality_gap_side", ">=", 0.50, "xq>=0.5"),
    ("attack_vs_def_side",  ">=", 0.30, "ad>=0.3"),
    ("attack_vs_def_side",  ">=", 0.50, "ad>=0.5"),
    # ── FORM ──
    ("form_advantage_side", ">=", 0.5, "f>=0.5"),
    ("form_advantage_side", ">=", 1.0, "f>=1.0"),
    ("form_advantage_side", ">=", 1.5, "f>=1.5"),
    ("ppg_side",            ">=", 1.5, "ppg>=1.5"),
    ("ppg_side",            ">=", 1.8, "ppg>=1.8"),
    # ── MOMENTUM ──
    ("xg_trend_side",       ">=",  0.0,  "xt>=0"),
    ("xg_trend_side",       ">=",  0.1,  "xt>=0.1"),
    ("glicko_momentum_side",">=",  0,    "gm>=0"),
    ("glicko_momentum_side",">=", 20,    "gm>=20"),
    ("win_streak_side",     ">=",  1,    "ws>=1"),
    ("win_streak_side",     ">=",  2,    "ws>=2"),
    ("opp_lose_streak",     ">=",  1,    "ols>=1"),
    ("opp_lose_streak",     ">=",  2,    "ols>=2"),
    # ── STYLE ──
    ("possession_10_side", ">=", 0.50, "pos>=50"),
    ("possession_10_side", ">=", 0.55, "pos>=55"),
    ("sot_10_side",        ">=", 4.0,  "sot>=4"),
    ("sot_10_side",        ">=", 5.0,  "sot>=5"),
    ("pass_acc_10_side",   ">=", 0.80, "pa>=80"),
    # ── PHYSICAL ──
    ("rest_advantage_side",">=",  0,   "ra>=0"),
    ("rest_advantage_side",">=",  3,   "ra>=3"),
    # ── H2H ──
    ("h2h_wr_side",        ">=", 0.50, "h2h>=0.5"),
    # ── MARKET (mispricing) ──
    ("market_prob_side",   "<=", 0.60, "m<=0.60"),
    ("market_prob_side",   "<=", 0.65, "m<=0.65"),
    ("market_prob_side",   "<=", 0.70, "m<=0.70"),
]

ODDS_RANGES = [
    (1.40, 1.60),
    (1.55, 1.85),
    (1.70, 2.10),
    (1.85, 2.40),
    (2.10, 3.00),
]


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def build_side_df(match_factors: pd.DataFrame, league: str, side: str) -> pd.DataFrame:
    """
    Re-orient match factors so columns are 'side perspective' instead of 'home perspective'.
    Adds columns suffixed _side for the betting team and _other for the opponent.
    """
    df = match_factors[match_factors["league_name"] == league].copy()
    if df.empty:
        return df

    # We need extra info — pull from team_state if not in match_factors. Use what's available.
    # match_factors has: glicko_gap (h-a), xg_diff_home, xg_diff_away, attack_vs_def_home/away,
    # form_advantage (h-a), home/away_glicko_prob, home/away_market_prob, home/away_odds.
    # Need to add: ppg_side, xg_trend_side, glicko_momentum_side, win_streak_side, opp_lose_streak,
    # possession_10_side, sot_10_side, pass_acc_10_side, rest_advantage_side, h2h_wr_side
    # → these come from pure_features.parquet (long-format).
    pf_path = REPORTS / "pure_features.parquet"
    if not pf_path.exists():
        logger.error("pure_features.parquet missing — run model.pure.features first")
        raise SystemExit(1)
    pf = pd.read_parquet(pf_path)
    pf_side = pf[(pf["league_name"] == league) & (pf["side"] == side)].copy()

    # Merge: pf_side already has match_id + side-perspective columns
    pf_side = pf_side.rename(columns={
        "ppg_10":        "ppg_side",
        "xg_trend":      "xg_trend_side",
        "glicko_momentum": "glicko_momentum_side",
        "win_streak":    "win_streak_side",
        "lose_streak":   "lose_streak_side",
        "possession_10": "possession_10_side",
        "sot_10":        "sot_10_side",
        "pass_acc_10":   "pass_acc_10_side",
        "rest_advantage": "rest_advantage_side",
        "h2h_wr":        "h2h_wr_side",
        "form_advantage": "form_advantage_side",
        "xg_diff":       "xg_diff_side",
        "xg_quality_gap": "xg_quality_gap_side",
        "attack_vs_def": "attack_vs_def_side",
        "glicko_prob":   "glicko_prob",
        "market_prob":   "market_prob_side",
    })
    pf_side["glicko_gap_side"] = pf_side["glicko_gap"]

    # Need opp_lose_streak: take from the OTHER side's lose_streak
    pf_other = pf[(pf["league_name"] == league) & (pf["side"] != side)][["match_id", "lose_streak"]].copy()
    pf_other = pf_other.rename(columns={"lose_streak": "opp_lose_streak"})
    pf_side = pf_side.merge(pf_other, on="match_id", how="left")

    # Won marker
    pf_side["won"] = ((side == "home") & (pf_side["result"] == "H")) | \
                     ((side == "away") & (pf_side["result"] == "A"))
    pf_side["won"] = pf_side["won"].astype(int)
    pf_side = pf_side.sort_values("date").reset_index(drop=True)
    return pf_side


def evaluate_niche(df: pd.DataFrame, odds_lo: float, odds_hi: float, factors: list[tuple]) -> dict | None:
    """
    factors: list of (column, comparison, threshold) tuples.
    Returns dict with metrics or None if doesn't qualify.
    """
    odds = df["odds"].to_numpy()
    mask = (odds >= odds_lo) & (odds <= odds_hi)
    for col, op, thr, _label in factors:
        col_arr = df[col].to_numpy(dtype=float)
        valid = ~np.isnan(col_arr)
        if op == ">=":
            mask &= valid & (col_arr >= thr)
        else:
            mask &= valid & (col_arr <= thr)
    n = int(mask.sum())
    if n == 0:
        return None
    won = df["won"].to_numpy()[mask]
    sub_odds = odds[mask]
    wins = int(won.sum())
    wr = wins / n
    pnl_unit = (sub_odds - 1) * won - (1 - won)
    roi = float(pnl_unit.mean())
    avg_odds = float(sub_odds.mean())
    return {
        "n": n, "wins": wins, "wr": float(wr),
        "wr_lower_95": wilson_lower(wins, n),
        "avg_odds": avg_odds,
        "roi": roi,
        "ev_point":   float(wr) * avg_odds - 1,
        "ev_lower_95": wilson_lower(wins, n) * avg_odds - 1,
        "mask_idx":   np.where(mask)[0],
    }


def stability(df: pd.DataFrame, odds_lo: float, odds_hi: float, factors: list[tuple]) -> dict:
    n_total = len(df)
    cutoff = n_total // 2
    out = {}
    for label, sub in [("h1", df.iloc[:cutoff]), ("h2", df.iloc[cutoff:])]:
        e = evaluate_niche(sub.reset_index(drop=True), odds_lo, odds_hi, factors)
        if e is None:
            out[label] = {"n": 0, "wr": None, "roi": None}
        else:
            out[label] = {"n": e["n"], "wr": e["wr"], "roi": e["roi"]}
    return out


def jaccard(a, b) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def search_league_side(
    match_factors: pd.DataFrame,
    league: str, side: str,
    min_n: int, min_wr: float, min_roi: float,
    max_factors: int,
) -> list[dict]:
    df = build_side_df(match_factors, league, side)
    if len(df) < 100:
        return []

    found: list[dict] = []
    n_eval = 0

    for (odds_lo, odds_hi) in ODDS_RANGES:
        # k=0: just odds (no extra factors) — usually too generic, skip but allow if great
        for k in range(0, max_factors + 1):
            for combo in itertools.combinations(FACTOR_GRID, k):
                # Skip combos with same column appearing twice with different thresholds
                cols_in_combo = [f[0] for f in combo]
                if len(set(cols_in_combo)) < len(cols_in_combo):
                    continue

                e = evaluate_niche(df, odds_lo, odds_hi, list(combo))
                n_eval += 1
                if e is None:
                    continue
                if e["n"] < min_n:
                    continue
                if e["wr"] < min_wr:
                    continue
                if e["roi"] < min_roi:
                    continue

                # Compose niche id
                parts = [f"{side}[{odds_lo},{odds_hi})"] + [f[3] for f in combo]
                niche_id = " ".join(parts)
                found.append({
                    "league": league, "side": side, "niche_id": niche_id,
                    "odds_lo": odds_lo, "odds_hi": odds_hi,
                    "factors": "; ".join(f[3] for f in combo) if combo else "(odds-only)",
                    "n_factors": k,
                    **{k_: v for k_, v in e.items() if k_ != "mask_idx"},
                    "_combo": combo,
                    "_mask_idx": e["mask_idx"],
                })

    if not found:
        logger.info(f"  {league}/{side}: 0 hits ({n_eval:,} combos evaluated)")
        return []

    # Stability check: bothhalves must be acceptable
    for f in found:
        st = stability(df, f["odds_lo"], f["odds_hi"], list(f["_combo"]))
        f["h1_n"]  = st["h1"]["n"];  f["h1_wr"] = st["h1"]["wr"]
        f["h2_n"]  = st["h2"]["n"];  f["h2_wr"] = st["h2"]["wr"]
        f["stable"] = (
            st["h1"]["n"] >= 5 and st["h2"]["n"] >= 5 and
            (st["h1"]["wr"] or 0) >= min_wr - 0.10 and
            (st["h2"]["wr"] or 0) >= min_wr - 0.10
        )

    # Dedup by Jaccard, keep highest ROI
    found.sort(key=lambda x: x["roi"], reverse=True)
    kept: list[dict] = []
    for cand in found:
        is_dup = False
        for k in kept:
            if jaccard(cand["_mask_idx"], k["_mask_idx"]) > 0.85:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    logger.info(
        f"  {league}/{side}: {len(found):,} raw → {len(kept):,} after dedup "
        f"({n_eval:,} combos evaluated)"
    )
    return kept


def to_excel(all_niches: list[dict], min_n: int, min_wr: float, min_roi: float) -> Path:
    out = REPORTS / "profitable_niches.xlsx"
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_niches)
    if df.empty:
        logger.error("No niches passed filters!")
        return out

    cols_keep = [
        "league", "side", "niche_id", "n_factors",
        "n", "wins", "wr", "wr_lower_95", "avg_odds",
        "roi", "ev_point", "ev_lower_95",
        "h1_n", "h1_wr", "h2_n", "h2_wr", "stable",
        "factors",
    ]
    df = df[cols_keep].copy()

    # Round
    for c in ("wr", "wr_lower_95", "avg_odds", "roi", "ev_point", "ev_lower_95"):
        df[c] = df[c].round(4)
    for c in ("h1_wr", "h2_wr"):
        df[c] = df[c].apply(lambda x: round(x, 4) if pd.notna(x) else None)

    df = df.sort_values(["league", "roi"], ascending=[True, False]).reset_index(drop=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Summary: top 100 across all leagues
        summary = df.sort_values("roi", ascending=False).head(100)
        summary.to_excel(writer, sheet_name="Summary_Top100", index=False)

        # Per-league sheets
        for league in df["league"].unique():
            sub = df[df["league"] == league].sort_values("roi", ascending=False)
            sheet_name = league[:31]  # Excel limit
            sub.to_excel(writer, sheet_name=sheet_name, index=False)

        # Stable-only sheet (for safety-conscious users)
        stable = df[df["stable"]].sort_values("roi", ascending=False)
        stable.to_excel(writer, sheet_name="Stable_Only", index=False)

    logger.info(f"Excel saved → {out}")
    print(f"\n📊 {out}")
    print(f"  Total niches: {len(df):,}")
    print(f"  Stable: {df['stable'].sum():,}")
    print(f"  Filters: n>={min_n}, WR>={min_wr*100:.0f}%, ROI>={min_roi*100:.0f}%")
    return out


def run(min_n: int = 25, min_wr: float = 0.60, min_roi: float = 0.20, max_factors: int = 3) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    logger.info(f"Loaded {len(factors):,} matches")
    logger.info(f"Filters: n>={min_n}, WR>={min_wr:.0%}, ROI>={min_roi:.0%}, max_factors={max_factors}")

    all_niches = []
    for league in LEAGUES:
        for side in ("home", "away"):
            niches = search_league_side(
                factors, league, side,
                min_n=min_n, min_wr=min_wr, min_roi=min_roi,
                max_factors=max_factors,
            )
            all_niches.extend(niches)

    if all_niches:
        # Strip transient _mask_idx and _combo for serialization
        for n in all_niches:
            n.pop("_mask_idx", None)
            n.pop("_combo", None)

    to_excel(all_niches, min_n, min_wr, min_roi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n",       type=int,   default=25)
    ap.add_argument("--min-wr",      type=float, default=0.60)
    ap.add_argument("--min-roi",     type=float, default=0.20)
    ap.add_argument("--max-factors", type=int,   default=3)
    args = ap.parse_args()
    run(min_n=args.min_n, min_wr=args.min_wr, min_roi=args.min_roi, max_factors=args.max_factors)
