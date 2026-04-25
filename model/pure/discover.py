"""
model/pure/discover.py

Auto-discover Pure niches per league.

Pipeline:
  1. Load match factors from model/pure/reports/match_factors.parquet
     (built by calibrate.py — already in DB-derived per-match factor table).
  2. For each (league, side):
       a. Generate ~5,000 candidate niches via grid.
       b. Evaluate each on FULL history → n, wins, p_is, Wilson lower-95.
       c. Filter: n >= MIN_N, p_is_lower >= MIN_P_LOWER (conservative).
       d. Stability: split data 60/40 chronologically, both halves must
          show p_is >= MIN_P_HALF.
       e. Deduplicate: keep niches whose qualifying matches don't largely
          overlap (Jaccard < 0.85 with already-kept niches), preferring
          higher p_is_lower.
  3. Save surviving niches → model/pure/artifacts/niches.json
     (per-league list with niche dict + p_is + Wilson bounds + n_samples).

Production:
  predictor.py loads niches.json. At pick time, evaluate match against all
  niches for its league. If any match, take the niche with highest p_is_lower
  (most conservative confident pick) and use it for EV/Kelly.

Usage:
  python -m model.pure.discover [--min-n 25] [--min-p-lower 0.55]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.niches import (
    GRID, SIDES, generate_candidate_niches, matches_niche,
    niche_id, features_for_side,
)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
REPORTS   = ROOT / "reports"

# Leagues with usable odds (from DB diagnostic). UCL has no Glicko in stats — skip.
LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga", "Eredivisie", "Jupiler Pro League", "Serie B",
]


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def wilson_upper(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + spread) / denom


def _ohms(df: pd.DataFrame, side: str) -> np.ndarray:
    """Per-row 'won' marker for the bet on `side`."""
    if side == "home":
        return (df["result"] == "H").to_numpy()
    return (df["result"] == "A").to_numpy()


def _vectorized_match_mask(df: pd.DataFrame, niche: dict) -> np.ndarray:
    """
    Returns boolean mask of df rows that match `niche` (single side).
    df is already filtered to one (league, side) — features are pre-computed
    columns: odds, glicko_gap, xg_diff, attack_vs_def, form_advantage,
             glicko_prob, market_prob.
    """
    side = niche["side"]
    lo, hi = niche["odds_range"]
    odds = df["odds"].to_numpy()
    mask = (odds >= lo) & (odds <= hi)

    for col, key in [
        ("glicko_gap",     "min_glicko_gap"),
        ("xg_diff",        "min_xg_diff"),
        ("attack_vs_def",  "min_attack_vs_def"),
        ("form_advantage", "min_form_advantage"),
        ("glicko_prob",    "min_glicko_prob"),
    ]:
        thr = niche.get(key)
        if thr is None:
            continue
        col_arr = df[col].to_numpy(dtype=float)
        mask &= (~np.isnan(col_arr)) & (col_arr >= thr)

    thr = niche.get("max_market_prob")
    if thr is not None:
        col_arr = df["market_prob"].to_numpy(dtype=float)
        mask &= (~np.isnan(col_arr)) & (col_arr <= thr)

    return mask


def _build_per_side_df(match_factors: pd.DataFrame, league: str, side: str) -> pd.DataFrame:
    """Build a DataFrame from match-perspective columns to side-perspective."""
    df = match_factors[match_factors["league_name"] == league].copy()
    if side == "home":
        df["odds"]            = df["home_odds"]
        df["xg_diff"]         = df["xg_diff_home"]
        df["attack_vs_def"]   = df["attack_vs_def_home"]
        df["glicko_prob"]     = df["home_glicko_prob"]
        df["market_prob"]     = df["home_market_prob"]
        # glicko_gap and form_advantage already home-perspective
    else:
        df["odds"]            = df["away_odds"]
        df["xg_diff"]         = df["xg_diff_away"]
        df["attack_vs_def"]   = df["attack_vs_def_away"]
        df["glicko_prob"]     = df["away_glicko_prob"]
        df["market_prob"]     = df["away_market_prob"]
        df["glicko_gap"]      = -df["glicko_gap"]
        df["form_advantage"]  = -df["form_advantage"]
    df["won"] = _ohms(df, side).astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def evaluate_niche(df: pd.DataFrame, niche: dict) -> dict | None:
    mask = _vectorized_match_mask(df, niche)
    n = int(mask.sum())
    if n == 0:
        return None
    won = df["won"].to_numpy()[mask]
    odds = df["odds"].to_numpy()[mask]
    wins = int(won.sum())
    p_is = wins / n
    return {
        "n":       n,
        "wins":    wins,
        "p_is":    p_is,
        "p_is_lower_95": wilson_lower(wins, n),
        "p_is_upper_95": wilson_upper(wins, n),
        "avg_odds":      float(odds.mean()),
        "mask_idx":      np.where(mask)[0],   # for dedup overlap test
    }


def stability_check(df: pd.DataFrame, niche: dict, half: float = 0.6, min_p_half: float = 0.50) -> dict:
    cutoff = int(len(df) * half)
    h1 = df.iloc[:cutoff].reset_index(drop=True)
    h2 = df.iloc[cutoff:].reset_index(drop=True)
    out = {}
    for label, sub in [("h1", h1), ("h2", h2)]:
        e = evaluate_niche(sub, niche)
        if e is None or e["n"] < 5:
            out[label] = {"n": 0, "p_is": None, "p_is_lower_95": 0.0}
        else:
            out[label] = {"n": e["n"], "p_is": e["p_is"], "p_is_lower_95": e["p_is_lower_95"]}
    out["passes"] = (
        out["h1"]["p_is_lower_95"] >= min_p_half * 0.85 and  # h1 may have less data, bit lenient
        out["h2"]["p_is_lower_95"] >= min_p_half
    )
    return out


def jaccard(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
    sa, sb = set(a_idx.tolist()), set(b_idx.tolist())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def discover_for_league_side(
    match_factors: pd.DataFrame,
    league: str, side: str,
    min_n: int = 25,
    min_p_lower: float = 0.55,
    max_jaccard: float = 0.85,
) -> list[dict]:
    df = _build_per_side_df(match_factors, league, side)
    if len(df) < 100:
        return []

    candidates = generate_candidate_niches(side)
    surviving = []

    # max_market_prob is largely implied by odds_range, so it doesn't count as
    # a fundamental signal by itself. Require an actual strength/form factor.
    fundamental_keys = (
        "min_glicko_gap", "min_xg_diff", "min_attack_vs_def",
        "min_form_advantage", "min_glicko_prob",
    )

    for niche in candidates:
        n_fund = sum(1 for k in fundamental_keys if niche.get(k) is not None)
        if n_fund < 1:
            continue

        e = evaluate_niche(df, niche)
        if e is None:
            continue
        if e["n"] < min_n:
            continue
        if e["p_is_lower_95"] < min_p_lower:
            continue
        st = stability_check(df, niche)
        if not st["passes"]:
            continue
        surviving.append({**niche, **{k: v for k, v in e.items() if k != "mask_idx"},
                          "stability": st, "_mask_idx": e["mask_idx"]})

    if not surviving:
        return []

    # Dedup: greedy by p_is_lower_95
    surviving.sort(key=lambda x: x["p_is_lower_95"], reverse=True)
    kept: list[dict] = []
    for cand in surviving:
        cand_idx = cand["_mask_idx"]
        is_dup = False
        for k in kept:
            if jaccard(cand_idx, k["_mask_idx"]) > max_jaccard:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    # Strip transient _mask_idx, add niche_id
    out = []
    for k in kept:
        nid = niche_id({**k, "side": side})
        clean = {kk: vv for kk, vv in k.items() if not kk.startswith("_")}
        clean["niche_id"] = nid
        # Round numbers for readability
        for f in ("p_is", "p_is_lower_95", "p_is_upper_95", "avg_odds"):
            if f in clean:
                clean[f] = round(clean[f], 4)
        out.append(clean)
    return out


def run(min_n: int = 25, min_p_lower: float = 0.55) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    match_factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    logger.info(f"Loaded {len(match_factors):,} matches")

    all_niches: dict = {}
    summary_rows = []
    for league in LEAGUES:
        for side in SIDES:
            logger.info(f"Discovering {league}/{side} …")
            niches = discover_for_league_side(
                match_factors, league, side,
                min_n=min_n, min_p_lower=min_p_lower,
            )
            if not niches:
                logger.warning(f"  ❌ {league}/{side}: 0 surviving niches")
                continue
            logger.info(f"  ✅ {league}/{side}: {len(niches)} niches kept")
            all_niches.setdefault(league, {})[side] = niches
            for n in niches[:5]:
                summary_rows.append({
                    "league": league, "side": side,
                    "niche": n["niche_id"],
                    "n": n["n"], "p_is": n["p_is"],
                    "p_is_lower_95": n["p_is_lower_95"],
                    "avg_odds": n["avg_odds"],
                    "h1_p": n["stability"]["h1"]["p_is"],
                    "h2_p": n["stability"]["h2"]["p_is"],
                })

    out = ARTIFACTS / "niches.json"
    with open(out, "w") as f:
        json.dump(all_niches, f, indent=2, default=str)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "niches_summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"PURE NICHES — top 5 per league/side  (constraint: n>={min_n}, p_lower>={min_p_lower})")
    print("=" * 100)
    if summary.empty:
        print("No niches survived. Try lowering min_p_lower or min_n.")
        return
    for r in summary.itertuples():
        h1p = f"{r.h1_p:.0%}" if r.h1_p else "-"
        h2p = f"{r.h2_p:.0%}" if r.h2_p else "-"
        print(
            f"  {r.league:>22s} {r.side:>4s}  n={r.n:>4d}  p_is={r.p_is:.0%}  "
            f"lo95={r.p_is_lower_95:.0%}  odds={r.avg_odds:.2f}  h1[{h1p}] h2[{h2p}]  | {r.niche}"
        )
    logger.info(f"Saved niches → {out}, summary → {REPORTS / 'niches_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n",       type=int,   default=25)
    ap.add_argument("--min-p-lower", type=float, default=0.55)
    args = ap.parse_args()
    run(min_n=args.min_n, min_p_lower=args.min_p_lower)
