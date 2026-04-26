"""
model/pure/compute_pis.py

Compute historical p_is + Wilson lower-95 for each user-curated niche
and save to selected_niches_with_pis.json. Used by scheduler at pick time
to compute EV / Kelly stake.

Note: Computes p_is only on rows BEFORE OOS_START.
This prevents in-sample overfit when niches are refreshed against the
same data they'll be evaluated against. Mirrors the methodology used in
model/monster/features.py::compute_p_is and model/monster/niches.OOS_START.

The OOS_START default ("2025-08-01") matches Monster's choice so both
models share an apples-to-apples evaluation horizon. The scheduler reads
`p_is` from selected_niches_with_pis.json as the true (untouched-by-OOS)
win rate and uses it as `prediction.probability` for EV/Kelly.

Optionally, p_oos / n_oos / wins_oos / avg_odds_oos diagnostic fields are
also computed for visibility — they are NOT consumed by production picks.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.selected_niches import parse_all

REPORTS = Path(__file__).parent / "reports"
ARTIFACTS = Path(__file__).parent / "artifacts"

# Default cutoff — anything on/after this date is OOS and EXCLUDED from p_is.
# Mirrors model/monster/niches.OOS_START.
OOS_START = "2025-08-01"


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def _build_side_df(df_factors: pd.DataFrame, niche: dict) -> pd.DataFrame:
    """Apply per-side column renaming + merge pure_features. Returns sub df with
    all columns niche thresholds need, plus 'won' / 'odds' / 'date'."""
    side = niche["side"]
    sub = df_factors[df_factors["league_name"] == niche["_league"]].copy()
    if sub.empty:
        return sub

    if side == "home":
        sub["odds"]            = sub["home_odds"]
        sub["xg_diff"]         = sub["xg_diff_home"]
        sub["attack_vs_def"]   = sub["attack_vs_def_home"]
        sub["glicko_prob"]     = sub["home_glicko_prob"]
        sub["market_prob"]     = sub["home_market_prob"]
        sub["won"]             = (sub["result"] == "H").astype(int)
    else:
        sub["odds"]            = sub["away_odds"]
        sub["xg_diff"]         = sub["xg_diff_away"]
        sub["attack_vs_def"]   = sub["attack_vs_def_away"]
        sub["glicko_prob"]     = sub["away_glicko_prob"]
        sub["market_prob"]     = sub["away_market_prob"]
        sub["glicko_gap"]      = -sub["glicko_gap"]
        sub["form_advantage"]  = -sub["form_advantage"]
        sub["won"]             = (sub["result"] == "A").astype(int)

    # Merge with pure_features.parquet for momentum/style/streaks/h2h/rest/ppg
    pf = pd.read_parquet(REPORTS / "pure_features.parquet")
    pf_side = pf[pf["side"] == side][[
        "match_id", "ppg_10", "xg_trend", "glicko_momentum",
        "win_streak", "lose_streak",
        "possession_10", "sot_10", "pass_acc_10",
        "rest_advantage", "h2h_wr", "form_advantage",
    ]].copy()
    pf_side = pf_side.rename(columns={"form_advantage": "_form_adv_check"})
    sub = sub.merge(pf_side, on="match_id", how="left", suffixes=("", "_pf"))

    # opp_lose_streak from other side
    pf_other = pf[pf["side"] != side][["match_id", "lose_streak"]].rename(
        columns={"lose_streak": "opp_lose_streak"}
    )
    sub = sub.merge(pf_other, on="match_id", how="left")

    sub["date"] = pd.to_datetime(sub["date"])
    return sub


def _apply_thresholds(sub: pd.DataFrame, niche: dict) -> np.ndarray:
    """Return boolean mask of rows that pass odds_range + all niche thresholds."""
    lo, hi = niche["odds_range"]
    mask = sub["odds"].between(lo, hi).to_numpy()

    checks = [
        ("glicko_gap",       "min_glicko_gap",      ">="),
        ("glicko_prob",      "min_glicko_prob",     ">="),
        ("xg_diff",          "min_xg_diff",         ">="),
        ("attack_vs_def",    "min_attack_vs_def",   ">="),
        ("form_advantage",   "min_form_advantage",  ">="),
        ("ppg_10",           "min_ppg",             ">="),
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
    for col, key, op in checks:
        thr = niche.get(key)
        if thr is None:
            continue
        if col not in sub.columns:
            # Column missing entirely — drop everything (cannot evaluate)
            return np.zeros(len(sub), dtype=bool)
        col_arr = sub[col].to_numpy(dtype=float)
        valid = ~np.isnan(col_arr)
        if op == ">=":
            mask &= valid & (col_arr >= thr)
        else:
            mask &= valid & (col_arr <= thr)
    return mask


def _stats_from_mask(sub: pd.DataFrame, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "wins": 0, "p_is": None,
                "p_is_lower_95": 0.0, "avg_odds": None,
                "ev_point": None, "ev_lower_95": None}
    won = sub["won"].to_numpy()[mask]
    odds = sub["odds"].to_numpy()[mask]
    wins = int(won.sum())
    p = wins / n
    avg_odds = float(odds.mean())
    lo95 = wilson_lower(wins, n)
    return {
        "n":             n,
        "wins":          wins,
        "p_is":          float(p),
        "p_is_lower_95": lo95,
        "avg_odds":      avg_odds,
        "ev_point":      p * avg_odds - 1,
        "ev_lower_95":   lo95 * avg_odds - 1,
    }


def evaluate_niche(df_factors: pd.DataFrame, niche: dict,
                   oos_start: str | None = OOS_START) -> dict:
    """Apply niche to per-side dataframe, return n/wins/p_is/p_lower stats.

    Args:
        df_factors: per-match factor table (must include `date` column).
        niche: parsed niche dict (must include `_league`).
        oos_start: ISO date string. Rows with date >= oos_start are EXCLUDED
            from the IS computation and reported separately as p_oos / n_oos.
            Pass None to compute on the full history (legacy behaviour).

    Returns:
        Dict with IS keys (n, wins, p_is, p_is_lower_95, avg_odds, ev_*),
        and OOS diagnostic keys (n_oos, wins_oos, p_oos, avg_odds_oos)
        when oos_start is provided.
    """
    sub = _build_side_df(df_factors, niche)
    if sub.empty:
        out = {"n": 0, "wins": 0, "p_is": None, "p_is_lower_95": 0.0,
               "avg_odds": None, "ev_point": None, "ev_lower_95": None}
        if oos_start is not None:
            out.update({"n_oos": 0, "wins_oos": 0, "p_oos": None,
                        "avg_odds_oos": None})
        return out

    mask_all = _apply_thresholds(sub, niche)

    if oos_start is None:
        # Legacy: full-history evaluation
        return _stats_from_mask(sub, mask_all)

    cutoff = pd.Timestamp(oos_start)
    is_mask  = mask_all & (sub["date"].to_numpy() <  np.datetime64(cutoff))
    oos_mask = mask_all & (sub["date"].to_numpy() >= np.datetime64(cutoff))

    is_stats = _stats_from_mask(sub, is_mask)

    # OOS diagnostic
    n_oos = int(oos_mask.sum())
    if n_oos == 0:
        is_stats.update({"n_oos": 0, "wins_oos": 0, "p_oos": None,
                         "avg_odds_oos": None})
    else:
        won_oos  = sub["won"].to_numpy()[oos_mask]
        odds_oos = sub["odds"].to_numpy()[oos_mask]
        w = int(won_oos.sum())
        is_stats.update({
            "n_oos":        n_oos,
            "wins_oos":     w,
            "p_oos":        float(w / n_oos),
            "avg_odds_oos": float(odds_oos.mean()),
        })
    return is_stats


def run(oos_start: str = OOS_START) -> None:
    """Run niche evaluation against IS data only.

    Args:
        oos_start: ISO date string. Default = module-level OOS_START
            ("2025-08-01"), matching Monster's cutoff.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df_factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    df_factors["date"] = pd.to_datetime(df_factors["date"])
    n_total = len(df_factors)
    n_is    = int((df_factors["date"] < pd.Timestamp(oos_start)).sum())
    logger.info(
        f"Loaded {n_total:,} matches "
        f"(IS pre-{oos_start}: {n_is:,}, OOS: {n_total - n_is:,})"
    )

    niches_by_league = parse_all()
    out: dict[str, list[dict]] = {}

    for league, niches in niches_by_league.items():
        league_out = []
        for n in niches:
            n["_league"] = league
            stats = evaluate_niche(df_factors, n, oos_start=oos_start)
            del n["_league"]
            league_out.append({**n, **stats})
        out[league] = league_out

    # Pretty print
    print("\n" + "=" * 110)
    print(f"USER-CURATED NICHES ({sum(len(v) for v in out.values())} total) — "
          f"IS p_is (pre-{oos_start}) + OOS diagnostic")
    print("=" * 110)
    for league, items in out.items():
        print(f"\n  {league}:")
        for it in items:
            n = it.get("n", 0)
            p_is = it.get("p_is")
            lo = it.get("p_is_lower_95", 0)
            ev = it.get("ev_lower_95") or 0
            n_oos = it.get("n_oos", 0)
            p_oos = it.get("p_oos")
            if p_is is not None:
                oos_str = (
                    f"  oos: n={n_oos:>3d} p={p_oos*100:>5.1f}%"
                    if p_oos is not None
                    else f"  oos: n={n_oos:>3d}"
                )
                print(
                    f"    n={n:>4d}  p_is={p_is*100:>5.1f}%  lo95={lo*100:>5.1f}%  "
                    f"avg_odds={it.get('avg_odds') or 0:.2f}  ev_lo={ev*100:+.1f}%"
                    f"{oos_str}   {it['niche_id']}"
                )
            else:
                print(f"    n=   0  (no IS matches)   {it['niche_id']}")

    out_path = ARTIFACTS / "selected_niches_with_pis.json"
    # Convert tuples (odds_range) to lists for JSON
    serializable = {}
    for league, items in out.items():
        serializable[league] = []
        for it in items:
            d = dict(it)
            d["odds_range"] = list(d["odds_range"])
            d["_oos_start"] = oos_start
            serializable[league].append(d)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved → {out_path}  (oos_start={oos_start})")


if __name__ == "__main__":
    import sys
    cutoff = sys.argv[1] if len(sys.argv) > 1 else OOS_START
    run(cutoff)
