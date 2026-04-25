"""
model/pure/compute_pis.py

Compute historical p_is + Wilson lower-95 for each user-curated niche
and save to selected_niches_with_pis.json. Used by scheduler at pick time
to compute EV / Kelly stake.
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


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def evaluate_niche(df_factors: pd.DataFrame, niche: dict) -> dict:
    """Apply niche to per-side dataframe, return n/wins/p_is/p_lower."""
    side = niche["side"]
    sub = df_factors[df_factors["league_name"] == niche["_league"]].copy()
    if sub.empty:
        return {"n": 0, "wins": 0, "p_is": None, "p_is_lower_95": 0.0, "avg_odds": None}

    if side == "home":
        sub["odds"]            = sub["home_odds"]
        sub["xg_diff"]         = sub["xg_diff_home"]
        sub["attack_vs_def"]   = sub["attack_vs_def_home"]
        sub["glicko_prob"]     = sub["home_glicko_prob"]
        sub["market_prob"]     = sub["home_market_prob"]
        sub["won"]             = (sub["result"] == "H").astype(int)
        # We need ppg, momentum, streaks, style, rest from gem.team_state via pure_features
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
    pf_other = pf[pf["side"] != side][["match_id", "lose_streak"]].rename(columns={"lose_streak": "opp_lose_streak"})
    sub = sub.merge(pf_other, on="match_id", how="left")

    # Apply odds_range
    lo, hi = niche["odds_range"]
    mask = sub["odds"].between(lo, hi)

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
        col_arr = sub[col].to_numpy(dtype=float)
        valid = ~np.isnan(col_arr)
        if op == ">=":
            mask &= valid & (col_arr >= thr)
        else:
            mask &= valid & (col_arr <= thr)

    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "wins": 0, "p_is": None, "p_is_lower_95": 0.0, "avg_odds": None}

    won = sub["won"].to_numpy()[mask]
    odds = sub["odds"].to_numpy()[mask]
    wins = int(won.sum())
    return {
        "n":              n,
        "wins":           wins,
        "p_is":           float(wins / n),
        "p_is_lower_95":  wilson_lower(wins, n),
        "avg_odds":       float(odds.mean()),
        "ev_point":       float(wins / n) * float(odds.mean()) - 1,
        "ev_lower_95":    wilson_lower(wins, n) * float(odds.mean()) - 1,
    }


def run() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df_factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    logger.info(f"Loaded {len(df_factors):,} matches")

    niches_by_league = parse_all()
    out: dict[str, list[dict]] = {}

    for league, niches in niches_by_league.items():
        league_out = []
        for n in niches:
            n["_league"] = league
            stats = evaluate_niche(df_factors, n)
            del n["_league"]
            league_out.append({**n, **stats})
        out[league] = league_out

    # Pretty print
    print("\n" + "=" * 100)
    print(f"USER-CURATED NICHES ({sum(len(v) for v in out.values())} total) — historical p_is")
    print("=" * 100)
    for league, items in out.items():
        print(f"\n  {league}:")
        for it in items:
            n = it.get("n", 0)
            p_is = it.get("p_is")
            lo = it.get("p_is_lower_95", 0)
            ev = it.get("ev_lower_95", 0)
            print(
                f"    n={n:>4d}  p_is={p_is*100:>5.1f}%  lo95={lo*100:>5.1f}%  "
                f"avg_odds={it.get('avg_odds') or 0:.2f}  ev_lo={ev*100:+.1f}%   {it['niche_id']}"
                if p_is is not None else
                f"    n=   0  (no historical matches)   {it['niche_id']}"
            )

    out_path = ARTIFACTS / "selected_niches_with_pis.json"
    # Convert tuples (odds_range) to lists for JSON
    serializable = {}
    for league, items in out.items():
        serializable[league] = []
        for it in items:
            d = dict(it)
            d["odds_range"] = list(d["odds_range"])
            serializable[league].append(d)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
