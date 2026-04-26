"""
model/gem/features.py

Builds the 50-feature vector for a single match.

No leakage: all inputs are expected to be PRE-match snapshots (see team_state.py).
"""
import math
from datetime import datetime

from model.gem.niches import LEAGUE_NAMES_ORDERED, league_cluster


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _mul(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a * b


def _sign(x: float | None) -> int | None:
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def market_probs_from_odds(
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
) -> dict[str, float | None]:
    """Removes bookmaker margin via proportional normalization to 1.0."""
    if not (home_odds and draw_odds and away_odds):
        return {"home": None, "draw": None, "away": None}
    raw = {"home": 1 / home_odds, "draw": 1 / draw_odds, "away": 1 / away_odds}
    s = sum(raw.values())
    if s <= 0:
        return {"home": None, "draw": None, "away": None}
    return {k: v / s for k, v in raw.items()}


def shin_probs_from_odds(
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
) -> dict[str, float | None]:
    """
    Shin (1992) de-vig: assumes a fraction z ∈ (0, 0.5) of bets come from
    insiders with perfect info. Solves for z such that de-vigged probabilities
    sum to 1.0 exactly. Better than proportional for draw-heavy / favorite-bias
    markets — corrects the long-shot bias that proportional ignores.

    For 3 outcomes with q_i = 1/odds_i, Q = Σ q_i:
        true_p_i = (sqrt(z² + 4(1-z) q_i² / Q) - z) / (2(1-z))

    Solves z by binary search such that Σ true_p_i = 1.

    Falls back to proportional if odds are degenerate or solver fails to
    converge (rare — happens only on near-fair markets where Q ≤ 1).
    """
    if not (home_odds and draw_odds and away_odds):
        return {"home": None, "draw": None, "away": None}
    qs = [1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds]
    Q = sum(qs)
    if Q <= 1.0:
        return {"home": qs[0] / Q, "draw": qs[1] / Q, "away": qs[2] / Q}

    def sum_p(z: float) -> float:
        if z <= 0 or z >= 1:
            return 99.0
        s = 0.0
        for q in qs:
            s += (math.sqrt(z * z + 4 * (1 - z) * q * q / Q) - z) / (2 * (1 - z))
        return s

    # Binary search for z ∈ (1e-6, 0.5) where sum_p(z) = 1.0
    lo, hi = 1e-6, 0.5
    if sum_p(lo) <= 1.0 or sum_p(hi) >= 1.0:
        # Fallback: market is too tight or too loose for Shin
        return {"home": qs[0] / Q, "draw": qs[1] / Q, "away": qs[2] / Q}
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum_p(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    p = [
        (math.sqrt(z * z + 4 * (1 - z) * q * q / Q) - z) / (2 * (1 - z))
        for q in qs
    ]
    return {"home": p[0], "draw": p[1], "away": p[2]}


def market_probs(
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
    method: str = "proportional",
) -> dict[str, float | None]:
    """Dispatch to proportional or Shin de-vig."""
    if method == "shin":
        return shin_probs_from_odds(home_odds, draw_odds, away_odds)
    return market_probs_from_odds(home_odds, draw_odds, away_odds)


def build_gem_features(
    match_date: datetime,
    league_name: str,
    home_state: dict,
    away_state: dict,
    h2h: dict,
    home_glicko_prob: float | None,   # match_stats.home_win_prob (pre-match Glicko)
    away_glicko_prob: float | None,   # match_stats.away_win_prob
    home_has_injuries: bool,
    away_has_injuries: bool,
    league_priors: dict | None = None,  # target-encoded league rates from training fold
) -> dict[str, float | None]:
    """
    Returns feature dict for XGBoost/LGB/CatBoost.
    Missing values stay as None → tree models handle NaN natively.

    Market-odds features were removed in audit: 94% of historical odds rows
    were recorded post-match (closing), so using them as features leaks
    post-match information. Odds are used only for simulation + inference gem filter.
    """
    f: dict[str, float | None] = {}

    # ── 1. Raw strength (13) ───────────────────────────────────────────
    f["home_xg_for"]     = home_state["xg_for_10"]
    f["home_xg_against"] = home_state["xg_against_10"]
    f["home_xg_diff"]    = _diff(home_state["xg_for_10"], home_state["xg_against_10"])
    f["away_xg_for"]     = away_state["xg_for_10"]
    f["away_xg_against"] = away_state["xg_against_10"]
    f["away_xg_diff"]    = _diff(away_state["xg_for_10"], away_state["xg_against_10"])
    f["xg_diff_gap"]     = _diff(f["home_xg_diff"], f["away_xg_diff"])

    f["home_glicko"]  = home_state["glicko_now"]
    f["away_glicko"]  = away_state["glicko_now"]
    f["glicko_gap"]   = _diff(home_state["glicko_now"], away_state["glicko_now"])

    f["home_ppg"]   = home_state["ppg_10"]
    f["away_ppg"]   = away_state["ppg_10"]
    f["ppg_gap"]    = _diff(home_state["ppg_10"], away_state["ppg_10"])

    # ── 2. Home/Away splits (6) ────────────────────────────────────────
    f["home_xg_diff_at_home"] = _diff(home_state["xg_for_home_10"], home_state["xg_against_home_10"])
    f["away_xg_diff_on_road"] = _diff(away_state["xg_for_away_10"], away_state["xg_against_away_10"])
    f["context_xg_gap"] = _diff(f["home_xg_diff_at_home"], f["away_xg_diff_on_road"])

    f["home_ppg_at_home"] = home_state["ppg_home_10"]
    f["away_ppg_on_road"] = away_state["ppg_away_10"]
    f["context_ppg_gap"]  = _diff(home_state["ppg_home_10"], away_state["ppg_away_10"])

    # ── 3. Short-term momentum (12) ───────────────────────────────────
    f["home_form_5"]       = home_state["form_5"]
    f["away_form_5"]       = away_state["form_5"]
    f["form_gap"]          = _diff(home_state["form_5"], away_state["form_5"])
    f["home_xg_trend"]     = home_state["xg_trend"]
    f["away_xg_trend"]     = away_state["xg_trend"]
    f["home_win_streak"]   = home_state["win_streak"]
    f["home_lose_streak"]  = home_state["lose_streak"]
    f["away_win_streak"]   = away_state["win_streak"]
    f["away_lose_streak"]  = away_state["lose_streak"]
    f["home_glicko_momentum"]  = home_state["glicko_momentum"]
    f["away_glicko_momentum"]  = away_state["glicko_momentum"]
    f["glicko_momentum_gap"]   = _diff(home_state["glicko_momentum"], away_state["glicko_momentum"])

    # ── 4. Style (6) ──────────────────────────────────────────────────
    f["home_possession"] = home_state["possession_10"]
    f["away_possession"] = away_state["possession_10"]
    f["home_sot"]        = home_state["sot_10"]
    f["away_sot"]        = away_state["sot_10"]
    f["home_pass_acc"]   = home_state["pass_acc_10"]
    f["away_pass_acc"]   = away_state["pass_acc_10"]

    # ── 5. H2H (3) ────────────────────────────────────────────────────
    f["h2h_home_wr"]           = h2h.get("h2h_home_wr")
    f["h2h_avg_goals"]         = h2h.get("h2h_avg_goals")
    f["h2h_home_last_result"]  = h2h.get("h2h_home_last_result")

    # ── 6. Physical (5 — removed suspensions per decision) ────────────
    home_rest = None
    if home_state["last_match_date"] is not None:
        home_rest = (match_date - home_state["last_match_date"]).days
    away_rest = None
    if away_state["last_match_date"] is not None:
        away_rest = (match_date - away_state["last_match_date"]).days
    f["home_rest_days"] = home_rest
    f["away_rest_days"] = away_rest
    f["rest_gap"]       = _diff(home_rest, away_rest)
    f["home_has_any_injuries"] = int(bool(home_has_injuries))
    f["away_has_any_injuries"] = int(bool(away_has_injuries))

    # ── 7. Glicko pre-match probabilities (3 — market features removed) ──
    f["glicko_home_prob"] = home_glicko_prob
    f["glicko_away_prob"] = away_glicko_prob
    glicko_draw = None
    if home_glicko_prob is not None and away_glicko_prob is not None:
        residual = 1.0 - home_glicko_prob - away_glicko_prob
        glicko_draw = residual if residual >= 0 else 0.0
    f["glicko_draw_prob"] = glicko_draw

    # ── 8. League context (2 binary clusters + 3 target-encoded priors + one-hot) ─
    f["league_cluster_top5"]   = int(league_cluster(league_name) == "top5_ucl")
    f["league_cluster_second"] = int(league_cluster(league_name) == "second_tier")
    if league_priors is not None:
        f["league_prior_home_wr"]  = league_priors.get("home_wr")
        f["league_prior_draw_rate"] = league_priors.get("draw_rate")
        f["league_prior_away_wr"]  = league_priors.get("away_wr")
    else:
        f["league_prior_home_wr"]  = None
        f["league_prior_draw_rate"] = None
        f["league_prior_away_wr"]  = None
    for lg in LEAGUE_NAMES_ORDERED:
        key = _league_feature_key(lg)
        f[key] = int(league_name == lg)

    # ── 9. Composite / interaction features (v2) ──────────────────────
    # dominance_score: multiplicative interaction between strength gap and quality gap
    f["dominance_score"] = _mul(f["glicko_gap"], f["xg_diff_gap"])

    # Direct attack-vs-defense quality matchup
    home_attack_minus_away_def  = _diff(home_state["xg_for_10"], away_state["xg_against_10"])
    away_attack_minus_home_def  = _diff(away_state["xg_for_10"], home_state["xg_against_10"])
    f["xg_quality_gap"] = _diff(home_attack_minus_away_def, away_attack_minus_home_def)

    # Momentum alignment: count of momentum signals agreeing on direction (-3..+3)
    form_gap_sign      = _sign(f.get("form_gap"))
    xg_trend_gap_sign  = _sign(_diff(home_state["xg_trend"], away_state["xg_trend"]))
    glicko_mom_sign    = _sign(f.get("glicko_momentum_gap"))
    if None in (form_gap_sign, xg_trend_gap_sign, glicko_mom_sign):
        f["momentum_alignment_score"] = None
    else:
        f["momentum_alignment_score"] = form_gap_sign + xg_trend_gap_sign + glicko_mom_sign

    # Home advantage strength: how much team's home PPG outperforms league's home WR baseline
    # league_prior_home_wr is in [0,1]; home_ppg_at_home is in [0,3] — scale prior by 3 for comparable units
    if home_state["ppg_home_10"] is not None and f.get("league_prior_home_wr") is not None:
        f["home_advantage_factor"] = home_state["ppg_home_10"] - f["league_prior_home_wr"] * 3.0
    else:
        f["home_advantage_factor"] = None

    # Away "hotness" trap: hot away side often overrated by market — negative signal for OUR home pick
    f["away_hotness_signal"] = _mul(away_state.get("win_streak"), away_state.get("xg_trend"))

    return f


def _league_feature_key(league_name: str) -> str:
    return "is_" + league_name.lower().replace(" ", "_").replace(".", "").replace("ü", "u")


def expected_feature_names() -> list[str]:
    """Stable ordering of features — used at training to build numpy matrix."""
    base = [
        "home_xg_for", "home_xg_against", "home_xg_diff",
        "away_xg_for", "away_xg_against", "away_xg_diff", "xg_diff_gap",
        "home_glicko", "away_glicko", "glicko_gap",
        "home_ppg", "away_ppg", "ppg_gap",
        "home_xg_diff_at_home", "away_xg_diff_on_road", "context_xg_gap",
        "home_ppg_at_home", "away_ppg_on_road", "context_ppg_gap",
        "home_form_5", "away_form_5", "form_gap",
        "home_xg_trend", "away_xg_trend",
        "home_win_streak", "home_lose_streak",
        "away_win_streak", "away_lose_streak",
        "home_glicko_momentum", "away_glicko_momentum", "glicko_momentum_gap",
        "home_possession", "away_possession",
        "home_sot", "away_sot",
        "home_pass_acc", "away_pass_acc",
        "h2h_home_wr", "h2h_avg_goals", "h2h_home_last_result",
        "home_rest_days", "away_rest_days", "rest_gap",
        "home_has_any_injuries", "away_has_any_injuries",
        "glicko_home_prob", "glicko_away_prob", "glicko_draw_prob",
        "league_cluster_top5", "league_cluster_second",
        "league_prior_home_wr", "league_prior_draw_rate", "league_prior_away_wr",
        "dominance_score", "xg_quality_gap", "momentum_alignment_score",
        "home_advantage_factor", "away_hotness_signal",
    ]
    leagues = [_league_feature_key(lg) for lg in LEAGUE_NAMES_ORDERED]
    return base + leagues
