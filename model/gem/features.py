"""
model/gem/features.py

Builds the 50-feature vector for a single match.

No leakage: all inputs are expected to be PRE-match snapshots (see team_state.py).
"""
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


def market_probs_from_odds(
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
) -> dict[str, float | None]:
    """Removes bookmaker margin via normalization to 1.0."""
    if not (home_odds and draw_odds and away_odds):
        return {"home": None, "draw": None, "away": None}
    raw = {"home": 1 / home_odds, "draw": 1 / draw_odds, "away": 1 / away_odds}
    s = sum(raw.values())
    if s <= 0:
        return {"home": None, "draw": None, "away": None}
    return {k: v / s for k, v in raw.items()}


def build_gem_features(
    match_date: datetime,
    league_name: str,
    home_state: dict,
    away_state: dict,
    h2h: dict,
    home_glicko_prob: float | None,   # match_stats.home_win_prob (pre-match Glicko)
    away_glicko_prob: float | None,   # match_stats.away_win_prob
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
    home_has_injuries: bool,
    away_has_injuries: bool,
) -> dict[str, float | None]:
    """
    Returns 50-feature dict for XGBoost.
    Missing values stay as None → XGBoost handles NaN natively.
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

    # ── 7. Market + Glicko reference (10) ─────────────────────────────
    mp = market_probs_from_odds(home_odds, draw_odds, away_odds)
    f["glicko_home_prob"] = home_glicko_prob
    f["glicko_away_prob"] = away_glicko_prob
    # Implied draw when Glicko sum < 1: treat as residual
    glicko_draw = None
    if home_glicko_prob is not None and away_glicko_prob is not None:
        residual = 1.0 - home_glicko_prob - away_glicko_prob
        glicko_draw = residual if residual >= 0 else 0.0
    f["glicko_draw_prob"] = glicko_draw

    f["market_home_odds"] = home_odds
    f["market_draw_odds"] = draw_odds
    f["market_away_odds"] = away_odds
    f["market_home_prob"] = mp["home"]
    f["market_draw_prob"] = mp["draw"]
    f["market_away_prob"] = mp["away"]
    f["glicko_minus_market_home"] = _diff(home_glicko_prob, mp["home"])  # central gem signal

    # ── 8. League context (2 + one-hot) ───────────────────────────────
    f["league_cluster_top5"]   = int(league_cluster(league_name) == "top5_ucl")
    f["league_cluster_second"] = int(league_cluster(league_name) == "second_tier")
    for lg in LEAGUE_NAMES_ORDERED:
        key = _league_feature_key(lg)
        f[key] = int(league_name == lg)

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
        "market_home_odds", "market_draw_odds", "market_away_odds",
        "market_home_prob", "market_draw_prob", "market_away_prob",
        "glicko_minus_market_home",
        "league_cluster_top5", "league_cluster_second",
    ]
    leagues = [_league_feature_key(lg) for lg in LEAGUE_NAMES_ORDERED]
    return base + leagues
