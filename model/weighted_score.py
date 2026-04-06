"""
model/weighted_score.py

Production версія зваженого scenario score.
Ваги розраховані з даних через lift-аналіз (experiments/weighted_score.py).

dyn_A логіка:
    WS >= 100 → min EV = 10%
    WS >= 90  → min EV = 12%
    WS >= 75  → min EV = 16.5%
    WS < 75   → не ставимо
"""

# ---------------------------------------------------------------------------
# Ваги факторів (розраховані автоматично, оновлюються при ретрейнінгу)
# ---------------------------------------------------------------------------

HOME_WEIGHTS: dict[str, int] = {
    "market_favors_home": 11,
    "market_strong_home": 11,
    "xg_ratio_home": 8,
    "elo_gap_large": 8,
    "xg_diff_positive": 8,
    "elo_gap_moderate": 8,
    "elo_win_prob_high": 7,
    "home_elo_strong": 7,
    "table_points_home_better": 7,
    "table_home_higher": 7,
    "home_strong_form": 5,
    "home_home_wins": 5,
    "home_in_form": 5,
    "home_home_form": 5,
    "home_scoring_strong": 4,
    "away_away_poor": 3,
    "away_elo_weak": 3,
    "away_out_of_form": 3,
    "xg_attack_edge": 3,
    "away_poor_form": 3,
    "home_defense_solid": 2,
    "away_conceding_lots": 2,
    "injury_advantage": 1,
    "home_rested": 1,
    "big_injury_advantage": 1,
    "away_away_loses": 1,
    "rest_advantage": 1,
    "home_xg_regression": 1,
    "away_xg_overperforming": 1,
    "away_tired": 1,
}

AWAY_WEIGHTS: dict[str, int] = {
    "market_sees_away": 16,
    "market_strong_away": 14,
    "elo_gap_away_large": 12,
    "elo_gap_away_moderate": 12,
    "elo_win_prob_low": 12,
    "table_points_away_better": 11,
    "table_away_higher": 9,
    "xg_diff_away_positive": 8,
    "xg_ratio_away": 8,
    "away_elo_strong": 8,
    "home_elo_weak": 7,
    "home_out_of_form": 5,
    "home_home_poor": 5,
    "away_scoring_strong": 5,
    "away_in_form": 4,
    "home_poor_form": 4,
    "away_away_form": 4,
    "away_strong_form": 4,
    "away_defense_solid": 3,
    "away_away_wins": 3,
    "home_conceding_lots": 3,
    "xg_away_attack": 2,
    "injury_adv_away": 1,
    "big_injury_adv_away": 1,
    "home_home_loses": 1,
    "away_rested": 1,
    "home_tired": 1,
    "away_xg_regression": 1,
    "rest_advantage_away": 1,
    "home_xg_overperforming": 1,
}

# dyn_A: (min_ws, min_ev) — від вищого до нижчого
EV_BY_WS = [(100, 0.10), (90, 0.12), (75, 0.165)]


# ---------------------------------------------------------------------------
# Фактори
# ---------------------------------------------------------------------------

def _get_factors(features: dict, outcome: str) -> list[tuple[str, bool]]:
    r = features
    if outcome == "home":
        return [
            ("market_favors_home",       r.get("market_home_prob", 0) > 0.50),
            ("market_strong_home",       r.get("market_home_prob", 0) > 0.60),
            ("home_in_form",             r.get("home_form_points", 0.5) > 0.55),
            ("home_strong_form",         r.get("home_form_points", 0.5) > 0.65),
            ("away_out_of_form",         r.get("away_form_points", 0.5) < 0.45),
            ("away_poor_form",           r.get("away_form_points", 0.5) < 0.35),
            ("home_home_form",           r.get("home_home_form_points", 0.5) > 0.55),
            ("home_home_wins",           r.get("home_home_form_wins", 0.33) > 0.60),
            ("away_away_poor",           r.get("away_away_form_points", 0.5) < 0.40),
            ("away_away_loses",          r.get("away_away_form_losses", 0.33) > 0.50),
            ("elo_gap_large",            r.get("elo_diff", 0) > 50),
            ("elo_gap_moderate",         r.get("elo_diff", 0) > 25),
            ("elo_win_prob_high",        r.get("elo_home_win_prob", 0.5) > 0.55),
            ("home_elo_strong",          r.get("home_elo", 1500) > 1600),
            ("away_elo_weak",            r.get("away_elo", 1500) < 1400),
            ("xg_attack_edge",           r.get("home_xg_for_avg_10", 1.3) > r.get("away_xg_against_avg_10", 1.3)),
            ("xg_ratio_home",            (r.get("home_xg_for_avg_10", 1) / max(r.get("home_xg_against_avg_10", 1), 0.1)) > 1.2),
            ("xg_diff_positive",         r.get("home_xg_diff_avg_10", 0) > 0.3),
            ("home_xg_regression",       r.get("home_xg_overperformance", 0) < -0.2),
            ("away_xg_overperforming",   r.get("away_xg_overperformance", 0) > 0.3),
            ("home_scoring_strong",      r.get("home_form_goals_for_avg", 1.3) > 1.8),
            ("away_conceding_lots",      r.get("away_form_goals_against_avg", 1.3) > 1.8),
            ("home_defense_solid",       r.get("home_form_goals_against_avg", 1.3) < 1.2),
            ("table_home_higher",        r.get("table_position_diff", 0) < -3),
            ("table_points_home_better", r.get("table_points_diff", 0) > 5),
            ("home_rested",              r.get("home_rest_days", 7) >= 5),
            ("away_tired",               r.get("away_rest_days", 7) <= 3),
            ("rest_advantage",           r.get("rest_days_diff", 0) >= 3),
            ("injury_advantage",         r.get("away_injured_count", 0) > r.get("home_injured_count", 0)),
            ("big_injury_advantage",     r.get("away_injured_count", 0) - r.get("home_injured_count", 0) >= 3),
        ]
    else:  # away
        return [
            ("market_sees_away",         r.get("market_away_prob", 0) > 0.38),
            ("market_strong_away",       r.get("market_away_prob", 0) > 0.50),
            ("away_in_form",             r.get("away_form_points", 0.5) > 0.55),
            ("away_strong_form",         r.get("away_form_points", 0.5) > 0.65),
            ("home_out_of_form",         r.get("home_form_points", 0.5) < 0.45),
            ("home_poor_form",           r.get("home_form_points", 0.5) < 0.35),
            ("away_away_form",           r.get("away_away_form_points", 0.5) > 0.50),
            ("away_away_wins",           r.get("away_away_form_wins", 0.33) > 0.50),
            ("home_home_poor",           r.get("home_home_form_points", 0.5) < 0.45),
            ("home_home_loses",          r.get("home_home_form_losses", 0.33) > 0.40),
            ("elo_gap_away_large",       r.get("elo_diff", 0) < -50),
            ("elo_gap_away_moderate",    r.get("elo_diff", 0) < -25),
            ("elo_win_prob_low",         r.get("elo_home_win_prob", 0.5) < 0.45),
            ("away_elo_strong",          r.get("away_elo", 1500) > 1600),
            ("home_elo_weak",            r.get("home_elo", 1500) < 1400),
            ("xg_away_attack",           r.get("away_xg_for_avg_10", 1.3) > r.get("home_xg_against_avg_10", 1.3)),
            ("xg_ratio_away",            (r.get("away_xg_for_avg_10", 1) / max(r.get("away_xg_against_avg_10", 1), 0.1)) > 1.2),
            ("xg_diff_away_positive",    r.get("away_xg_diff_avg_10", 0) > 0.3),
            ("away_xg_regression",       r.get("away_xg_overperformance", 0) < -0.2),
            ("home_xg_overperforming",   r.get("home_xg_overperformance", 0) > 0.3),
            ("away_scoring_strong",      r.get("away_form_goals_for_avg", 1.3) > 1.8),
            ("home_conceding_lots",      r.get("home_form_goals_against_avg", 1.3) > 1.8),
            ("away_defense_solid",       r.get("away_form_goals_against_avg", 1.3) < 1.2),
            ("table_away_higher",        r.get("table_position_diff", 0) > 3),
            ("table_points_away_better", r.get("table_points_diff", 0) < -5),
            ("away_rested",              r.get("away_rest_days", 7) >= 5),
            ("home_tired",               r.get("home_rest_days", 7) <= 3),
            ("rest_advantage_away",      r.get("rest_days_diff", 0) <= -3),
            ("injury_adv_away",          r.get("home_injured_count", 0) > r.get("away_injured_count", 0)),
            ("big_injury_adv_away",      r.get("home_injured_count", 0) - r.get("away_injured_count", 0) >= 3),
        ]


def compute_weighted_score(features: dict, outcome: str) -> int:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    total = 0
    for name, active in _get_factors(features, outcome):
        if active:
            total += weights.get(name, 1)
    return total


def get_min_ev(ws: int) -> float | None:
    """dyn_A: повертає мінімальний EV для даного WS або None якщо нижче порогу."""
    for min_ws, min_ev in EV_BY_WS:
        if ws >= min_ws:
            return min_ev
    return None
