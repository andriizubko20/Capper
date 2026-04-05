import pandas as pd
import pytest
from model.features.form import compute_form, compute_rest_days
from model.features.xg import compute_xg_features, compute_xg_overperformance
from model.features.odds_features import market_implied_features, remove_overround


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def matches():
    return pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_score": 2, "away_score": 1, "date": pd.Timestamp("2024-01-01")},
        {"home_team_id": 3, "away_team_id": 1, "home_score": 0, "away_score": 0, "date": pd.Timestamp("2024-01-08")},
        {"home_team_id": 1, "away_team_id": 4, "home_score": 1, "away_score": 3, "date": pd.Timestamp("2024-01-15")},
        {"home_team_id": 2, "away_team_id": 1, "home_score": 1, "away_score": 2, "date": pd.Timestamp("2024-01-22")},
        {"home_team_id": 1, "away_team_id": 5, "home_score": 3, "away_score": 0, "date": pd.Timestamp("2024-01-29")},
    ])


@pytest.fixture
def stats():
    return pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_score": 2, "away_score": 1,
         "home_xg": 1.8, "away_xg": 0.9, "date": pd.Timestamp("2024-01-01")},
        {"home_team_id": 3, "away_team_id": 1, "home_score": 0, "away_score": 0,
         "home_xg": 0.7, "away_xg": 0.5, "date": pd.Timestamp("2024-01-08")},
        {"home_team_id": 1, "away_team_id": 4, "home_score": 1, "away_score": 3,
         "home_xg": 1.2, "away_xg": 2.1, "date": pd.Timestamp("2024-01-15")},
    ])


# ── compute_form ──────────────────────────────────────────────────────────────

def test_form_win_gives_max_points(matches):
    result = compute_form(matches, team_id=1, before_date=pd.Timestamp("2024-01-10"))
    # 2 matches: win (3pts) + draw (1pt) → 4/6 = 0.667
    assert pytest.approx(result["form_points"], rel=0.01) == 4 / 6


def test_form_empty_returns_neutral(matches):
    result = compute_form(matches, team_id=99, before_date=pd.Timestamp("2024-01-01"))
    assert result["form_points"] == 0.5
    assert result["form_goals_for_avg"] == 1.3


def test_form_excludes_future_matches(matches):
    result = compute_form(matches, team_id=1, before_date=pd.Timestamp("2024-01-02"))
    # Only 1 match (2024-01-01): win
    assert pytest.approx(result["form_points"], rel=0.01) == 1.0


def test_form_goals_avg(matches):
    result = compute_form(matches, team_id=1, before_date=pd.Timestamp("2024-02-01"))
    # Team 1 goals for: 2, 0, 1, 2, 3 → avg 1.6
    assert pytest.approx(result["form_goals_for_avg"], rel=0.01) == 1.6


# ── compute_rest_days ─────────────────────────────────────────────────────────

def test_rest_days_correct(matches):
    result = compute_rest_days(matches, team_id=1, before_date=pd.Timestamp("2024-02-05"))
    # Last match: 2024-01-29 → 7 days
    assert result["rest_days"] == 7


def test_rest_days_no_matches_returns_default(matches):
    result = compute_rest_days(matches, team_id=99, before_date=pd.Timestamp("2024-01-01"))
    assert result["rest_days"] == 7


def test_rest_days_capped_at_30(matches):
    result = compute_rest_days(matches, team_id=1, before_date=pd.Timestamp("2024-03-10"))
    assert result["rest_days"] == 30


# ── compute_xg_features ───────────────────────────────────────────────────────

def test_xg_features_home_team(stats):
    result = compute_xg_features(stats, team_id=1, before_date=pd.Timestamp("2024-01-10"))
    # Team 1 before 2024-01-10: match1 home xg=1.8, match2 away xg=0.5
    assert pytest.approx(result["xg_for_avg"], rel=0.01) == (1.8 + 0.5) / 2
    assert pytest.approx(result["xg_against_avg"], rel=0.01) == (0.9 + 0.7) / 2


def test_xg_features_empty_returns_neutral(stats):
    result = compute_xg_features(stats, team_id=99, before_date=pd.Timestamp("2024-01-20"))
    assert result["xg_for_avg"] == 1.3
    assert result["xg_against_avg"] == 1.3
    assert result["xg_diff_avg"] == 0.0


def test_xg_features_ratio(stats):
    result = compute_xg_features(stats, team_id=1, before_date=pd.Timestamp("2024-01-20"))
    expected_ratio = result["xg_for_avg"] / result["xg_against_avg"]
    assert pytest.approx(result["xg_ratio"], rel=0.01) == expected_ratio


# ── compute_xg_overperformance ────────────────────────────────────────────────

def test_xg_overperformance_positive(stats):
    # Team 1, match1: scored 2, xg 1.8 → +0.2; match2: scored 0, xg 0.5 → -0.5; match3: scored 1, xg 1.2 → -0.2
    result = compute_xg_overperformance(stats, team_id=1, before_date=pd.Timestamp("2024-02-01"))
    expected = ((2 - 1.8) + (0 - 0.5) + (1 - 1.2)) / 3
    assert pytest.approx(result["xg_overperformance"], rel=0.01) == expected


def test_xg_overperformance_empty_returns_zero(stats):
    result = compute_xg_overperformance(stats, team_id=99, before_date=pd.Timestamp("2024-01-20"))
    assert result["xg_overperformance"] == 0.0


# ── market_implied_features ───────────────────────────────────────────────────

def test_market_probs_sum_to_one(matches):
    result = market_implied_features(2.0, 3.5, 4.0)
    total = result["market_home_prob"] + result["market_draw_prob"] + result["market_away_prob"]
    assert pytest.approx(total, abs=0.001) == 1.0


def test_market_overround_positive():
    result = market_implied_features(1.8, 3.5, 4.5)
    assert result["market_overround"] > 0


def test_favourite_has_highest_prob():
    result = market_implied_features(1.5, 4.0, 6.0)
    assert result["market_home_prob"] > result["market_draw_prob"]
    assert result["market_home_prob"] > result["market_away_prob"]


def test_remove_overround_balanced_odds():
    h, d, a = remove_overround(2.0, 2.0, 2.0)
    assert pytest.approx(h, abs=0.001) == 1 / 3
    assert pytest.approx(d, abs=0.001) == 1 / 3
    assert pytest.approx(a, abs=0.001) == 1 / 3
