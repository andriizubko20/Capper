import pandas as pd
import pytest

from model.features.elo import (
    DEFAULT_ELO,
    K_FACTOR,
    build_elo_snapshots,
    compute_dynamic_elo,
    elo_features,
    expected_score,
    update_elo,
)


# ── expected_score ─────────────────────────────────────────────────────────────

def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_a():
    assert expected_score(1600, 1500) > 0.5


def test_expected_score_lower_a():
    assert expected_score(1400, 1500) < 0.5


def test_expected_score_symmetry():
    a = expected_score(1600, 1400)
    b = expected_score(1400, 1600)
    assert a + b == pytest.approx(1.0)


# ── update_elo ─────────────────────────────────────────────────────────────────

def test_home_win_increases_home_elo():
    new_h, new_a = update_elo(1500, 1500, 1, 0)
    assert new_h > 1500
    assert new_a < 1500


def test_away_win_increases_away_elo():
    new_h, new_a = update_elo(1500, 1500, 0, 1)
    assert new_h < 1500
    assert new_a > 1500


def test_draw_equal_teams_no_change():
    new_h, new_a = update_elo(1500, 1500, 0, 0)
    assert new_h == pytest.approx(1500.0)
    assert new_a == pytest.approx(1500.0)


def test_elo_sum_conserved():
    """Total Elo across two teams stays constant."""
    new_h, new_a = update_elo(1600, 1400, 2, 1)
    assert new_h + new_a == pytest.approx(1600 + 1400)


def test_upset_causes_larger_swing():
    """Away win (upset against stronger home) causes larger swing than expected win."""
    new_h_upset, new_a_upset = update_elo(1700, 1300, 0, 1)  # away wins as underdog
    new_h_expected, new_a_expected = update_elo(1300, 1700, 0, 1)  # away wins as favorite
    away_gain_upset = new_a_upset - 1300
    away_gain_expected = new_a_expected - 1700
    assert away_gain_upset > away_gain_expected


# ── elo_features ───────────────────────────────────────────────────────────────

def test_elo_features_diff():
    feats = elo_features(1600, 1500)
    assert feats["elo_diff"] == pytest.approx(100)


def test_elo_features_equal_prob():
    feats = elo_features(1500, 1500)
    assert feats["elo_home_win_prob"] == pytest.approx(0.5)


def test_elo_features_stronger_home():
    feats = elo_features(1700, 1500)
    assert feats["elo_home_win_prob"] > 0.5


# ── build_elo_snapshots ────────────────────────────────────────────────────────

def test_snapshots_no_leakage():
    """First match should use DEFAULT_ELO for both teams."""
    matches = pd.DataFrame([
        {"id": 1, "home_team_id": 1, "away_team_id": 2,
         "home_score": 2, "away_score": 0, "date": pd.Timestamp("2024-01-01")},
    ])
    snapshots = build_elo_snapshots(matches)
    assert snapshots[1][1] == DEFAULT_ELO
    assert snapshots[1][2] == DEFAULT_ELO


def test_snapshots_second_match_uses_updated_elo():
    matches = pd.DataFrame([
        {"id": 1, "home_team_id": 1, "away_team_id": 2,
         "home_score": 2, "away_score": 0, "date": pd.Timestamp("2024-01-01")},
        {"id": 2, "home_team_id": 1, "away_team_id": 3,
         "home_score": 1, "away_score": 1, "date": pd.Timestamp("2024-01-08")},
    ])
    snapshots = build_elo_snapshots(matches)
    assert snapshots[2][1] > DEFAULT_ELO  # team 1 won match 1


# ── compute_dynamic_elo ────────────────────────────────────────────────────────

def test_compute_dynamic_elo_winner_gains():
    matches = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2,
         "home_score": 1, "away_score": 0, "date": pd.Timestamp("2024-01-01")},
    ])
    elos = compute_dynamic_elo(matches)
    assert elos[1] > DEFAULT_ELO
    assert elos[2] < DEFAULT_ELO
