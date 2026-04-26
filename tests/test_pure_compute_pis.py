"""
Regression tests for model/pure/compute_pis.py OOS_START handling.

The fix this protects: previously p_is was computed on the FULL parquet,
including matches that would later be evaluated as OOS. That caused niche
selection (which is refreshed against the same parquet) to be in-sample
overfit. Now p_is must be computed only on rows with date < OOS_START,
and an OOS-only diagnostic (p_oos / n_oos) must also be returned.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from model.pure import compute_pis


def _make_factors(n_is: int, n_oos: int, win_rate_is: float, win_rate_oos: float,
                  league: str = "Bundesliga",
                  oos_start: str = "2025-08-01") -> pd.DataFrame:
    """Build a synthetic per-match factor frame straddling OOS_START.

    All matches pass the niche thresholds (we use a permissive niche in tests).
    Home wins are encoded via result='H'; losses via 'A'.
    """
    rows = []
    is_dates = pd.date_range(end=pd.Timestamp(oos_start) - pd.Timedelta(days=1),
                             periods=n_is, freq="D")
    oos_dates = pd.date_range(start=pd.Timestamp(oos_start),
                              periods=n_oos, freq="D")

    rng = np.random.default_rng(42)
    is_wins = (rng.random(n_is) < win_rate_is)
    oos_wins = (rng.random(n_oos) < win_rate_oos)

    mid = 0
    for d, won in zip(is_dates, is_wins):
        rows.append({
            "match_id": (mid := mid + 1),
            "date": d,
            "league_name": league,
            "home_odds": 1.9, "away_odds": 4.0,
            "xg_diff_home": 1.0, "xg_diff_away": -1.0,
            "attack_vs_def_home": 1.0, "attack_vs_def_away": 0.5,
            "home_glicko_prob": 0.55, "away_glicko_prob": 0.25,
            "home_market_prob": 0.50, "away_market_prob": 0.30,
            "glicko_gap": 50.0, "form_advantage": 0.5,
            "result": "H" if won else "A",
        })
    for d, won in zip(oos_dates, oos_wins):
        rows.append({
            "match_id": (mid := mid + 1),
            "date": d,
            "league_name": league,
            "home_odds": 1.9, "away_odds": 4.0,
            "xg_diff_home": 1.0, "xg_diff_away": -1.0,
            "attack_vs_def_home": 1.0, "attack_vs_def_away": 0.5,
            "home_glicko_prob": 0.55, "away_glicko_prob": 0.25,
            "home_market_prob": 0.50, "away_market_prob": 0.30,
            "glicko_gap": 50.0, "form_advantage": 0.5,
            "result": "H" if won else "A",
        })
    return pd.DataFrame(rows)


def _make_pure_features(df_factors: pd.DataFrame) -> pd.DataFrame:
    """Build a permissive pure_features parquet so all niche thresholds pass."""
    rows = []
    for mid in df_factors["match_id"].tolist():
        for side in ("home", "away"):
            rows.append({
                "match_id": mid, "side": side,
                "ppg_10": 2.0, "xg_trend": 0.5, "glicko_momentum": 1.0,
                "win_streak": 3, "lose_streak": 0,
                "possession_10": 0.55, "sot_10": 5.0, "pass_acc_10": 0.85,
                "rest_advantage": 1.0, "h2h_wr": 0.6,
                "form_advantage": 0.5,
            })
    return pd.DataFrame(rows)


def _permissive_niche() -> dict:
    """A niche that the synthetic data is designed to pass."""
    return {
        "side": "home",
        "_league": "Bundesliga",
        "odds_range": (1.5, 2.5),
        "min_glicko_gap": None,
        "min_glicko_prob": None,
        "min_xg_diff": None,
        "min_xg_quality_gap": None,
        "min_attack_vs_def": None,
        "min_form_advantage": None,
        "min_ppg": None,
        "min_xg_trend": None,
        "min_glicko_momentum": None,
        "min_win_streak": None,
        "min_opp_lose_streak": None,
        "min_possession_10": None,
        "min_sot_10": None,
        "min_pass_acc_10": None,
        "min_rest_advantage": None,
        "min_h2h_wr": None,
        "max_market_prob": None,
        "niche_id": "home[1.5,2.5)",
    }


@pytest.fixture
def patched_pure_features(tmp_path, monkeypatch):
    """Redirect REPORTS to a tmp dir and stub pure_features.parquet for tests."""
    fake_reports = tmp_path / "reports"
    fake_reports.mkdir()
    monkeypatch.setattr(compute_pis, "REPORTS", fake_reports)
    return fake_reports


def test_p_is_uses_only_pre_oos_rows(patched_pure_features):
    """p_is should match IS-only win rate; OOS rows must NOT contribute."""
    n_is, n_oos = 200, 200
    win_rate_is, win_rate_oos = 0.70, 0.20  # very different so a leak is obvious
    df = _make_factors(n_is, n_oos, win_rate_is, win_rate_oos)
    pf = _make_pure_features(df)
    pf.to_parquet(patched_pure_features / "pure_features.parquet")

    stats = compute_pis.evaluate_niche(df, _permissive_niche(),
                                       oos_start="2025-08-01")

    # IS-only: n must equal n_is (all rows pass the permissive niche).
    assert stats["n"] == n_is, f"expected n={n_is} (IS only), got {stats['n']}"
    # p_is should be close to win_rate_is, NOT the blended ~0.45 of IS+OOS.
    assert 0.62 <= stats["p_is"] <= 0.78, (
        f"p_is={stats['p_is']:.3f} looks like leakage from OOS "
        f"(blended would be ~0.45)"
    )
    # OOS diagnostic should be exposed and reflect the OOS slice.
    assert stats["n_oos"] == n_oos
    assert 0.12 <= stats["p_oos"] <= 0.28, (
        f"p_oos={stats['p_oos']:.3f} doesn't match win_rate_oos={win_rate_oos}"
    )


def test_legacy_full_history_when_oos_start_none(patched_pure_features):
    """Backwards-compat: oos_start=None must use ALL rows (no split)."""
    n_is, n_oos = 100, 100
    df = _make_factors(n_is, n_oos, win_rate_is=0.80, win_rate_oos=0.20)
    pf = _make_pure_features(df)
    pf.to_parquet(patched_pure_features / "pure_features.parquet")

    stats = compute_pis.evaluate_niche(df, _permissive_niche(), oos_start=None)

    assert stats["n"] == n_is + n_oos
    # Blended ~0.5 (avg of 0.8 and 0.2).
    assert 0.40 <= stats["p_is"] <= 0.60
    # Legacy path does not emit OOS diagnostics.
    assert "p_oos" not in stats
    assert "n_oos" not in stats


def test_methodology_guard_positive_ev_at_lower_bound():
    """The recency_check guard rejects thin-sample winners whose lower-95 EV<=0."""
    from model.pure.recency_check import positive_ev_at_lower_bound

    # Strong sample with comfortable margin -> admit
    assert positive_ev_at_lower_bound(p_lower_95=0.60, avg_odds=2.0) is True
    # Borderline at exactly zero EV -> reject (strict >)
    assert positive_ev_at_lower_bound(p_lower_95=0.50, avg_odds=2.0) is False
    # Negative EV at lower bound -> reject
    assert positive_ev_at_lower_bound(p_lower_95=0.40, avg_odds=2.0) is False
    # Missing inputs -> reject
    assert positive_ev_at_lower_bound(p_lower_95=None, avg_odds=2.0) is False
    assert positive_ev_at_lower_bound(p_lower_95=0.60, avg_odds=None) is False
