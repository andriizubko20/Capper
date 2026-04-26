"""
Unit tests for model.gem.movement_filter.

Uses a fake DB session that mimics the SQLAlchemy `execute(text(...)).fetchall()`
contract just enough for movement_signals to consume it — no real DB needed.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from model.gem import movement_filter as mf


# ── Fake DB plumbing ────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeDB:
    """Returns the canned rows on every execute() call."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, *args, **kwargs):
        return _FakeResult(self._rows)


def _row(bookmaker, outcome, value, opening, recorded_at):
    return SimpleNamespace(
        bookmaker=bookmaker,
        outcome=outcome,
        value=value,
        opening_value=opening,
        recorded_at=recorded_at,
    )


# ── movement_signals ────────────────────────────────────────────────────


def test_signals_empty_match_returns_none_dict():
    sigs = mf.movement_signals(_FakeDB([]), match_id=1)
    assert sigs["drift_home"] is None
    assert sigs["velocity_home"] is None
    assert sigs["dispersion_home"] is None
    assert sigs["n_snapshots"] == 0
    assert sigs["n_bookmakers"] == 0


def test_signals_drift_positive_when_odds_widen():
    """Odds widening (1.95 → 2.10) ⇒ positive drift on that side."""
    base = datetime(2026, 4, 26, 12, 0)
    rows = [
        _row("Bk1", "home", 1.95, 1.95, base),
        _row("Bk1", "home", 2.00, 1.95, base + timedelta(minutes=30)),
        _row("Bk1", "home", 2.10, 1.95, base + timedelta(minutes=60)),
        _row("Bk2", "home", 1.97, 1.97, base),
        _row("Bk2", "home", 2.05, 1.97, base + timedelta(minutes=30)),
        _row("Bk2", "home", 2.12, 1.97, base + timedelta(minutes=60)),
    ]
    sigs = mf.movement_signals(_FakeDB(rows), match_id=1)
    assert sigs["drift_home"] is not None
    assert sigs["drift_home"] > 0.05  # > 5%
    assert sigs["n_snapshots"] == 3
    assert sigs["n_bookmakers"] == 2


def test_signals_drift_negative_when_odds_shorten():
    """Odds shortening (2.10 → 1.95) ⇒ negative drift = market moved TOWARD us."""
    base = datetime(2026, 4, 26, 12, 0)
    rows = [
        _row("Bk1", "home", 2.10, 2.10, base),
        _row("Bk1", "home", 2.00, 2.10, base + timedelta(minutes=30)),
        _row("Bk1", "home", 1.90, 2.10, base + timedelta(minutes=60)),
    ]
    sigs = mf.movement_signals(_FakeDB(rows), match_id=1)
    assert sigs["drift_home"] is not None
    assert sigs["drift_home"] < 0


def test_signals_dispersion_high_with_disagreeing_books():
    base = datetime(2026, 4, 26, 12, 0)
    rows = [
        _row("Bk1", "home", 1.50, 1.50, base),
        _row("Bk1", "home", 1.50, 1.50, base + timedelta(minutes=30)),
        _row("Bk1", "home", 1.50, 1.50, base + timedelta(minutes=60)),
        _row("Bk2", "home", 2.50, 2.50, base),
        _row("Bk2", "home", 2.50, 2.50, base + timedelta(minutes=30)),
        _row("Bk2", "home", 2.50, 2.50, base + timedelta(minutes=60)),
    ]
    sigs = mf.movement_signals(_FakeDB(rows), match_id=1)
    # std/mean for [1.5, 2.5] is 0.5 / 2.0 = 0.25 (well above 10%)
    assert sigs["dispersion_home"] is not None
    assert sigs["dispersion_home"] > 0.10


def test_signals_dispersion_none_with_single_book():
    base = datetime(2026, 4, 26, 12, 0)
    rows = [
        _row("Bk1", "home", 2.00, 2.00, base),
        _row("Bk1", "home", 2.05, 2.00, base + timedelta(minutes=30)),
        _row("Bk1", "home", 2.10, 2.00, base + timedelta(minutes=60)),
    ]
    sigs = mf.movement_signals(_FakeDB(rows), match_id=1)
    assert sigs["dispersion_home"] is None
    assert sigs["n_books_home"] == 1


# ── should_skip_pick ────────────────────────────────────────────────────


def _signals_with_drift(side: str, drift: float, n_snaps: int = 5, n_books: int = 3):
    base = {f"drift_{s}": None for s in mf.SIDES}
    for s in mf.SIDES:
        base[f"velocity_{s}"]   = None
        base[f"dispersion_{s}"] = None
        base[f"n_books_{s}"]    = 0
        base[f"n_snaps_{s}"]    = 0
    base[f"drift_{side}"]    = drift
    base[f"n_books_{side}"]  = n_books
    base[f"n_snaps_{side}"]  = n_snaps
    base["n_snapshots"]      = n_snaps
    base["n_bookmakers"]     = n_books
    return base


def test_skip_when_drift_against_us_above_threshold():
    sigs = _signals_with_drift("home", 0.10)  # 10% drift against home pick
    skip, reason = mf.should_skip_pick(sigs, "home")
    assert skip is True
    assert "drift" in reason


def test_no_skip_when_drift_negative():
    sigs = _signals_with_drift("home", -0.10)  # toward us
    skip, _ = mf.should_skip_pick(sigs, "home")
    assert skip is False


def test_no_skip_when_drift_below_threshold():
    sigs = _signals_with_drift("home", 0.02)  # 2% — within noise
    skip, _ = mf.should_skip_pick(sigs, "home")
    assert skip is False


def test_no_skip_when_data_too_sparse():
    sigs = _signals_with_drift("home", 0.20, n_snaps=1)  # only 1 snapshot
    skip, _ = mf.should_skip_pick(sigs, "home")
    assert skip is False


def test_no_skip_for_draw_pick():
    """Gem never picks draws, but defensive: filter is no-op for draw."""
    sigs = _signals_with_drift("draw", 0.20)
    skip, _ = mf.should_skip_pick(sigs, "draw")
    assert skip is False


def test_skip_when_velocity_against_us_above_threshold():
    sigs = _signals_with_drift("home", 0.0)
    sigs["velocity_home"] = 0.05  # 5% in last 30 min
    skip, reason = mf.should_skip_pick(sigs, "home")
    assert skip is True
    assert "velocity" in reason


def test_skip_when_dispersion_high_with_enough_books():
    sigs = _signals_with_drift("home", 0.0, n_books=3)
    sigs["dispersion_home"] = 0.20
    skip, reason = mf.should_skip_pick(sigs, "home")
    assert skip is True
    assert "dispersion" in reason


def test_no_skip_when_dispersion_high_but_few_books():
    sigs = _signals_with_drift("home", 0.0, n_books=1)
    sigs["dispersion_home"] = 0.20
    skip, _ = mf.should_skip_pick(sigs, "home")
    assert skip is False


def test_filter_disabled(monkeypatch):
    monkeypatch.setattr(mf, "ENABLE_MOVEMENT_FILTER", False)
    sigs = _signals_with_drift("home", 0.30)
    skip, _ = mf.should_skip_pick(sigs, "home")
    assert skip is False
