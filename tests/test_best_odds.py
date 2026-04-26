"""Unit tests for bookmaker-shopping helper data.best_odds.best_1x2_odds.

Uses a tiny in-memory SQLite session so we don't touch Postgres."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.session import Base
from db.models import League, Match, Odds, Team
from data.best_odds import best_1x2_odds, best_1x2_odds_dict


@pytest.fixture
def db():
    """In-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Minimal scaffolding so Match FKs satisfy
        league = League(id=1, api_id=1, name="Test", country="X", season=2024)
        home = Team(id=1, api_id=1, name="A", country="X", league_id=1, elo=1500.0)
        away = Team(id=2, api_id=2, name="B", country="X", league_id=1, elo=1500.0)
        match = Match(
            id=42, api_id=42, league_id=1,
            home_team_id=1, away_team_id=2,
            date=__import__("datetime").datetime(2024, 1, 1),
            status="Not Started",
        )
        session.add_all([league, home, away, match])
        session.commit()
        yield session
    finally:
        session.close()


def _add_odds(db, *, bm: str, h: float, d: float, a: float, is_closing: bool = False):
    rows = [
        Odds(match_id=42, market="1x2", bookmaker=bm, outcome="home", value=h, is_closing=is_closing),
        Odds(match_id=42, market="1x2", bookmaker=bm, outcome="draw", value=d, is_closing=is_closing),
        Odds(match_id=42, market="1x2", bookmaker=bm, outcome="away", value=a, is_closing=is_closing),
    ]
    db.add_all(rows)
    db.commit()


def test_best_1x2_odds_picks_max_per_outcome(db):
    """Across 3 bookmakers, each side independently picks the best price."""
    _add_odds(db, bm="Bet365",     h=1.80, d=3.40, a=3.80)
    _add_odds(db, bm="Pinnacle",   h=1.85, d=3.30, a=3.95)  # best home + best away
    _add_odds(db, bm="William Hill", h=1.78, d=3.50, a=3.70)  # best draw

    home, draw, away = best_1x2_odds(db, match_id=42)

    assert home == 1.85
    assert draw == 3.50
    assert away == 3.95


def test_best_1x2_odds_returns_none_when_no_rows(db):
    home, draw, away = best_1x2_odds(db, match_id=42)
    assert (home, draw, away) == (None, None, None)


def test_best_1x2_odds_skips_invalid_prices(db):
    """Odds <= 1.0 are skipped (invalid price); others picked normally."""
    _add_odds(db, bm="Bet365", h=1.80, d=3.40, a=3.80)
    # Garbage row: a price of 1.0 should be ignored
    db.add(Odds(match_id=42, market="1x2", bookmaker="Junk",
                outcome="home", value=1.0, is_closing=False))
    db.commit()

    home, _, _ = best_1x2_odds(db, match_id=42)
    assert home == 1.80  # not 1.0


def test_best_1x2_odds_filters_by_is_closing(db):
    """is_closing=False (default) excludes closing rows; True selects them."""
    _add_odds(db, bm="Bet365", h=1.80, d=3.40, a=3.80, is_closing=False)
    _add_odds(db, bm="Pinnacle", h=2.50, d=3.10, a=2.90, is_closing=True)

    h_pre, _, _ = best_1x2_odds(db, match_id=42, is_closing=False)
    assert h_pre == 1.80

    h_close, _, _ = best_1x2_odds(db, match_id=42, is_closing=True)
    assert h_close == 2.50

    # is_closing=None ignores the flag → max across both
    h_any, _, _ = best_1x2_odds(db, match_id=42, is_closing=None)
    assert h_any == 2.50


def test_best_1x2_odds_dict_only_present_outcomes(db):
    """Dict variant omits missing outcomes rather than returning None placeholders."""
    db.add(Odds(match_id=42, market="1x2", bookmaker="X",
                outcome="home", value=1.90, is_closing=False))
    db.commit()

    out = best_1x2_odds_dict(db, match_id=42)
    assert out == {"home": 1.90}
