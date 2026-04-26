"""Bookmaker shopping helpers.

For each pick we want the BEST available price for our chosen side. Since the
SStats feed gives us 8-12 bookmakers per fixture, comparing them and using the
max odds per outcome captures a permanent ~3-5% EV uplift relative to relying
on a single bookmaker.

Note on devig vs stake odds (see scheduler/tasks/generate_picks_*.py):

We use BEST odds for both the market_p (devig) input and the stake odds. This
is the "simple" version:

    - Pros: easy to reason about, single source of truth.
    - Cons: max-of-N odds skews implied prob downward → market_p devig becomes
      a touch more conservative/biased upward for our side. EV is slightly
      inflated as a result.

The Kelly cap (KELLY_CAP) and fractional-Kelly (25%) staking already absorb
this kind of model risk, so we accept the simpler approach. If we ever measure
a concrete bias, switch market_p devig to AVERAGE bookmaker odds while keeping
stake odds as best.
"""
from __future__ import annotations

from typing import Iterable

from db.models import Odds


def best_1x2_odds(
    db,
    match_id: int,
    *,
    is_closing: bool | None = False,
) -> tuple[float | None, float | None, float | None]:
    """Return best (home, draw, away) odds across all bookmakers for this match.

    Each side is picked independently — the best home odd may come from a
    different bookmaker than the best away odd. That is intentional: at bet
    placement time the user (or the strategy) can shop the price per outcome.

    Args:
        db: SQLAlchemy session.
        match_id: internal Match.id (NOT api_id).
        is_closing: filter on Odds.is_closing. Defaults to False (live/pre-match).
            Pass True to query closing-line snapshots, or None to consider both.
    """
    q = db.query(Odds).filter(
        Odds.match_id == match_id,
        Odds.market == "1x2",
    )
    if is_closing is not None:
        q = q.filter(Odds.is_closing == is_closing)

    rows: Iterable[Odds] = q.all()

    best: dict[str, float] = {}
    for r in rows:
        try:
            v = float(r.value)
        except (TypeError, ValueError):
            continue
        if v <= 1.0:
            continue  # impossible price, skip
        cur = best.get(r.outcome)
        if cur is None or v > cur:
            best[r.outcome] = v

    return best.get("home"), best.get("draw"), best.get("away")


def best_1x2_odds_dict(
    db,
    match_id: int,
    *,
    is_closing: bool | None = False,
) -> dict[str, float]:
    """Same as best_1x2_odds but returns a dict {outcome: best_odd}.

    Outcomes with no rows are simply absent from the dict (unlike the tuple
    variant, which returns None placeholders)."""
    h, d, a = best_1x2_odds(db, match_id, is_closing=is_closing)
    out: dict[str, float] = {}
    if h is not None:
        out["home"] = h
    if d is not None:
        out["draw"] = d
    if a is not None:
        out["away"] = a
    return out
