"""
model/features/lineup_strength.py

Lineup-aware team strength feature.

Confirmed starting XI become available ~1h before kickoff via
API-Football /fixtures/lineups (collected by scheduler/tasks/collect_lineups.py).

Idea: per-side strength = Σ player_rating(starter). Until a per-player rating
table exists, we approximate with the team's current Glicko (TeamRating)
scaled by the share of starters that are healthy / available — i.e. for v1
the strength == team Glicko if a confirmed lineup exists, else None.

This keeps the function pluggable: when a per-player rating system lands
the implementation here changes without touching call sites.

Returns None when:
  - no lineup row for the requested side
  - no TeamRating row for the team
Callers (Pure / Monster / Aqua etc.) MUST tolerate None and proceed without it.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from db.models import Lineup, TeamRating

# Minimum starters to consider a lineup "valid" (full XI = 11; allow 10 for
# very early publishes that omit a slot).
MIN_STARTERS = 10


@dataclass(frozen=True)
class LineupStrength:
    """Container for a single side's lineup-derived strength."""
    side: str
    rating: float
    starters: int
    formation: str | None


def _load_lineup(db: Session, match_id: int, side: str) -> Lineup | None:
    return (
        db.query(Lineup)
        .filter(Lineup.match_id == match_id, Lineup.side == side)
        .first()
    )


def _team_glicko(db: Session, team_id: int) -> float | None:
    row = db.query(TeamRating).filter_by(team_id=team_id).first()
    return row.rating if row is not None else None


def compute_lineup_strength(
    db: Session, match_id: int, side: str
) -> float | None:
    """Strength score for one side of a match.

    v1: returns the team's Glicko rating gated on a confirmed lineup with
    at least MIN_STARTERS players. None if no lineup available — caller
    falls back to lineup-agnostic features.
    """
    if side not in ("home", "away"):
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")

    lineup = _load_lineup(db, match_id, side)
    if lineup is None:
        return None
    starters = lineup.starter_player_ids or []
    if len(starters) < MIN_STARTERS:
        logger.debug(
            f"[lineup_strength] match={match_id} side={side} "
            f"only {len(starters)} starters, skip"
        )
        return None

    rating = _team_glicko(db, lineup.team_id)
    if rating is None:
        return None
    return float(rating)


def compute_lineup_strength_diff(db: Session, match_id: int) -> float | None:
    """Home strength minus away strength. None if either side missing."""
    h = compute_lineup_strength(db, match_id, "home")
    a = compute_lineup_strength(db, match_id, "away")
    if h is None or a is None:
        return None
    return h - a


def compute_lineup_strength_pair(
    db: Session, match_id: int
) -> tuple[LineupStrength, LineupStrength] | None:
    """Both sides at once with metadata. None if either side incomplete."""
    out: list[LineupStrength] = []
    for side in ("home", "away"):
        lineup = _load_lineup(db, match_id, side)
        if lineup is None:
            return None
        starters = lineup.starter_player_ids or []
        if len(starters) < MIN_STARTERS:
            return None
        rating = _team_glicko(db, lineup.team_id)
        if rating is None:
            return None
        out.append(LineupStrength(
            side=side,
            rating=float(rating),
            starters=len(starters),
            formation=lineup.formation,
        ))
    return out[0], out[1]
