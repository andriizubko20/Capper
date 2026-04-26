"""
model/features/missing_xg.py

Feature: missing_xg_share — fraction of a team's total xG contribution that
is currently sidelined by injuries.

We approximate per-player xG contribution from `player_stats.xg_share`
(populated weekly by scheduler.tasks.collect_player_stats from API-Football
top scorers data). For a given match we sum xg_share across the players
listed in InjuryReport for each side.

This module is intentionally a wrapper — we do NOT modify
`model/gem/team_state.py`. Call `compute_missing_xg_share(match_id)` from
the existing feature builder pipeline (see `model/features/builder.py`) and
attach the returned tuple to your feature row.
"""
from __future__ import annotations

from typing import Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.session import SessionLocal


def compute_missing_xg_share(
    match_id: int,
    db: Session | None = None,
) -> Tuple[float, float]:
    """Return (home_share, away_share) — sum of injured players' xg_share
    per side for the given match.

    Returns (0.0, 0.0) if no injuries / no player_stats rows are available.
    Caller may pass an existing Session; otherwise a short-lived one is opened.
    """
    owns_session = db is None
    db = db or SessionLocal()
    try:
        # One query, one round-trip — sums xg_share grouped by side.
        rows = db.execute(text("""
            SELECT
                CASE WHEN ir.team_id = m.home_team_id THEN 'home'
                     WHEN ir.team_id = m.away_team_id THEN 'away'
                END AS side,
                COALESCE(SUM(ps.xg_share), 0.0) AS share
            FROM injury_reports ir
            JOIN matches m ON m.id = ir.match_id
            LEFT JOIN player_stats ps ON ps.player_id = ir.player_api_id
            WHERE ir.match_id = :match_id
              AND ir.team_id IN (m.home_team_id, m.away_team_id)
            GROUP BY side
        """), {"match_id": match_id}).fetchall()
    finally:
        if owns_session:
            db.close()

    home, away = 0.0, 0.0
    for side, share in rows:
        val = float(share or 0.0)
        if side == "home":
            home = val
        elif side == "away":
            away = val
    return home, away


def compute_missing_xg_share_batch(
    match_ids: list[int],
    db: Session | None = None,
) -> dict[int, Tuple[float, float]]:
    """Bulk variant — returns {match_id: (home_share, away_share)}.

    Useful inside the feature-matrix build loop where we materialize many
    matches at once. Missing matches default to (0.0, 0.0).
    """
    if not match_ids:
        return {}

    owns_session = db is None
    db = db or SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                ir.match_id AS match_id,
                CASE WHEN ir.team_id = m.home_team_id THEN 'home'
                     WHEN ir.team_id = m.away_team_id THEN 'away'
                END AS side,
                COALESCE(SUM(ps.xg_share), 0.0) AS share
            FROM injury_reports ir
            JOIN matches m ON m.id = ir.match_id
            LEFT JOIN player_stats ps ON ps.player_id = ir.player_api_id
            WHERE ir.match_id = ANY(:ids)
              AND ir.team_id IN (m.home_team_id, m.away_team_id)
            GROUP BY ir.match_id, side
        """), {"ids": list(match_ids)}).fetchall()
    finally:
        if owns_session:
            db.close()

    out: dict[int, list[float]] = {mid: [0.0, 0.0] for mid in match_ids}
    for match_id, side, share in rows:
        val = float(share or 0.0)
        if side == "home":
            out[match_id][0] = val
        elif side == "away":
            out[match_id][1] = val
    return {mid: (h, a) for mid, (h, a) in out.items()}
