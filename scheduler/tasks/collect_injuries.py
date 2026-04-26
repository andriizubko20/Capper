"""
scheduler/tasks/collect_injuries.py

Daily task — pull current injuries from API-Football for every team in the
target leagues and refresh InjuryReport rows.

Strategy
--------
* For each upcoming match (next 7 days), call /injuries?team={api_id}.
  Each call costs 1 request, so we deduplicate per team_id within the run.
* Replace stale rows: any (match_id, team_id) tuple that we re-fetch is
  cleared first, then re-inserted.
* The free tier allows ~100 calls/day — TRACKED_LEAGUES x ~20 teams stays
  well within budget when we restrict to teams that actually play soon.

Schedule once per day (suggested 03:30 UTC).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy.orm import Session

from data.api_football_client import APIFootballClient, APIFootballError
from db.models import InjuryReport, Match, Team
from db.session import SessionLocal


DELAY = 1.5         # seconds between requests (free tier safety)
LOOKAHEAD_DAYS = 7  # only refresh teams with a fixture in the next N days


def _collect_team_ids_for_upcoming(db: Session) -> dict[int, list[tuple[int, int]]]:
    """Return {team_api_id: [(match_id, team_db_id), ...]} for upcoming matches.

    We need this mapping so a single /injuries call per team can populate
    multiple InjuryReport rows (one per upcoming match for that team).
    """
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)

    matches = (
        db.query(Match)
        .filter(Match.date >= now, Match.date <= horizon)
        .all()
    )

    by_team: dict[int, list[tuple[int, int]]] = {}
    for m in matches:
        for team_db_id in (m.home_team_id, m.away_team_id):
            team = db.get(Team, team_db_id)
            if team is None:
                continue
            by_team.setdefault(team.api_id, []).append((m.id, team_db_id))
    return by_team


def _refresh_injuries_for_team(
    db: Session,
    client: APIFootballClient,
    team_api_id: int,
    matches_for_team: list[tuple[int, int]],
) -> int:
    """Fetch injuries for a single team and upsert InjuryReport rows.

    Returns the number of injury rows written.
    """
    try:
        injuries = client.get_injuries(team_api_id)
    except APIFootballError as e:
        logger.warning(f"injuries fetch failed for team api_id={team_api_id}: {e}")
        return 0

    written = 0
    for match_id, team_db_id in matches_for_team:
        # Wipe existing rows for this (match, team) so stale players disappear
        db.query(InjuryReport).filter_by(match_id=match_id, team_id=team_db_id).delete()

        for item in injuries:
            player = item.get("player") or {}
            player_api_id = player.get("id")
            player_name = player.get("name") or "Unknown"
            reason = (player.get("reason") or item.get("type"))
            if player_api_id is None:
                continue

            # Truncate to schema limits to avoid String overflows
            db.add(InjuryReport(
                match_id=match_id,
                team_id=team_db_id,
                player_api_id=int(player_api_id),
                player_name=str(player_name)[:150],
                reason=(str(reason)[:255] if reason else None),
            ))
            written += 1

    db.commit()
    return written


def run_collect_injuries() -> None:
    logger.info("Starting daily injuries collection")
    db = SessionLocal()
    try:
        by_team = _collect_team_ids_for_upcoming(db)
        logger.info(
            f"Refreshing injuries for {len(by_team)} unique teams "
            f"across upcoming {LOOKAHEAD_DAYS}d window"
        )

        with APIFootballClient() as client:
            total_rows = 0
            for team_api_id, matches_for_team in by_team.items():
                rows = _refresh_injuries_for_team(
                    db, client, team_api_id, matches_for_team
                )
                total_rows += rows
                time.sleep(DELAY)

        logger.info(f"Daily injuries collection complete — {total_rows} rows written")
    finally:
        db.close()


if __name__ == "__main__":
    run_collect_injuries()
