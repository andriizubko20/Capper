"""
scheduler/tasks/collect_player_stats.py

Weekly task — pull /players/topscorers per league, persist to player_stats,
and recompute each player's xg_share (their fraction of the team's total
goal contribution, used as a proxy for xG contribution until we have real
per-player xG values).

Schedule once per week (suggested Wed 04:00 UTC).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable

from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from data.api_football_client import APIFootballClient, APIFootballError
from db.models import League, Team
from db.session import SessionLocal


DELAY = 2.0
GOALS_WEIGHT = 1.2  # rough xG-per-goal multiplier; matches feature spec


def _current_seasons(db: Session) -> list[tuple[int, int, int]]:
    """Return [(league_db_id, league_api_id, season), ...] for the latest
    season per league (the most recent integer season we have on record)."""
    rows = db.execute(text("""
        SELECT id, api_id, season
        FROM leagues
        WHERE (api_id, season) IN (
            SELECT api_id, MAX(season) FROM leagues GROUP BY api_id
        )
    """)).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _ensure_player_row(db: Session, player_id: int) -> None:
    """Make sure a row exists in player_stats for this player so we can UPDATE."""
    db.execute(text("""
        INSERT INTO player_stats (player_id) VALUES (:pid)
        ON CONFLICT (player_id) DO NOTHING
    """), {"pid": player_id})


def _upsert_player(
    db: Session,
    player_id: int,
    team_db_id: int | None,
    name: str,
    league_db_id: int,
    season: int,
    goals: int,
    assists: int,
    minutes: int,
) -> None:
    _ensure_player_row(db, player_id)
    db.execute(text("""
        UPDATE player_stats
        SET team_id = :team_id,
            name = :name,
            league_id = :league_id,
            season = :season,
            goals = :goals,
            assists = :assists,
            minutes_played = :minutes,
            updated_at = :now
        WHERE player_id = :pid
    """), {
        "team_id":   team_db_id,
        "name":      name[:200],
        "league_id": league_db_id,
        "season":    season,
        "goals":     goals,
        "assists":   assists,
        "minutes":   minutes,
        "now":       datetime.now(timezone.utc),
        "pid":       player_id,
    })


def _recompute_xg_shares(db: Session, league_db_id: int, season: int) -> None:
    """Compute xg_share = (goals*GOALS_WEIGHT) / sum(team_total_xg_contribution).

    We compute team totals as Σ goals * GOALS_WEIGHT over all known players
    of that team in the given league/season. The weight cancels in the ratio,
    but we keep it explicit to match the spec.
    """
    db.execute(text("""
        WITH team_totals AS (
            SELECT team_id,
                   SUM(goals) * :w AS total_xg_proxy
            FROM player_stats
            WHERE league_id = :league_id AND season = :season AND team_id IS NOT NULL
            GROUP BY team_id
        )
        UPDATE player_stats ps
        SET xg_share = CASE
            WHEN tt.total_xg_proxy IS NULL OR tt.total_xg_proxy = 0 THEN 0
            ELSE (ps.goals * :w) / tt.total_xg_proxy
        END
        FROM team_totals tt
        WHERE ps.team_id = tt.team_id
          AND ps.league_id = :league_id
          AND ps.season    = :season
    """), {"league_id": league_db_id, "season": season, "w": GOALS_WEIGHT})


def _process_league(
    db: Session,
    client: APIFootballClient,
    league_db_id: int,
    league_api_id: int,
    season: int,
) -> int:
    try:
        scorers = client.get_top_scorers(league_api_id, season)
    except APIFootballError as e:
        logger.warning(
            f"top scorers fetch failed for league api_id={league_api_id} "
            f"season={season}: {e}"
        )
        return 0

    written = 0
    for entry in scorers:
        player = entry.get("player") or {}
        stats_list = entry.get("statistics") or []
        if not stats_list:
            continue
        st = stats_list[0]
        team_block = st.get("team") or {}
        goals_block = st.get("goals") or {}
        games_block = st.get("games") or {}

        player_id = player.get("id")
        if player_id is None:
            continue

        team_api_id = team_block.get("id")
        team_db = (
            db.query(Team).filter_by(api_id=team_api_id).first()
            if team_api_id is not None
            else None
        )

        goals = int(goals_block.get("total") or 0)
        assists = int(goals_block.get("assists") or 0)
        minutes = int(games_block.get("minutes") or 0)

        _upsert_player(
            db,
            player_id=int(player_id),
            team_db_id=team_db.id if team_db else None,
            name=str(player.get("name") or "Unknown"),
            league_db_id=league_db_id,
            season=season,
            goals=goals,
            assists=assists,
            minutes=minutes,
        )
        written += 1

    db.commit()
    _recompute_xg_shares(db, league_db_id, season)
    db.commit()
    return written


def run_collect_player_stats() -> None:
    logger.info("Starting weekly player stats collection")
    db = SessionLocal()
    try:
        seasons = _current_seasons(db)
        logger.info(f"Refreshing player stats for {len(seasons)} (league, season) pairs")

        with APIFootballClient() as client:
            total = 0
            for league_db_id, league_api_id, season in seasons:
                rows = _process_league(db, client, league_db_id, league_api_id, season)
                logger.info(
                    f"  league_id={league_db_id} api_id={league_api_id} "
                    f"season={season} → {rows} players upserted"
                )
                total += rows
                time.sleep(DELAY)

        logger.info(f"Weekly player stats collection complete — {total} rows written")
    finally:
        db.close()


if __name__ == "__main__":
    run_collect_player_stats()
