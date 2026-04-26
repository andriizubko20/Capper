"""
scheduler/tasks/collect_lineups.py

Collect confirmed starting XI for upcoming matches (next 1.5h–2h window).

API-Football publishes confirmed lineups roughly 1 hour before kickoff.
We poll every 15 minutes within the pre-match window so a lineup is captured
as soon as it's announced.

Endpoint: GET /fixtures/lineups?fixture={fixture_id}
Reference: https://www.api-football.com/documentation-v3#tag/Fixtures/operation/get-fixtures-lineups

Response shape (relevant subset):
    {
      "response": [
        {
          "team":      {"id": 33, "name": "Manchester United"},
          "formation": "4-2-3-1",
          "startXI":   [{"player": {"id": 909, "name": "..."}}, ...]
        },
        { ...away team... }
      ]
    }
"""
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from data.api_client import SStatsClient, SStatsAPIError
from db.models import Lineup, Match
from db.session import SessionLocal

# Window: pull lineups for matches starting between +0h and +2h.
# Combined with cron */15min, every match gets multiple chances to be captured.
LOOKAHEAD_MIN_HOURS = 0.0
LOOKAHEAD_MAX_HOURS = 2.0


def _fetch_lineups(client: SStatsClient, fixture_id: int) -> list[dict]:
    """Hit /fixtures/lineups for one fixture. Returns raw `response` array."""
    data = client.get("/fixtures/lineups", params={"fixture": fixture_id})
    if isinstance(data, dict):
        return data.get("response", []) or []
    return []


def _parse_side(entry: dict) -> dict | None:
    """Extract (api_team_id, formation, [player_api_ids]) from one team block."""
    team = entry.get("team") or {}
    api_team_id = team.get("id")
    if api_team_id is None:
        return None
    formation = entry.get("formation")
    start_xi = entry.get("startXI") or []
    player_ids: list[int] = []
    for slot in start_xi:
        player = (slot or {}).get("player") or {}
        pid = player.get("id")
        if pid is not None:
            player_ids.append(int(pid))
    if not player_ids:
        return None
    return {
        "api_team_id": int(api_team_id),
        "formation": str(formation) if formation else None,
        "player_ids": player_ids,
    }


def _upsert_lineup(
    db,
    match_id: int,
    team_id: int,
    side: str,
    formation: str | None,
    player_ids: list[int],
) -> bool:
    """UPSERT one lineup row. Returns True if inserted (new), False if existed."""
    stmt = pg_insert(Lineup).values(
        match_id=match_id,
        team_id=team_id,
        side=side,
        formation=formation,
        starter_player_ids=player_ids,
        fetched_at=datetime.utcnow(),
    ).on_conflict_do_update(
        index_elements=["match_id", "side"],
        set_={
            "team_id": team_id,
            "formation": formation,
            "starter_player_ids": player_ids,
            "fetched_at": datetime.utcnow(),
        },
    )
    result = db.execute(stmt)
    # rowcount=1 for both insert and update under ON CONFLICT DO UPDATE;
    # cheaper to log the action uniformly.
    return result.rowcount > 0


def run_collect_lineups() -> None:
    now = datetime.now(timezone.utc)
    win_start = now + timedelta(hours=LOOKAHEAD_MIN_HOURS)
    win_end = now + timedelta(hours=LOOKAHEAD_MAX_HOURS)

    logger.info(
        f"[Lineups] Collecting | window: "
        f"{win_start.strftime('%d.%m %H:%M')} – {win_end.strftime('%d.%m %H:%M')} UTC"
    )

    db = SessionLocal()
    try:
        upcoming = db.query(Match).filter(
            Match.date >= win_start.replace(tzinfo=None),
            Match.date <= win_end.replace(tzinfo=None),
            Match.status == "Not Started",
        ).all()

        if not upcoming:
            logger.info("[Lineups] No upcoming matches in window")
            return

        # Skip matches that already have BOTH sides recorded.
        existing_pairs = {
            (mid, side)
            for mid, side in db.execute(text(
                "SELECT match_id, side FROM lineups WHERE match_id = ANY(:ids)"
            ), {"ids": [m.id for m in upcoming]}).fetchall()
        }

        # Map api_team_id -> internal team_id for the relevant teams
        team_lookup: dict[int, int] = {}
        for m in upcoming:
            if m.home_team:
                team_lookup[m.home_team.api_id] = m.home_team_id
            if m.away_team:
                team_lookup[m.away_team.api_id] = m.away_team_id

        n_fetched = n_saved = n_skipped = 0
        with SStatsClient() as client:
            for match in upcoming:
                if (match.id, "home") in existing_pairs and (match.id, "away") in existing_pairs:
                    n_skipped += 1
                    continue
                try:
                    raw = _fetch_lineups(client, match.api_id)
                except SStatsAPIError as e:
                    logger.warning(f"[Lineups] fetch failed match={match.id} api={match.api_id}: {e}")
                    continue
                n_fetched += 1
                if not raw:
                    continue

                # Map each entry to home/away by team_id match
                for entry in raw:
                    parsed = _parse_side(entry)
                    if parsed is None:
                        continue
                    api_tid = parsed["api_team_id"]
                    internal_tid = team_lookup.get(api_tid)
                    if internal_tid is None:
                        logger.debug(f"[Lineups] unknown team api_id={api_tid} for match {match.id}")
                        continue
                    if internal_tid == match.home_team_id:
                        side = "home"
                    elif internal_tid == match.away_team_id:
                        side = "away"
                    else:
                        continue
                    if (match.id, side) in existing_pairs:
                        continue
                    _upsert_lineup(
                        db,
                        match_id=match.id,
                        team_id=internal_tid,
                        side=side,
                        formation=parsed["formation"],
                        player_ids=parsed["player_ids"],
                    )
                    n_saved += 1
                db.commit()

        logger.info(
            f"[Lineups] Done | fetched={n_fetched} saved={n_saved} "
            f"already_complete={n_skipped} matches_in_window={len(upcoming)}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    run_collect_lineups()
