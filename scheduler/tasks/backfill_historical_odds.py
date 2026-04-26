"""
scheduler/tasks/backfill_historical_odds.py

One-shot historical odds backfill for leagues that recently became accessible
again via the SStats proxy chain. Fetches per-fixture odds for already-finished
matches that have no `1x2` rows in the `odds` table yet.

Designed to run for hours and survive transient tunnel hiccups:
- Per-fixture errors are caught and counted, never abort the run.
- Progress is logged every PROGRESS_INTERVAL fixtures so you can tail the log
  while travelling.
- Sleep DELAY between fixtures to be polite to SStats + tunnel.

Usage on VPS (survives SSH disconnect):
    systemd-run --no-block --unit capper-backfill bash -c \\
      "cd /opt/capper && docker compose exec -T scheduler \\
       python -m scheduler.tasks.backfill_historical_odds \\
       > /var/log/capper-backfill.log 2>&1"

Watch progress:
    tail -f /var/log/capper-backfill.log
    systemctl status capper-backfill
"""
from __future__ import annotations

import time
from datetime import datetime

from loguru import logger
from sqlalchemy import tuple_
from sqlalchemy.exc import SQLAlchemyError

from data.api_client import SStatsAPIError, SStatsClient
from data.collectors.odds import fetch_odds
from db.models import League, Match, Odds
from db.session import SessionLocal

# Leagues whose historical odds we want to fill — same set we plan to add to
# Gem next. Format: (name, country) tuples.
TARGET_LEAGUES: list[tuple[str, str]] = [
    ("Premier League",  "Ukraine"),
    ("Eliteserien",     "Norway"),
    ("Premiership",     "Scotland"),
    ("Ekstraklasa",     "Poland"),
    ("Allsvenskan",     "Sweden"),
    ("Süper Lig",       "Turkey"),
    ("Serie B",         "Italy"),
]

DELAY = 1.5            # seconds between fixtures
PROGRESS_INTERVAL = 50 # log a status line every N processed fixtures


def _matches_needing_odds(db) -> list[Match]:
    """Finished matches in the 6 target leagues that have NO 1x2 odds yet."""
    league_ids = [
        lid for (lid,) in db.query(League.id).filter(
            tuple_(League.name, League.country).in_(TARGET_LEAGUES)
        ).all()
    ]
    if not league_ids:
        return []

    # All finished matches for these leagues
    finished = db.query(Match).filter(
        Match.league_id.in_(league_ids),
        Match.home_score.isnot(None),
        Match.away_score.isnot(None),
    ).all()

    # Match IDs that already have at least one 1x2 odds row
    already_have = {
        mid for (mid,) in db.query(Odds.match_id).filter(
            Odds.market == "1x2",
            Odds.match_id.in_([m.id for m in finished]),
        ).distinct().all()
    }

    return [m for m in finished if m.id not in already_have]


def run() -> None:
    started = datetime.utcnow()
    logger.info("Starting historical odds backfill")

    db = SessionLocal()
    try:
        targets = _matches_needing_odds(db)
        logger.info(
            f"Found {len(targets)} finished matches without 1x2 odds across "
            f"{len(TARGET_LEAGUES)} leagues"
        )
        if not targets:
            logger.info("Nothing to do — all 6 leagues already covered.")
            return

        # Group counts by league for the opening log line
        by_league: dict[int, int] = {}
        for m in targets:
            by_league[m.league_id] = by_league.get(m.league_id, 0) + 1
        for lid, n in by_league.items():
            league = db.query(League).get(lid)
            logger.info(f"  {league.country}/{league.name}: {n} matches to backfill")

        new_odds_total = 0
        skipped_no_data = 0
        errors = 0

        with SStatsClient() as client:
            for i, match in enumerate(targets, 1):
                try:
                    odds_list = fetch_odds(match.api_id, client)
                except SStatsAPIError as e:
                    errors += 1
                    logger.warning(f"  fixture {match.api_id}: SStats error {e}")
                    time.sleep(DELAY)
                    continue
                except Exception as e:
                    errors += 1
                    logger.warning(
                        f"  fixture {match.api_id}: {type(e).__name__}: {e}"
                    )
                    time.sleep(DELAY)
                    continue

                if not odds_list:
                    skipped_no_data += 1
                    time.sleep(DELAY)
                    continue

                added_for_match = 0
                for o in odds_list:
                    if o["market"] not in ("1x2", "double_chance"):
                        continue
                    exists = db.query(Odds).filter_by(
                        match_id=match.id,
                        market=o["market"],
                        bookmaker=o["bookmaker"],
                        outcome=o["outcome"],
                    ).first()
                    if exists:
                        continue
                    db.add(Odds(
                        match_id=match.id,
                        market=o["market"],
                        bookmaker=o["bookmaker"],
                        outcome=o["outcome"],
                        value=o["odds"],
                        opening_value=o.get("opening_odds"),
                        is_closing=False,
                    ))
                    added_for_match += 1

                if added_for_match:
                    try:
                        db.commit()
                        new_odds_total += added_for_match
                    except SQLAlchemyError as e:
                        db.rollback()
                        errors += 1
                        logger.warning(f"  fixture {match.api_id}: DB error {e}")

                if i % PROGRESS_INTERVAL == 0:
                    elapsed = (datetime.utcnow() - started).total_seconds()
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_min = (len(targets) - i) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"Progress {i}/{len(targets)} ({i*100/len(targets):.1f}%) "
                        f"· odds={new_odds_total} · skipped={skipped_no_data} "
                        f"· errors={errors} · ETA {eta_min:.0f} min"
                    )

                time.sleep(DELAY)

        elapsed = (datetime.utcnow() - started).total_seconds()
        logger.info(
            f"Backfill DONE in {elapsed/60:.1f} min · "
            f"fixtures={len(targets)} · new_odds={new_odds_total} · "
            f"skipped={skipped_no_data} · errors={errors}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    run()
