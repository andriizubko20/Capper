import time
from datetime import date, datetime, timedelta, timezone

from loguru import logger

from data.api_client import SStatsClient
from data.collectors.odds import fetch_odds
from db.models import Match, Odds
from db.session import SessionLocal

DAYS_AHEAD = 7   # збираємо odds на наступні 7 днів
DELAY = 1.5


def collect_odds_for_upcoming(db, client) -> None:
    """Збирає та оновлює odds для майбутніх матчів на DAYS_AHEAD днів вперед."""
    today = datetime.now(timezone.utc).date()

    for delta in range(0, DAYS_AHEAD + 1):
        target = today + timedelta(days=delta)
        window_start = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
        window_end   = window_start + timedelta(days=1)
        matches = db.query(Match).filter(
            Match.date >= window_start,
            Match.date < window_end,
        ).all()

        new_odds = updated_odds = 0
        for match in matches:
            try:
                odds_list = fetch_odds(match.api_id, client)
                time.sleep(DELAY)
                for o in odds_list:
                    if o["market"] not in ("1x2", "double_chance"):
                        continue
                    # Bookmaker shopping: store one row per
                    # (match, market, bookmaker, outcome). Multiple bookmakers
                    # per match are kept so downstream picks can shop the BEST
                    # price across all of them.
                    exists = db.query(Odds).filter_by(
                        match_id=match.id,
                        market=o["market"],
                        bookmaker=o["bookmaker"],
                        outcome=o["outcome"],
                        is_closing=False,
                    ).first()
                    if exists:
                        # Refresh the current price; do not touch opening_value.
                        exists.value = o["odds"]
                        updated_odds += 1
                    else:
                        db.add(Odds(
                            match_id=match.id,
                            market=o["market"],
                            bookmaker=o["bookmaker"],
                            outcome=o["outcome"],
                            value=o["odds"],
                            opening_value=o.get("opening_odds"),
                            is_closing=False,
                        ))
                        new_odds += 1
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"Odds fetch failed for {match.api_id}: {e}")
        if matches:
            logger.info(f"  {target}: {len(matches)} matches, {new_odds} new / {updated_odds} updated odds")


def run_daily_collection() -> None:
    logger.info("Starting daily odds collection")
    with SStatsClient() as client:
        db = SessionLocal()
        try:
            collect_odds_for_upcoming(db, client)
        finally:
            db.close()
    logger.info("Daily odds collection complete")


if __name__ == "__main__":
    run_daily_collection()
