"""
Backfill historical odds for finished matches without 1x2 odds.

Targets leagues where match_stats probabilities have weak coverage:
La Liga, Serie A, Ligue 1, Bundesliga, Premier League, Champions League.

Resumable: skips matches that already have 1x2 odds.

Run: python -m data.backfill_odds
"""
import time
from datetime import datetime
from loguru import logger

from data.api_client import SStatsClient, SStatsAPIError
from data.collectors.odds import fetch_odds
from db.models import League, Match, Odds
from db.session import SessionLocal

TARGET_LEAGUES = {
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Champions League",
}

DELAY = 1.2  # sec between API calls
LOG_EVERY = 50


def run_backfill_odds() -> None:
    db = SessionLocal()
    try:
        matches = (
            db.query(Match, League.name)
            .join(League, Match.league_id == League.id)
            .filter(
                Match.home_score.isnot(None),
                League.name.in_(TARGET_LEAGUES),
                ~Match.odds.any(market="1x2"),
            )
            .order_by(Match.date.asc())
            .all()
        )
        total = len(matches)
        logger.info(f"Found {total} finished matches without 1x2 odds in target leagues")
        if total == 0:
            return

        saved = 0
        skipped_empty = 0
        errors = 0
        started = time.time()

        with SStatsClient() as client:
            for idx, (match, league_name) in enumerate(matches, 1):
                try:
                    odds_list = fetch_odds(match.api_id, client)
                    if not odds_list:
                        skipped_empty += 1
                    else:
                        for o in odds_list:
                            db.add(Odds(
                                match_id=match.id,
                                market=o["market"],
                                bookmaker=o["bookmaker"],
                                outcome=o["outcome"],
                                value=o["odds"],
                                opening_value=o.get("opening_odds"),
                                is_closing=False,
                                recorded_at=datetime.utcnow(),
                            ))
                        saved += 1
                except SStatsAPIError as e:
                    errors += 1
                    logger.warning(f"API error match {match.api_id} [{league_name}]: {e}")
                except Exception as e:
                    errors += 1
                    logger.warning(f"Unexpected error match {match.api_id}: {e}")

                if idx % LOG_EVERY == 0:
                    db.commit()
                    elapsed = time.time() - started
                    rate = idx / elapsed if elapsed else 0
                    eta_min = (total - idx) / rate / 60 if rate else 0
                    logger.info(
                        f"Progress {idx}/{total} | saved={saved} empty={skipped_empty} "
                        f"err={errors} | {rate:.2f}/s | ETA {eta_min:.0f} min"
                    )

                time.sleep(DELAY)

        db.commit()
        logger.info(
            f"Backfill complete: {saved} saved, {skipped_empty} empty, {errors} errors "
            f"out of {total} matches"
        )
    finally:
        db.close()


if __name__ == "__main__":
    run_backfill_odds()
