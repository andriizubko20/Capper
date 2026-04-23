from datetime import date, datetime, timedelta, timezone

from loguru import logger

from data.api_client import SStatsClient
from data.collectors.odds import fetch_closing_odds
from db.models import Match, Odds, Prediction
from db.session import SessionLocal


def run_clv_update() -> None:
    """
    Збирає closing odds для вчорашніх матчів,
    оновлює CLV в таблиці predictions.
    """
    logger.info("Starting CLV update")
    today_utc = datetime.now(timezone.utc).date()
    yesterday = today_utc - timedelta(days=1)
    window_start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)
    window_end   = datetime(today_utc.year, today_utc.month, today_utc.day, tzinfo=timezone.utc)

    db = SessionLocal()
    try:
        matches = db.query(Match).filter(
            Match.date >= window_start,
            Match.date < window_end,
            Match.status.in_(("FT", "Finished")),
        ).all()

        with SStatsClient() as client:
            updated = 0
            for match in matches:
                try:
                    closing = fetch_closing_odds(match.api_id, client)
                    closing_map = {o["outcome"]: o["odds"] for o in closing if o["market"] == "1x2"}

                    predictions = db.query(Prediction).filter_by(match_id=match.id).all()
                    for pred in predictions:
                        closing_odd = closing_map.get(pred.outcome)
                        if not closing_odd or closing_odd <= 0:
                            continue
                        closing_implied = 1 / closing_odd
                        pred.clv = round(pred.probability / closing_implied - 1, 4)

                    for o in closing:
                        db.add(Odds(
                            match_id=match.id,
                            market=o["market"],
                            bookmaker=o["bookmaker"],
                            outcome=o["outcome"],
                            value=o["odds"],
                            is_closing=True,
                            recorded_at=datetime.utcnow(),
                        ))
                    updated += 1
                except Exception as e:
                    logger.warning(f"CLV update failed for match {match.api_id} [{type(e).__name__}]: {e}")

        db.commit()
        logger.info(f"CLV updated for {updated}/{len(matches)} matches")
    except Exception as e:
        db.rollback()
        logger.error(f"CLV update failed [{type(e).__name__}]: {e}")
    finally:
        db.close()
