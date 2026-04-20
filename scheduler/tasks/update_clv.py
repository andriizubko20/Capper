from datetime import date, timedelta

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
    yesterday = date.today() - timedelta(days=1)

    db = SessionLocal()
    try:
        matches = db.query(Match).filter(
            Match.date >= str(yesterday),
            Match.date < str(date.today()),
            Match.status == "FT",
        ).all()

        with SStatsClient() as client:
            for match in matches:
                closing = fetch_closing_odds(match.api_id, client)
                closing_map = {o["outcome"]: o["odds"] for o in closing if o["market"] == "1x2"}

                predictions = db.query(Prediction).filter_by(match_id=match.id).all()
                for pred in predictions:
                    closing_odd = closing_map.get(pred.outcome)
                    if not closing_odd or closing_odd <= 0:
                        continue

                    closing_implied = 1 / closing_odd
                    pred.clv = round(pred.probability / closing_implied - 1, 4)

                # Зберігаємо closing odds
                for o in closing:
                    db.add(Odds(
                        match_id=match.id,
                        market=o["market"],
                        bookmaker=o["bookmaker"],
                        outcome=o["outcome"],
                        value=o["odds"],
                        is_closing=True,
                    ))

        db.commit()
        logger.info(f"CLV updated for {len(matches)} matches")
    finally:
        db.close()
