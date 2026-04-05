"""
Дозбирає пропущені дані для вже існуючих матчів:
- Injuries для всіх завершених матчів
- Glicko/xG для матчів без match_stats
Запуск: python -m data.backfill
"""
import time
from loguru import logger

from data.api_client import SStatsClient
from data.collectors.matches import fetch_fixture_glicko, fetch_injuries
from data.historical import clean
from db.models import Match, MatchStats, InjuryReport, Team
from db.session import SessionLocal

DELAY = 1.5
RETRY_DELAY = 30
MAX_RETRIES = 3


def backfill_glicko(db, client):
    """Збирає xG/Glicko для матчів без match_stats."""
    missing = db.query(Match).filter(
        Match.status == "Finished",
        ~Match.id.in_(db.query(MatchStats.match_id))
    ).all()

    logger.info(f"Backfilling glicko for {len(missing)} matches without stats...")

    for i, match in enumerate(missing):
        for attempt in range(MAX_RETRIES):
            try:
                glicko = fetch_fixture_glicko(match.api_id, client)
                time.sleep(DELAY)
                db.add(MatchStats(
                    match_id=match.id,
                    home_xg=glicko.get("home_xg"),
                    away_xg=glicko.get("away_xg"),
                    home_glicko=glicko.get("home_glicko"),
                    away_glicko=glicko.get("away_glicko"),
                    home_win_prob=glicko.get("home_win_prob"),
                    draw_prob=glicko.get("draw_prob"),
                    away_win_prob=glicko.get("away_win_prob"),
                ))
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES - 1:
                    logger.warning(f"Rate limit, waiting {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.warning(f"Glicko failed for {match.api_id}: {e}")
                    break

        if (i + 1) % 50 == 0:
            db.commit()
            logger.info(f"  Glicko progress: {i+1}/{len(missing)}")

    db.commit()
    logger.info("Glicko backfill done.")


def backfill_injuries(db, client):
    """Збирає травми для всіх завершених матчів без injury_reports."""
    matches_with_injuries = db.query(InjuryReport.match_id).distinct().subquery()
    missing = db.query(Match).filter(
        Match.status == "Finished",
        ~Match.id.in_(matches_with_injuries)
    ).all()

    logger.info(f"Backfilling injuries for {len(missing)} matches...")

    for i, match in enumerate(missing):
        try:
            injuries = fetch_injuries(match.api_id, client)
            time.sleep(DELAY)
            for inj in injuries:
                team = db.query(Team).filter_by(api_id=inj["team_id"]).first()
                if not team:
                    continue
                exists = db.query(InjuryReport).filter_by(
                    match_id=match.id, team_id=team.id, player_api_id=inj["player_id"]
                ).first()
                if not exists:
                    db.add(InjuryReport(
                        match_id=match.id,
                        team_id=team.id,
                        player_api_id=inj["player_id"],
                        player_name=clean(inj["player_name"]),
                        reason=clean(inj.get("reason", "")),
                    ))
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit, waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.warning(f"Injuries failed for {match.api_id}: {e}")

        if (i + 1) % 50 == 0:
            db.commit()
            logger.info(f"  Injuries progress: {i+1}/{len(missing)}")

    db.commit()
    logger.info("Injuries backfill done.")


def backfill_dc_odds(db, client):
    """Збирає Double Chance odds для матчів без DC odds."""
    from data.collectors.odds import fetch_odds
    from db.models import Odds

    matches_with_dc = db.query(Odds.match_id).filter(Odds.market == "double_chance").distinct().subquery()
    missing = db.query(Match).filter(
        Match.status == "Finished",
        ~Match.id.in_(matches_with_dc)
    ).all()

    logger.info(f"Backfilling DC odds for {len(missing)} matches...")

    for i, match in enumerate(missing):
        try:
            odds_list = fetch_odds(match.api_id, client)
            time.sleep(DELAY)
            for o in odds_list:
                if o["market"] != "double_chance":
                    continue
                exists = db.query(Odds).filter_by(
                    match_id=match.id,
                    market="double_chance",
                    bookmaker=o["bookmaker"],
                    outcome=o["outcome"],
                ).first()
                if not exists:
                    db.add(Odds(
                        match_id=match.id,
                        market="double_chance",
                        bookmaker=o["bookmaker"],
                        outcome=o["outcome"],
                        value=o["odds"],
                        opening_value=o.get("opening_odds"),
                        is_closing=False,
                    ))
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit, waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.warning(f"DC odds failed for {match.api_id}: {e}")

        if (i + 1) % 50 == 0:
            db.commit()
            logger.info(f"  DC odds progress: {i+1}/{len(missing)}")

    db.commit()
    logger.info("DC odds backfill done.")


def run():
    with SStatsClient() as client:
        db = SessionLocal()
        try:
            backfill_glicko(db, client)
            backfill_injuries(db, client)
            backfill_dc_odds(db, client)
        finally:
            db.close()


if __name__ == "__main__":
    run()
