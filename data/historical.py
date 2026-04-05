"""
Збір історичних даних за 2 сезони (2024, 2025).
Запуск: python -m data.historical
"""
import time
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy.orm import Session

from data.api_client import SStatsClient, SStatsAPIError
from data.collectors.leagues import TRACKED_LEAGUES
from data.collectors.matches import fetch_fixtures, fetch_fixture_glicko, fetch_injuries
from data.collectors.odds import fetch_odds
from db.models import League, Team, Match, MatchStats, Odds, InjuryReport
from db.session import SessionLocal

SEASONS = [2021, 2022, 2023, 2024, 2025]
DELAY = 1.5  # секунд між запитами
RETRY_DELAY = 30  # секунд при 429
MAX_RETRIES = 3


def clean(value: str | None) -> str | None:
    """Видаляє NUL байти з рядків — PostgreSQL їх не приймає."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def parse_date(value: str | None):
    """Парсить ISO дату з API в datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def upsert_league(league_api_id: int, name: str, country: str, season: int, db: Session) -> League:
    league = db.query(League).filter_by(api_id=league_api_id, season=season).first()
    if not league:
        league = League(api_id=league_api_id, name=name, country=country, season=season)
        db.add(league)
        db.flush()
    return league


def upsert_team(api_id: int, name: str, country: str, league_id: int, db: Session) -> Team:
    team = db.query(Team).filter_by(api_id=api_id).first()
    if not team:
        team = Team(api_id=api_id, name=name, country=country, league_id=league_id)
        db.add(team)
        db.flush()
    return team


def collect_season(league_api_id: int, season: int, client: SStatsClient, db: Session) -> int:
    league_name, country = TRACKED_LEAGUES[league_api_id]
    league = upsert_league(league_api_id, league_name, country, season, db)

    # Матчі
    fixtures = fetch_fixtures(league_api_id, season, client)
    time.sleep(DELAY)

    new_matches = 0
    for f in fixtures:
        if db.query(Match).filter_by(api_id=f["api_id"]).first():
            continue

        # Upsert teams from fixture data (more reliable than /Teams/list)
        if not f.get("home_team_api_id") or not f.get("away_team_api_id"):
            continue
        home = upsert_team(f["home_team_api_id"], clean(f["home_team_name"]), clean(f["home_team_country"]), league.id, db)
        away = upsert_team(f["away_team_api_id"], clean(f["away_team_name"]), clean(f["away_team_country"]), league.id, db)

        match = Match(
            api_id=f["api_id"],
            league_id=league.id,
            home_team_id=home.id,
            away_team_id=away.id,
            date=parse_date(f["date"]),
            status=clean(f["status"]),
            home_score=f["home_score"],
            away_score=f["away_score"],
        )
        db.add(match)
        try:
            db.flush()
        except Exception as e:
            db.rollback()
            logger.warning(f"Skipping match {f['api_id']}: {e}")
            continue

        # xG + Glicko тільки для завершених матчів
        if f["status"] == "Finished":
            for attempt in range(MAX_RETRIES):
                try:
                    glicko = fetch_fixture_glicko(f["api_id"], client)
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
                except SStatsAPIError as e:
                    if "429" in str(e) and attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limit hit, waiting {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.warning(f"Glicko fetch failed for {f['api_id']}: {e}")
                        break
                except Exception as e:
                    logger.warning(f"Glicko fetch failed for {f['api_id']}: {e}")
                    break

            for attempt in range(MAX_RETRIES):
                try:
                    odds_list = fetch_odds(f["api_id"], client)
                    time.sleep(DELAY)
                    for o in odds_list:
                        db.add(Odds(
                            match_id=match.id,
                            market=clean(o["market"]),
                            bookmaker=clean(o["bookmaker"]),
                            outcome=clean(o["outcome"]),
                            value=o["odds"],
                            opening_value=o.get("opening_odds"),
                            is_closing=False,
                        ))
                    break
                except SStatsAPIError as e:
                    if "429" in str(e) and attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limit hit, waiting {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.warning(f"Odds fetch failed for {f['api_id']}: {e}")
                        break
                except Exception as e:
                    logger.warning(f"Odds fetch failed for {f['api_id']}: {e}")
                    break

            # Травми
            try:
                injuries = fetch_injuries(f["api_id"], client)
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
                logger.warning(f"Injuries fetch failed for {f['api_id']}: {e}")

        new_matches += 1
        if new_matches % 20 == 0:
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"Commit failed: {e}")
            logger.info(f"  Progress: {new_matches}/{len(fixtures)} matches")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Final commit failed: {e}")
    return new_matches


def run():
    logger.info(f"Starting historical data collection: seasons {SEASONS}")
    total = 0

    with SStatsClient() as client:
        db = SessionLocal()
        try:
            for season in SEASONS:
                for league_id, (name, country) in TRACKED_LEAGUES.items():
                    logger.info(f"Collecting {name} {season}/{season+1}...")
                    count = collect_season(league_id, season, client, db)
                    total += count
                    logger.info(f"  Done: {count} new matches")
        finally:
            db.close()

    logger.info(f"Historical collection complete. Total new matches: {total}")


if __name__ == "__main__":
    run()
