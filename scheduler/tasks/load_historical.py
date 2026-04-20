"""
scheduler/tasks/load_historical.py

Нічне завантаження історичних даних для нових ліг.
Завантажує матчі, xG/glicko, команди за поточний і попередній сезони.
"""
import time
from datetime import date

from loguru import logger

from data.api_client import SStatsClient
from data.collectors.leagues import TRACKED_LEAGUES, fetch_teams
from data.collectors.matches import fetch_fixtures, fetch_fixture_glicko
from db.models import League, Team, Match, MatchStats
from db.session import SessionLocal

DELAY = 1.0          # секунд між запитами
SEASONS = [2023, 2024, 2025, 2026]  # 3+ сезони для нових ліг

# Ліги що вже були з самого початку — для них пропускаємо повне завантаження
ORIGINAL_LEAGUES = {39, 140, 78, 135, 61, 2, 88, 144, 136, 94}


def _upsert_league(db, api_id: int, name: str, country: str, season: int) -> League:
    league = db.query(League).filter_by(api_id=api_id, season=season).first()
    if not league:
        league = League(api_id=api_id, name=name, country=country, season=season)
        db.add(league)
        db.flush()
    return league


def _upsert_team(db, api_id: int, name: str, country: str, league_id: int) -> Team:
    team = db.query(Team).filter_by(api_id=api_id).first()
    if not team:
        team = Team(api_id=api_id, name=name, country=country, league_id=league_id)
        db.add(team)
        db.flush()
    return team


def _upsert_match(db, fixture: dict, league_db_id: int, home_team_id: int, away_team_id: int) -> Match | None:
    existing = db.query(Match).filter_by(api_id=fixture["api_id"]).first()
    if existing:
        # Оновлюємо результат якщо завершено
        if fixture.get("status") == "Finished" and existing.home_score is None:
            existing.status = fixture["status"]
            existing.home_score = fixture.get("home_score")
            existing.away_score = fixture.get("away_score")
        return existing

    try:
        from datetime import datetime
        match_date = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
    except Exception:
        return None

    match = Match(
        api_id=fixture["api_id"],
        league_id=league_db_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        date=match_date,
        status=fixture.get("status", "Not Started"),
        home_score=fixture.get("home_score"),
        away_score=fixture.get("away_score"),
    )
    db.add(match)
    db.flush()
    return match


def run_load_historical() -> None:
    """Завантажує дані для нових ліг що ще не мають матчів в БД."""
    logger.info("Starting historical data load for new leagues")
    db = SessionLocal()

    try:
        with SStatsClient() as client:
            for league_api_id, (league_name, country) in TRACKED_LEAGUES.items():
                # Пропускаємо оригінальні ліги — вони вже завантажені
                if league_api_id in ORIGINAL_LEAGUES:
                    continue

                # Перевіряємо чи вже є дані
                existing_league = db.query(League).filter_by(api_id=league_api_id).first()
                existing_matches = 0
                if existing_league:
                    existing_matches = db.query(Match).filter_by(
                        league_id=existing_league.id
                    ).count()

                if existing_matches > 100:
                    logger.info(f"  {league_name}: вже є {existing_matches} матчів, пропускаємо")
                    continue

                logger.info(f"Loading {league_name} ({country})...")

                for season in SEASONS:
                    logger.info(f"  Season {season}...")

                    # Ліга
                    league_obj = _upsert_league(db, league_api_id, league_name, country, season)

                    # Команди
                    try:
                        teams_data = fetch_teams(league_api_id, season, client)
                        time.sleep(DELAY)
                        for t in teams_data:
                            _upsert_team(db, t["api_id"], t["name"], t["country"], league_obj.id)
                        db.flush()
                        logger.info(f"    Команди: {len(teams_data)}")
                    except Exception as e:
                        logger.warning(f"    Помилка завантаження команд: {e}")

                    # Матчі
                    try:
                        fixtures = fetch_fixtures(
                            league_id=league_api_id,
                            season=season,
                            client=client,
                        )
                        time.sleep(DELAY)
                    except Exception as e:
                        logger.warning(f"    Помилка завантаження матчів: {e}")
                        continue

                    new_matches = 0
                    xg_loaded = 0

                    for fixture in fixtures:
                        home_team = db.query(Team).filter_by(
                            api_id=fixture["home_team_api_id"]
                        ).first()
                        away_team = db.query(Team).filter_by(
                            api_id=fixture["away_team_api_id"]
                        ).first()

                        if not home_team:
                            home_team = _upsert_team(
                                db, fixture["home_team_api_id"],
                                fixture["home_team_name"],
                                fixture["home_team_country"],
                                league_obj.id,
                            )
                        if not away_team:
                            away_team = _upsert_team(
                                db, fixture["away_team_api_id"],
                                fixture["away_team_name"],
                                fixture["away_team_country"],
                                league_obj.id,
                            )

                        match = _upsert_match(
                            db, fixture, league_obj.id,
                            home_team.id, away_team.id,
                        )
                        if not match:
                            continue
                        new_matches += 1

                        # xG тільки для завершених матчів без статистики
                        if fixture.get("status") == "Finished" and not db.query(MatchStats).filter_by(match_id=match.id).first():
                            try:
                                glicko = fetch_fixture_glicko(fixture["api_id"], client)
                                time.sleep(DELAY)
                                if any(v is not None for v in glicko.values()):
                                    db.add(MatchStats(match_id=match.id, **glicko))
                                    xg_loaded += 1
                            except Exception:
                                pass

                        # Комітимо кожні 100 матчів
                        if new_matches % 100 == 0:
                            db.commit()
                            logger.info(f"    Прогрес: {new_matches}/{len(fixtures)} матчів")

                    db.commit()
                    logger.info(
                        f"    Сезон {season}: {new_matches} матчів, {xg_loaded} з xG"
                    )

        logger.info("Historical data load complete")
    finally:
        db.close()
