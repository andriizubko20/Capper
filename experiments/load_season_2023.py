"""
experiments/load_season_2023.py

Завантаження сезону 2023-24 (season=2023) для 6 основних ліг.
Завантажує: матчі, результати, odds (Bet365), xG/glicko.
Match stats (shots/possession) недоступні для цього сезону в API.

Запуск: python -m experiments.load_season_2023
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from loguru import logger

from data.api_client import SStatsClient
from data.collectors.matches import fetch_fixtures, fetch_fixture_glicko
from data.collectors.odds import fetch_odds
from db.models import League, Team, Match, MatchStats, Odds
from db.session import SessionLocal

SEASON = 2023
DELAY  = 0.5   # секунд між запитами

LEAGUES = {
    39:  ("Premier League",  "England"),
    140: ("La Liga",         "Spain"),
    78:  ("Bundesliga",      "Germany"),
    135: ("Serie A",         "Italy"),
    61:  ("Ligue 1",         "France"),
    94:  ("Primeira Liga",   "Portugal"),
}


def _upsert_league(db, api_id, name, country, season):
    obj = db.query(League).filter_by(api_id=api_id, season=season).first()
    if not obj:
        obj = League(api_id=api_id, name=name, country=country, season=season)
        db.add(obj)
        db.flush()
    return obj


def _upsert_team(db, api_id, name, country, league_id):
    obj = db.query(Team).filter_by(api_id=api_id).first()
    if not obj:
        obj = Team(api_id=api_id, name=name, country=country, league_id=league_id)
        db.add(obj)
        db.flush()
    return obj


def _upsert_match(db, fixture, league_db_id, home_id, away_id):
    obj = db.query(Match).filter_by(api_id=fixture["api_id"]).first()
    if obj:
        if fixture.get("status") == "Finished" and obj.home_score is None:
            obj.status    = fixture["status"]
            obj.home_score = fixture.get("home_score")
            obj.away_score = fixture.get("away_score")
        return obj, False
    try:
        dt = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
    except Exception:
        return None, False
    obj = Match(
        api_id=fixture["api_id"], league_id=league_db_id,
        home_team_id=home_id, away_team_id=away_id,
        date=dt, status=fixture.get("status", "Not Started"),
        home_score=fixture.get("home_score"), away_score=fixture.get("away_score"),
    )
    db.add(obj)
    db.flush()
    return obj, True


def _save_odds(db, match_id, odds_list):
    if not odds_list:
        return 0
    existing = db.query(Odds).filter_by(match_id=match_id).count()
    if existing:
        return 0
    count = 0
    for o in odds_list:
        db.add(Odds(
            match_id=match_id,
            bookmaker=o.get("bookmaker", ""),
            market=o.get("market", ""),
            outcome=o.get("outcome", ""),
            value=o.get("odds"),
            opening_value=o.get("opening_odds"),
            is_closing=o.get("is_closing", False),
        ))
        count += 1
    return count


def _save_stats(db, match_id, glicko):
    if not any(v is not None for v in glicko.values()):
        return False
    existing = db.query(MatchStats).filter_by(match_id=match_id).first()
    if existing:
        return False
    db.add(MatchStats(match_id=match_id, **glicko))
    return True


def run():
    db = SessionLocal()
    total_matches = total_odds = total_stats = 0

    try:
        with SStatsClient() as client:
            for api_id, (name, country) in LEAGUES.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"  {name} — сезон {SEASON}")
                logger.info(f"{'='*60}")

                # Перевіряємо чи вже завантажено
                existing_league = db.query(League).filter_by(api_id=api_id, season=SEASON).first()
                if existing_league:
                    existing_count = db.query(Match).filter_by(league_id=existing_league.id).count()
                    if existing_count >= 370:
                        logger.info(f"  Вже є {existing_count} матчів, пропускаємо")
                        continue

                league_obj = _upsert_league(db, api_id, name, country, SEASON)

                # Завантажуємо матчі
                fixtures = fetch_fixtures(api_id, SEASON, client)
                time.sleep(DELAY)
                finished = [f for f in fixtures if f.get("status") == "Finished"]
                logger.info(f"  Матчів: {len(fixtures)} | Finished: {len(finished)}")

                league_matches = league_odds = league_stats = 0

                for i, f in enumerate(finished):
                    # Команди
                    home = _upsert_team(db, f["home_team_api_id"], f["home_team_name"],
                                        f["home_team_country"], league_obj.id)
                    away = _upsert_team(db, f["away_team_api_id"], f["away_team_name"],
                                        f["away_team_country"], league_obj.id)

                    # Матч
                    match, is_new = _upsert_match(db, f, league_obj.id, home.id, away.id)
                    if match is None:
                        continue
                    if is_new:
                        league_matches += 1

                    # Odds
                    try:
                        odds_list = fetch_odds(f["api_id"], client)
                        time.sleep(DELAY)
                        saved = _save_odds(db, match.id, odds_list)
                        league_odds += saved
                    except Exception as e:
                        logger.warning(f"  Odds error {f['api_id']}: {e}")

                    # xG / glicko
                    try:
                        glicko = fetch_fixture_glicko(f["api_id"], client)
                        time.sleep(DELAY)
                        if _save_stats(db, match.id, glicko):
                            league_stats += 1
                    except Exception as e:
                        logger.warning(f"  Stats error {f['api_id']}: {e}")

                    # Прогрес кожні 50 матчів
                    if (i + 1) % 50 == 0:
                        db.commit()
                        logger.info(f"  [{i+1}/{len(finished)}] матчів | odds: {league_odds} | xG: {league_stats}")

                db.commit()
                logger.info(f"  Готово: {league_matches} нових матчів | {league_odds} odds | {league_stats} xG")
                total_matches += league_matches
                total_odds    += league_odds
                total_stats   += league_stats

    finally:
        db.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"  ПІДСУМОК сезону {SEASON}:")
    logger.info(f"  Матчів: {total_matches} | Odds: {total_odds} | xG: {total_stats}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run()
