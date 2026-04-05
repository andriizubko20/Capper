from datetime import date, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from data.api_client import APIFootballClient
from data.collectors.leagues import TRACKED_LEAGUES, fetch_teams
from data.collectors.matches import fetch_fixtures, fetch_fixture_stats
from data.collectors.odds import fetch_odds, fetch_closing_odds
from db.models import League, Team, Match, MatchStats, Odds
from db.session import SessionLocal


def upsert_teams(season: int, db: Session, client: APIFootballClient) -> None:
    for league_api_id, (league_name, country) in TRACKED_LEAGUES.items():
        league = db.query(League).filter_by(api_id=league_api_id).first()
        if not league:
            league = League(
                api_id=league_api_id,
                name=league_name,
                country=country,
                season=season,
            )
            db.add(league)
            db.flush()

        teams = fetch_teams(league_api_id, season, client)
        for t in teams:
            existing = db.query(Team).filter_by(api_id=t["api_id"]).first()
            if not existing:
                db.add(Team(
                    api_id=t["api_id"],
                    name=t["name"],
                    country=t["country"],
                    league_id=league.id,
                ))
    db.commit()
    logger.info("Teams upserted")


def collect_fixtures(season: int, db: Session, client: APIFootballClient) -> None:
    tomorrow = date.today() + timedelta(days=1)

    for league_api_id in TRACKED_LEAGUES:
        league = db.query(League).filter_by(api_id=league_api_id).first()
        if not league:
            continue

        fixtures = fetch_fixtures(
            league_id=league_api_id,
            season=season,
            client=client,
            from_date=tomorrow,
            to_date=tomorrow,
        )

        for f in fixtures:
            if db.query(Match).filter_by(api_id=f["api_id"]).first():
                continue

            home = db.query(Team).filter_by(api_id=f["home_team_api_id"]).first()
            away = db.query(Team).filter_by(api_id=f["away_team_api_id"]).first()
            if not home or not away:
                continue

            db.add(Match(
                api_id=f["api_id"],
                league_id=league.id,
                home_team_id=home.id,
                away_team_id=away.id,
                date=f["date"],
                status=f["status"] or "NS",
            ))

    db.commit()
    logger.info(f"Fixtures collected for {tomorrow}")


def collect_odds_for_tomorrow(db: Session, client: APIFootballClient) -> None:
    tomorrow = date.today() + timedelta(days=1)
    matches = db.query(Match).filter(
        Match.date >= str(tomorrow),
        Match.date < str(tomorrow + timedelta(days=1)),
        Match.status == "NS",
    ).all()

    for match in matches:
        odds = fetch_odds(match.api_id, client)
        for o in odds:
            db.add(Odds(
                match_id=match.id,
                market=o["market"],
                bookmaker=o["bookmaker"],
                outcome=o["outcome"],
                value=o["odds"],
                is_closing=False,
            ))

    db.commit()
    logger.info(f"Odds collected for {len(matches)} matches")


def collect_finished_stats(db: Session, client: APIFootballClient) -> None:
    yesterday = date.today() - timedelta(days=1)
    matches = db.query(Match).filter(
        Match.date >= str(yesterday),
        Match.date < str(date.today()),
        Match.status == "FT",
        ~Match.stats.has(),
    ).all()

    for match in matches:
        stats = fetch_fixture_stats(match.api_id, client)
        if stats:
            db.add(MatchStats(match_id=match.id, **stats))

    db.commit()
    logger.info(f"Stats collected for {len(matches)} finished matches")


def run_daily_collection(season: int = 2024) -> None:
    logger.info("Starting daily data collection")
    with APIFootballClient() as client:
        db = SessionLocal()
        try:
            collect_fixtures(season, db, client)
            collect_odds_for_tomorrow(db, client)
            collect_finished_stats(db, client)
        finally:
            db.close()
    logger.info("Daily data collection complete")
