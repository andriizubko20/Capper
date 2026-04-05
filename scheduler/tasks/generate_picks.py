from datetime import date, timedelta

import pandas as pd
from loguru import logger

from db.models import Match, Odds, Prediction, Team
from db.session import SessionLocal
from model.features.builder import build_match_features
from model.predict import predict_match


MODEL_VERSION = "v1-test"


def _load_teams_elo(db) -> dict:
    teams = db.query(Team).all()
    return {t.id: {"elo": t.elo} for t in teams}


def _load_matches_df(db) -> pd.DataFrame:
    matches = db.query(Match).filter(Match.status == "FT").all()
    return pd.DataFrame([{
        "id": m.id,
        "date": pd.Timestamp(m.date),
        "home_team_id": m.home_team_id,
        "away_team_id": m.away_team_id,
        "home_score": m.home_score,
        "away_score": m.away_score,
        "league_id": m.league_id,
    } for m in matches])


def _load_stats_df(db) -> pd.DataFrame:
    from db.models import MatchStats
    stats = db.query(MatchStats).join(Match).all()
    return pd.DataFrame([{
        "match_id": s.match_id,
        "date": pd.Timestamp(s.match.date),
        "home_team_id": s.match.home_team_id,
        "away_team_id": s.match.away_team_id,
        "home_xg": s.home_xg,
        "away_xg": s.away_xg,
    } for s in stats])


def _get_odds_for_match(match_id: int, db) -> dict | None:
    odds = db.query(Odds).filter_by(match_id=match_id, market="1x2", is_closing=False).all()
    if not odds:
        return None
    return {o.outcome: o.value for o in odds}


def run_generate_picks() -> None:
    logger.info("Starting picks generation")
    tomorrow = date.today() + timedelta(days=1)

    db = SessionLocal()
    try:
        matches = db.query(Match).filter(
            Match.date >= str(tomorrow),
            Match.date < str(tomorrow + timedelta(days=1)),
            Match.status == "NS",
        ).all()

        if not matches:
            logger.info("No matches tomorrow, skipping")
            return

        matches_df = _load_matches_df(db)
        stats_df = _load_stats_df(db)
        teams = _load_teams_elo(db)

        for match in matches:
            odds = _get_odds_for_match(match.id, db)
            if not odds:
                logger.warning(f"No odds for match {match.id}, skipping")
                continue

            features = build_match_features(
                match={
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "date": match.date,
                    "league_id": match.league_id,
                },
                matches_df=matches_df,
                stats_df=stats_df,
                teams=teams,
                odds=odds,
            )

            picks = predict_match(
                features=features,
                odds=odds,
                bankroll=0,  # bankroll per user — розраховується в боті
                version=MODEL_VERSION,
            )

            for pick in picks:
                db.add(Prediction(
                    match_id=match.id,
                    market="1x2",
                    outcome=pick["outcome"],
                    probability=pick["probability"],
                    odds_used=pick["odds"],
                    ev=pick["ev"],
                    kelly_fraction=pick["kelly_fraction"],
                    model_version=MODEL_VERSION,
                ))

        db.commit()
        logger.info(f"Picks generated for {len(matches)} matches")
    finally:
        db.close()
