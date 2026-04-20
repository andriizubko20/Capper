import pandas as pd
from loguru import logger

from db.models import Match, Odds, Team
from db.session import SessionLocal
from model.features.builder import build_dataset
from model.train import train


def run_retrain(version: str = "v1") -> None:
    logger.info("Starting model retraining")

    db = SessionLocal()
    try:
        matches = db.query(Match).filter(Match.status == "FT").all()
        if len(matches) < 200:
            logger.warning(f"Not enough data for retraining: {len(matches)} matches (need 200+)")
            return

        matches_df = pd.DataFrame([{
            "id": m.id,
            "date": pd.Timestamp(m.date),
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "league_id": m.league_id,
        } for m in matches])

        from db.models import MatchStats
        stats = db.query(MatchStats).join(Match).all()
        stats_df = pd.DataFrame([{
            "match_id": s.match_id,
            "date": pd.Timestamp(s.match.date),
            "home_team_id": s.match.home_team_id,
            "away_team_id": s.match.away_team_id,
            "home_score": s.match.home_score,
            "away_score": s.match.away_score,
            "home_xg": s.home_xg,
            "away_xg": s.away_xg,
            "home_shots_on_target":  s.home_shots_on_target,
            "away_shots_on_target":  s.away_shots_on_target,
            "home_shots_inside_box": s.home_shots_inside_box,
            "away_shots_inside_box": s.away_shots_inside_box,
            "home_possession":       s.home_possession,
            "away_possession":       s.away_possession,
            "home_corners":          s.home_corners,
            "away_corners":          s.away_corners,
            "home_gk_saves":         s.home_gk_saves,
            "away_gk_saves":         s.away_gk_saves,
            "home_passes_accurate":  s.home_passes_accurate,
            "away_passes_accurate":  s.away_passes_accurate,
        } for s in stats])

        odds = db.query(Odds).filter_by(market="1x2").all()
        odds_df = pd.DataFrame([{
            "match_id": o.match_id,
            "market": o.market,
            "bookmaker": o.bookmaker,
            "outcome": o.outcome,
            "value": o.value,
            "is_closing": o.is_closing,
        } for o in odds])

        teams = {t.id: {"elo": t.elo} for t in db.query(Team).all()}

        dataset = build_dataset(matches_df, stats_df, odds_df, teams)
        logger.info(f"Dataset built: {len(dataset)} samples")

        metrics = train(dataset, version=version)
        logger.info(f"Retraining complete: {metrics}")
    finally:
        db.close()
