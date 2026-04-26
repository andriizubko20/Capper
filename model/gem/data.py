"""
model/gem/data.py

Loads historical data for Gem model training and backtesting.
"""
import pandas as pd
from loguru import logger
from sqlalchemy import text, tuple_

from db.models import League as LeagueModel
from db.session import SessionLocal
from model.gem.niches import TARGET_LEAGUES, to_canonical


def load_historical() -> dict[str, pd.DataFrame]:
    """
    Loads finished matches in target leagues with all features we need.

    Returns dict with keys:
      matches    — id, date, league, home/away team_id, scores, result
      stats      — xG, possession, shots, passes, glicko (pre-match), win_probs
      odds       — 1x2 market (home/draw/away) from best bookmaker
      injuries   — flattened per-match injury list (team_id granularity)
    """
    db = SessionLocal()
    try:
        # Resolve TARGET_LEAGUES (set of (name, country) tuples) → list of league.id.
        # Tuples-IN at SQL level is awkward to portably express; doing the lookup in
        # SQLAlchemy and feeding a tuple of integers to the raw-SQL queries keeps
        # the rest of the loaders unchanged.
        league_rows = (
            db.query(LeagueModel.id, LeagueModel.name, LeagueModel.country)
            .filter(tuple_(LeagueModel.name, LeagueModel.country).in_(list(TARGET_LEAGUES)))
            .all()
        )
        if not league_rows:
            logger.warning("Loaded zero target leagues from DB — check TARGET_LEAGUES")
        league_ids = tuple(r.id for r in league_rows) or (-1,)
        league_canonical_by_id = {r.id: to_canonical(r.name, r.country) for r in league_rows}

        matches = pd.DataFrame(
            db.execute(
                text(
                    """
                    SELECT m.id AS match_id, m.date, m.league_id,
                           m.home_team_id, m.away_team_id,
                           m.home_score, m.away_score
                    FROM matches m
                    WHERE m.status IN ('Finished','FT','finished','ft','Match Finished')
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                      AND m.league_id IN :league_ids
                    ORDER BY m.date ASC
                    """
                ),
                {"league_ids": league_ids},
            ).fetchall(),
            columns=[
                "match_id", "date", "league_id",
                "home_team_id", "away_team_id", "home_score", "away_score",
            ],
        )
        matches["date"] = pd.to_datetime(matches["date"])
        matches["league_name"] = matches["league_id"].map(league_canonical_by_id)
        matches["result"] = matches.apply(
            lambda r: "H" if r.home_score > r.away_score
            else ("A" if r.away_score > r.home_score else "D"),
            axis=1,
        )

        stats = pd.DataFrame(
            db.execute(
                text(
                    """
                    SELECT s.match_id,
                           s.home_xg, s.away_xg,
                           s.home_possession, s.away_possession,
                           s.home_shots_on_target, s.away_shots_on_target,
                           s.home_passes_accurate, s.away_passes_accurate,
                           s.home_passes_total, s.away_passes_total,
                           s.home_glicko, s.away_glicko,
                           s.home_win_prob, s.away_win_prob
                    FROM match_stats s
                    JOIN matches m ON m.id = s.match_id
                    WHERE m.home_score IS NOT NULL
                      AND m.league_id IN :league_ids
                    """
                ),
                {"league_ids": league_ids},
            ).fetchall(),
            columns=[
                "match_id", "home_xg", "away_xg",
                "home_possession", "away_possession",
                "home_sot", "away_sot",
                "home_pass_acc", "away_pass_acc",
                "home_pass_total", "away_pass_total",
                "home_glicko", "away_glicko",
                "home_win_prob", "away_win_prob",
            ],
        )

        odds = pd.DataFrame(
            db.execute(
                text(
                    """
                    SELECT o.match_id,
                           MAX(CASE WHEN o.outcome='home' THEN o.value END) AS home_odds,
                           MAX(CASE WHEN o.outcome='draw' THEN o.value END) AS draw_odds,
                           MAX(CASE WHEN o.outcome='away' THEN o.value END) AS away_odds
                    FROM odds o
                    JOIN matches m ON m.id = o.match_id
                    WHERE o.market='1x2' AND o.is_closing=false
                      AND m.league_id IN :league_ids
                    GROUP BY o.match_id
                    """
                ),
                {"league_ids": league_ids},
            ).fetchall(),
            columns=["match_id", "home_odds", "draw_odds", "away_odds"],
        )

        injuries = pd.DataFrame(
            db.execute(
                text(
                    """
                    SELECT i.match_id, i.team_id, COUNT(*) AS cnt
                    FROM injury_reports i
                    JOIN matches m ON m.id = i.match_id
                    WHERE m.league_id IN :league_ids
                    GROUP BY i.match_id, i.team_id
                    """
                ),
                {"league_ids": league_ids},
            ).fetchall(),
            columns=["match_id", "team_id", "cnt"],
        )

    finally:
        db.close()

    logger.info(
        f"Loaded: {len(matches)} matches, {len(stats)} stat rows, "
        f"{len(odds)} odds rows, {len(injuries)} injury rows"
    )
    return {"matches": matches, "stats": stats, "odds": odds, "injuries": injuries}
