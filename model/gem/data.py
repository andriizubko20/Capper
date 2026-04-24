"""
model/gem/data.py

Loads historical data for Gem model training and backtesting.
"""
import pandas as pd
from loguru import logger
from sqlalchemy import text

from db.session import SessionLocal
from model.gem.niches import TARGET_LEAGUES


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
        leagues_sql = tuple(TARGET_LEAGUES)

        matches = pd.DataFrame(
            db.execute(
                text(
                    """
                    SELECT m.id AS match_id, m.date, l.name AS league_name,
                           m.home_team_id, m.away_team_id,
                           m.home_score, m.away_score
                    FROM matches m JOIN leagues l ON l.id = m.league_id
                    WHERE m.status IN ('Finished','FT','finished','ft','Match Finished')
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                      AND l.name IN :leagues
                    ORDER BY m.date ASC
                    """
                ),
                {"leagues": leagues_sql},
            ).fetchall(),
            columns=[
                "match_id", "date", "league_name",
                "home_team_id", "away_team_id", "home_score", "away_score",
            ],
        )
        matches["date"] = pd.to_datetime(matches["date"])
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
                    JOIN leagues l ON l.id = m.league_id
                    WHERE m.home_score IS NOT NULL
                      AND l.name IN :leagues
                    """
                ),
                {"leagues": leagues_sql},
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
                    JOIN leagues l ON l.id = m.league_id
                    WHERE o.market='1x2' AND o.is_closing=false
                      AND l.name IN :leagues
                    GROUP BY o.match_id
                    """
                ),
                {"leagues": leagues_sql},
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
                    JOIN leagues l ON l.id = m.league_id
                    WHERE l.name IN :leagues
                    GROUP BY i.match_id, i.team_id
                    """
                ),
                {"leagues": leagues_sql},
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
