"""
BETA/data/extract.py

Pulls all raw data from DB into pandas DataFrames.
No feature engineering here — just clean extraction.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from sqlalchemy import text
from db.session import SessionLocal


def extract_matches() -> pd.DataFrame:
    """
    All finished matches with scores.
    Columns: match_id, date, league_id, home_team_id, away_team_id,
             home_score, away_score, result (H/D/A)
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                m.id          AS match_id,
                m.date        AS date,
                m.league_id,
                m.home_team_id,
                m.away_team_id,
                m.home_score,
                m.away_score,
                l.name        AS league_name
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE m.status IN ('Finished','FT','finished','ft','Match Finished')
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.date ASC
        """)).fetchall()
    finally:
        db.close()

    df = pd.DataFrame(rows, columns=[
        'match_id','date','league_id','home_team_id','away_team_id',
        'home_score','away_score','league_name'
    ])
    df['date'] = pd.to_datetime(df['date'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['result'] = df.apply(
        lambda r: 'H' if r.home_score > r.away_score
                  else ('A' if r.away_score > r.home_score else 'D'),
        axis=1
    )
    return df


def extract_stats() -> pd.DataFrame:
    """
    match_stats joined with match dates.
    Excludes post-match probability columns (leakage risk).
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                s.match_id,
                m.date,
                m.home_team_id,
                m.away_team_id,
                s.home_xg, s.away_xg,
                s.home_shots, s.away_shots,
                s.home_shots_on_target, s.away_shots_on_target,
                s.home_shots_inside_box, s.away_shots_inside_box,
                s.home_possession, s.away_possession,
                s.home_corners, s.away_corners,
                s.home_passes_accurate, s.away_passes_accurate,
                s.home_passes_total, s.away_passes_total,
                s.home_gk_saves, s.away_gk_saves,
                s.home_glicko, s.away_glicko
            FROM match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE m.home_score IS NOT NULL
            ORDER BY m.date ASC
        """)).fetchall()
    finally:
        db.close()

    cols = [
        'match_id','date','home_team_id','away_team_id',
        'home_xg','away_xg',
        'home_shots','away_shots',
        'home_shots_on_target','away_shots_on_target',
        'home_shots_inside_box','away_shots_inside_box',
        'home_possession','away_possession',
        'home_corners','away_corners',
        'home_passes_accurate','away_passes_accurate',
        'home_passes_total','away_passes_total',
        'home_gk_saves','away_gk_saves',
        'home_glicko','away_glicko',
    ]
    df = pd.DataFrame(rows, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    return df


def extract_odds() -> pd.DataFrame:
    """
    1x2 pre-match odds (is_closing=False).
    Returns one row per match with home/draw/away odds.
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                o.match_id,
                MAX(CASE WHEN o.outcome = 'home' THEN o.value END) AS home_odds,
                MAX(CASE WHEN o.outcome = 'draw' THEN o.value END) AS draw_odds,
                MAX(CASE WHEN o.outcome = 'away' THEN o.value END) AS away_odds
            FROM odds o
            WHERE o.market = '1x2'
              AND o.is_closing = false
            GROUP BY o.match_id
        """)).fetchall()
    finally:
        db.close()

    df = pd.DataFrame(rows, columns=['match_id','home_odds','draw_odds','away_odds'])
    return df


def extract_injuries() -> pd.DataFrame:
    """
    Injury counts per match per team.
    Returns: match_id, team_id, injury_count
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT match_id, team_id, COUNT(*) AS injury_count
            FROM injury_reports
            GROUP BY match_id, team_id
        """)).fetchall()
    finally:
        db.close()

    return pd.DataFrame(rows, columns=['match_id','team_id','injury_count'])


def load_all():
    """
    Convenience: load everything and merge into a single DataFrame.
    Returns (matches_df, stats_df, odds_df, injuries_df)
    """
    print("Extracting matches...")
    matches = extract_matches()
    print(f"  {len(matches)} finished matches")

    print("Extracting stats...")
    stats = extract_stats()
    print(f"  {len(stats)} stat rows")

    print("Extracting odds...")
    odds = extract_odds()
    print(f"  {len(odds)} matches with odds")

    print("Extracting injuries...")
    injuries = extract_injuries()
    print(f"  {len(injuries)} team-match injury rows")

    return matches, stats, odds, injuries


if __name__ == "__main__":
    m, s, o, i = load_all()
    print("\nMatches columns:", list(m.columns))
    print("Stats coverage:", len(s), "/", len(m))
    print("Odds coverage:", len(o), "/", len(m))
    print(m.head(3))
