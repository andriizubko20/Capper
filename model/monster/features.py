"""
model/monster/features.py

Builds features for upcoming matches using historical data from DB.
No leakage: all features computed from matches BEFORE the target date.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from loguru import logger
from sqlalchemy import text

from db.session import SessionLocal

ELO_DEFAULT = 1500.0
ELO_K = 32


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def load_historical_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load finished matches, stats, odds from DB."""
    db = SessionLocal()
    try:
        matches = pd.DataFrame(db.execute(text("""
            SELECT m.id AS match_id, m.date, m.league_id, l.name AS league_name,
                   m.home_team_id, m.away_team_id, m.home_score, m.away_score
            FROM matches m JOIN leagues l ON l.id = m.league_id
            WHERE m.status IN ('Finished','FT','finished','ft','Match Finished')
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
            ORDER BY m.date ASC
        """)).fetchall(), columns=[
            'match_id','date','league_id','league_name',
            'home_team_id','away_team_id','home_score','away_score'
        ])

        stats = pd.DataFrame(db.execute(text("""
            SELECT s.match_id, s.home_xg, s.away_xg,
                   s.home_possession, s.away_possession
            FROM match_stats s
            JOIN matches m ON m.id = s.match_id
            WHERE m.home_score IS NOT NULL
        """)).fetchall(), columns=['match_id','home_xg','away_xg','home_possession','away_possession'])

        odds = pd.DataFrame(db.execute(text("""
            SELECT o.match_id,
                   MAX(CASE WHEN o.outcome='home' THEN o.value END) AS home_odds,
                   MAX(CASE WHEN o.outcome='draw' THEN o.value END) AS draw_odds,
                   MAX(CASE WHEN o.outcome='away' THEN o.value END) AS away_odds
            FROM odds o WHERE o.market='1x2' AND o.is_closing=false
            GROUP BY o.match_id
        """)).fetchall(), columns=['match_id','home_odds','draw_odds','away_odds'])

    finally:
        db.close()

    matches['date'] = pd.to_datetime(matches['date'])
    matches['result'] = matches.apply(
        lambda r: 'H' if r.home_score > r.away_score else ('A' if r.away_score > r.home_score else 'D'),
        axis=1
    )
    return matches, stats, odds


def build_team_state(matches: pd.DataFrame, stats: pd.DataFrame,
                     cutoff_date: datetime | None = None) -> dict[int, dict]:
    """
    Computes current state for each team: elo, rolling pts_5, xg_ratio_5.
    If cutoff_date given — uses only matches before that date.
    Returns {team_id: {elo, pts_5, xg_ratio_5}}.
    """
    m = matches.copy()
    if cutoff_date:
        m = m[m['date'] < pd.Timestamp(cutoff_date)]

    m = m.merge(stats[['match_id','home_xg','away_xg']], on='match_id', how='left')
    m = m.sort_values('date').reset_index(drop=True)

    elos: dict[int, float] = {}
    team_history: dict[int, list] = defaultdict(list)

    for _, row in m.iterrows():
        h, a = int(row.home_team_id), int(row.away_team_id)
        elo_h = elos.get(h, ELO_DEFAULT)
        elo_a = elos.get(a, ELO_DEFAULT)

        exp_h = _elo_expected(elo_h, elo_a)
        s_h = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        s_a = 1.0 - s_h

        elos[h] = elo_h + ELO_K * (s_h - exp_h)
        elos[a] = elo_a + ELO_K * (s_a - (1 - exp_h))

        pts_h = 3 if row.result == 'H' else (1 if row.result == 'D' else 0)
        pts_a = 3 if row.result == 'A' else (1 if row.result == 'D' else 0)

        team_history[h].append({
            'pts': pts_h,
            'xgf': row.get('home_xg') or np.nan,
            'xga': row.get('away_xg') or np.nan,
        })
        team_history[a].append({
            'pts': pts_a,
            'xgf': row.get('away_xg') or np.nan,
            'xga': row.get('home_xg') or np.nan,
        })

    result = {}
    for team_id, history in team_history.items():
        last5 = history[-5:]
        pts_5 = np.mean([h['pts'] for h in last5]) if last5 else np.nan
        xgf_5 = np.nanmean([h['xgf'] for h in last5]) if last5 else np.nan
        xga_5 = np.nanmean([h['xga'] for h in last5]) if last5 else np.nan
        xg_ratio_5 = xgf_5 / max(xga_5, 0.1) if not np.isnan(xgf_5) else np.nan
        result[team_id] = {
            'elo': elos.get(team_id, ELO_DEFAULT),
            'pts_5': pts_5,
            'xg_ratio_5': xg_ratio_5,
        }

    return result


def build_upcoming_features(match: dict, team_state: dict[int, dict],
                             odds: dict) -> dict:
    """
    Build features for a single upcoming match.

    match: {home_team_id, away_team_id, league_name}
    team_state: output of build_team_state()
    odds: {home: float, draw: float, away: float}
    """
    h_id = match['home_team_id']
    a_id = match['away_team_id']

    h = team_state.get(h_id, {})
    a = team_state.get(a_id, {})

    elo_h = h.get('elo', ELO_DEFAULT)
    elo_a = a.get('elo', ELO_DEFAULT)
    elo_diff = elo_h - elo_a

    h_odds = odds.get('home')
    a_odds = odds.get('away')
    d_odds = odds.get('draw')

    # Market implied probs (margin-normalized)
    mkt_home_prob = mkt_away_prob = None
    if h_odds and d_odds and a_odds:
        raw_h = 1.0 / h_odds
        raw_d = 1.0 / d_odds
        raw_a = 1.0 / a_odds
        margin = raw_h + raw_d + raw_a
        mkt_home_prob = raw_h / margin
        mkt_away_prob = raw_a / margin

    return {
        'home_odds':        h_odds,
        'away_odds':        a_odds,
        'draw_odds':        d_odds,
        'elo_diff':         elo_diff,
        'home_pts_5':       h.get('pts_5'),
        'away_pts_5':       a.get('pts_5'),
        'xg_ratio_home_5':  h.get('xg_ratio_5'),
        'xg_ratio_away_5':  a.get('xg_ratio_5'),
        'mkt_home_prob':    mkt_home_prob,
        'mkt_away_prob':    mkt_away_prob,
        'league_name':      match.get('league_name', ''),
    }


def compute_p_is(matches: pd.DataFrame, stats: pd.DataFrame,
                 odds_df: pd.DataFrame, league: str,
                 niche: dict, cutoff: datetime) -> float | None:
    """
    Compute IS win rate for a niche (matches before cutoff date).
    Returns float or None if < 3 IS samples.
    """
    from model.monster.niches import match_niche

    m = matches[matches['date'] < pd.Timestamp(cutoff)].copy()
    m = m.merge(stats[['match_id','home_xg','away_xg']], on='match_id', how='left')
    m = m.merge(odds_df, on='match_id', how='left')

    # Need team state up to cutoff for each match — approximate with rolling
    # For IS p estimation we just use historical features per match
    wins = total = 0

    # Pre-build team state incrementally would be expensive;
    # approximate: use rolling last-5 pts from match log
    team_pts: dict[int, list] = defaultdict(list)
    team_xgf: dict[int, list] = defaultdict(list)
    team_xga: dict[int, list] = defaultdict(list)
    elos: dict[int, float] = {}

    for _, row in m.sort_values('date').iterrows():
        h_id, a_id = int(row.home_team_id), int(row.away_team_id)
        elo_h = elos.get(h_id, ELO_DEFAULT)
        elo_a = elos.get(a_id, ELO_DEFAULT)

        # Compute features BEFORE updating
        h_pts5 = np.mean(team_pts[h_id][-5:]) if team_pts[h_id] else np.nan
        a_pts5 = np.mean(team_pts[a_id][-5:]) if team_pts[a_id] else np.nan
        h_xgf5 = np.nanmean(team_xgf[h_id][-5:]) if team_xgf[h_id] else np.nan
        h_xga5 = np.nanmean(team_xga[h_id][-5:]) if team_xga[h_id] else np.nan
        a_xgf5 = np.nanmean(team_xgf[a_id][-5:]) if team_xgf[a_id] else np.nan
        a_xga5 = np.nanmean(team_xga[a_id][-5:]) if team_xga[a_id] else np.nan

        h_xg_ratio = h_xgf5 / max(h_xga5, 0.1) if not np.isnan(h_xgf5 or np.nan) else np.nan
        a_xg_ratio = a_xgf5 / max(a_xga5, 0.1) if not np.isnan(a_xgf5 or np.nan) else np.nan

        h_odds = row.get('home_odds')
        a_odds = row.get('away_odds')
        d_odds = row.get('draw_odds')
        mkt_h = mkt_a = None
        if h_odds and d_odds and a_odds and h_odds > 0 and d_odds > 0 and a_odds > 0:
            raw = 1/h_odds + 1/d_odds + 1/a_odds
            mkt_h = (1/h_odds) / raw
            mkt_a = (1/a_odds) / raw

        features = {
            'home_odds': h_odds, 'away_odds': a_odds,
            'elo_diff': elo_h - elo_a,
            'home_pts_5': h_pts5, 'away_pts_5': a_pts5,
            'xg_ratio_home_5': h_xg_ratio, 'xg_ratio_away_5': a_xg_ratio,
            'mkt_home_prob': mkt_h, 'mkt_away_prob': mkt_a,
        }

        if match_niche(features, niche, league, row.get('league_name', '')):
            side = niche['side']
            won = (row.result == 'H') if side == 'home' else (row.result == 'A')
            wins += int(won)
            total += 1

        # Update state
        exp_h = _elo_expected(elo_h, elo_a)
        s_h = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        elos[h_id] = elo_h + ELO_K * (s_h - exp_h)
        elos[a_id] = elo_a + ELO_K * ((1 - s_h) - (1 - exp_h))

        pts_h = 3 if row.result == 'H' else (1 if row.result == 'D' else 0)
        pts_a = 3 if row.result == 'A' else (1 if row.result == 'D' else 0)
        team_pts[h_id].append(pts_h); team_pts[a_id].append(pts_a)

        hxg = row.get('home_xg'); axg = row.get('away_xg')
        team_xgf[h_id].append(hxg if hxg else np.nan)
        team_xga[h_id].append(axg if axg else np.nan)
        team_xgf[a_id].append(axg if axg else np.nan)
        team_xga[a_id].append(hxg if hxg else np.nan)

    if total < 3:
        return None
    return wins / total
