"""
model/monster/niches.py

Monster model — niche definitions + matching logic.
Each niche: side[lo,hi) + optional xg/elo/form/mkt filters.
p_is = IS win rate (before OOS_START) used as Kelly probability.
"""
import re
from datetime import datetime

OOS_START = datetime(2025, 8, 1)

MODELS = {
    'Premier League': [
        'home[2.5,3.5) xg>=1.0',
        'home[2.2,2.8) xg>=1.2 form>=1.5',
        'home[1.8,2.2) form>=2.2',
        'home[1.7,2.0) xg>=1.8',
        'away[1.55,1.8) xg>=1.2 elo<=-75 form>=2.2',
    ],
    'Bundesliga': [
        'home[2.2,2.8) xg>=1.2 form>=1.5',
        'home[1.8,2.2) xg>=1.2 elo>=30 mkt>=0.5',
        'home[1.8,2.2) elo>=75 form>=1.5 mkt>=0.5',
        'home[1.55,1.8) elo>=150 form>=1.5',
        'away[2.5,3.5) xg>=1.0 elo<=-30 form>=1.5',
        'away[2.5,3.5) form>=1.5 mkt>=0.35',
        'away[2.5,3.5) elo<=-75 mkt>=0.35',
        'away[2.5,3.5) elo<=-30 form>=1.5 mkt>=0.35',
        'away[2.2,2.8) xg>=1.5 mkt>=0.4',
        'away[2.2,2.8) mkt>=0.4',
        'away[2.2,2.8) elo<=-75 mkt>=0.35',
        'away[2.0,2.5) mkt>=0.4',
        'away[1.7,2.0) xg>=1.0 form>=2.2',
    ],
    'Serie A': [
        'home[2.5,3.5) elo>=30 form>=1.5',
        'home[2.0,2.5) xg>=1.0 mkt>=0.45',
        'home[2.0,2.5) form>=2.2 mkt>=0.45',
        'home[1.8,2.2) xg>=1.5 form>=1.5',
        'home[1.3,1.55) xg>=1.5 elo>=150',
        'away[2.5,3.5) xg>=1.0 form>=1.8 mkt>=0.35',
        'away[2.2,2.8) xg>=1.2 form>=1.8 mkt>=0.35',
        'away[2.2,2.8) xg>=1.0 elo<=-150',
        'away[1.8,2.2) xg>=1.5 form>=2.2',
    ],
    'La Liga': [
        'home[2.2,2.8) xg>=1.0 form>=1.5',
        'home[2.0,2.5) xg>=1.0 form>=1.5',
        'home[1.8,2.2) xg>=1.2 form>=1.5',
        'home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5',
        'home[1.8,2.2) xg>=1.0 elo>=30 mkt>=0.5',
        'home[1.55,1.8) xg>=1.8',
        'away[2.5,3.5) xg>=1.8 form>=1.5',
        'away[2.5,3.5) elo<=-75 form>=2.2',
        'away[2.2,2.8) xg>=1.2 mkt>=0.4',
        'away[1.8,2.2) xg>=1.8 elo<=-75 mkt>=0.45',
    ],
    'Ligue 1': [
        'home[2.5,3.5) xg>=1.5',
        'home[2.2,2.8) xg>=1.2',
        'home[2.0,2.5) elo>=75',
        'home[1.7,2.0) xg>=1.8',
        'home[1.7,2.0) xg>=1.5 mkt>=0.5',
        'home[1.7,2.0) xg>=1.5 elo>=75',
        'home[1.7,2.0) xg>=1.5 elo>=30',
        'home[1.55,1.8) xg>=1.2 elo>=75',
        'away[2.5,3.5) xg>=1.0 form>=2.2',
        'away[2.2,2.8) form>=1.5 mkt>=0.4',
        'away[2.2,2.8) elo<=-30 form>=1.5 mkt>=0.4',
    ],
    'Primeira Liga': [
        'away[2.2,2.8) xg>=1.2 mkt>=0.4',
        'home[2.0,2.5) form>=1.5 mkt>=0.45',
        'home[1.55,1.8) elo>=150',
        'home[1.8,2.2) form>=2.2',
    ],
    'Serie B': [
        'home[2.2,2.8) form>=2.2',
        'home[2.0,2.5) elo>=75 form>=2.2',
        'home[1.7,2.0) xg>=1.0 form>=2.2',
        'home[1.55,1.8) xg>=1.8',
        'home[1.55,1.8) xg>=1.5 elo>=75 form>=2.2',
        'home[1.55,1.8) xg>=1.2 elo>=150',
        'home[1.55,1.8) xg>=1.5',
        'away[2.0,2.5) xg>=1.5 elo<=-30',
    ],
    'Eredivisie': [
        'home[2.5,3.5) xg>=1.0',
        'home[2.0,2.5) elo>=75',
        'away[2.5,3.5) elo<=-75 mkt>=0.35',
        'away[1.8,2.2) xg>=1.2 form>=2.2',
        'away[1.7,2.0) xg>=1.2 form>=2.2',
        'away[1.55,1.8) elo<=-150',
    ],
    'Jupiler Pro League': [
        'home[2.5,3.5) form>=2.2',
        'home[2.0,2.5) xg>=1.0 mkt>=0.45',
        'home[1.8,2.2) xg>=1.2 mkt>=0.5',
        'home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5',
        'home[1.7,2.0) xg>=1.0 form>=1.5 mkt>=0.5',
        'home[1.55,1.8) xg>=1.8',
        'home[1.55,1.8) form>=2.2',
        'home[1.3,1.55) xg>=1.5 elo>=150 form>=1.5',
        'away[2.2,2.8) xg>=1.2 form>=2.2 mkt>=0.35',
        'away[2.0,2.5) xg>=1.5 form>=2.2',
    ],
    'Champions League': [
        'home[2.2,2.8) xg>=1.8 form>=1.5',
        'home[2.0,2.5) form>=1.5 mkt>=0.45',
        'home[1.8,2.2) mkt>=0.5',
        'home[1.7,2.0) xg>=1.5 mkt>=0.55',
        'home[1.7,2.0) xg>=1.5 elo>=75 mkt>=0.5',
        'home[1.55,1.8) elo>=30 form>=2.2',
        'home[1.55,1.8) elo>=150',
        'home[1.3,1.55) xg>=1.8 form>=1.8',
        'away[2.2,2.8) xg>=1.8 mkt>=0.4',
        'away[2.2,2.8) xg>=1.5 form>=1.8 mkt>=0.4',
    ],
}

# Map league names as they appear in DB
LEAGUE_NAME_MAP = {
    'Premier League': 'Premier League',   # England
    'La Liga':        'La Liga',
    'Bundesliga':     'Bundesliga',
    'Serie A':        'Serie A',
    'Ligue 1':        'Ligue 1',
    'Primeira Liga':  'Primeira Liga',
    'Serie B':        'Serie B',
    'Eredivisie':     'Eredivisie',
    'Jupiler Pro League': 'Jupiler Pro League',
    'Champions League':   'Champions League',
}

# API-Football league IDs — used to avoid name collisions (e.g. Ukrainian PL also named "Premier League")
LEAGUE_API_IDS = {
    39,   # Premier League (England)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    140,  # La Liga (Spain)
    61,   # Ligue 1 (France)
    94,   # Primeira Liga (Portugal)
    136,  # Serie B (Italy)
    88,   # Eredivisie (Netherlands)
    144,  # Jupiler Pro League (Belgium)
    2,    # Champions League
}

HIGH_RISK_ODDS = 2.5  # odds >= this → HIGH RISK label


def parse_niche(s: str) -> dict:
    """Parse niche string into filter dict."""
    m = re.match(r'(home|away)\[(\d+\.?\d*),(\d+\.?\d*)\)', s)
    side = m.group(1)
    lo = float(m.group(2))
    hi = float(m.group(3))

    xg = elo = fm = mk = None
    mx = re.search(r'xg>=([\d.]+)', s);   xg = float(mx.group(1)) if mx else None
    mx = re.search(r'elo>=([-\d]+)', s);  elo_ge = int(mx.group(1)) if mx else None
    mx = re.search(r'elo<=([-\d]+)', s);  elo_le = int(mx.group(1)) if mx else None
    mx = re.search(r'form>=([\d.]+)', s); fm = float(mx.group(1)) if mx else None
    mx = re.search(r'mkt>=([\d.]+)', s);  mk = float(mx.group(1)) if mx else None

    return {
        'side': side, 'lo': lo, 'hi': hi,
        'xg': xg, 'elo_ge': elo_ge, 'elo_le': elo_le,
        'form': fm, 'mkt': mk,
    }


def match_niche(features: dict, niche: dict, league: str, match_league: str) -> bool:
    """
    Check if a match matches a niche.

    features keys:
      home_odds, away_odds,
      xg_ratio_home_5, xg_ratio_away_5,
      elo_diff,  (home_elo - away_elo)
      home_pts_5, away_pts_5,
      mkt_home_prob, mkt_away_prob
    """
    if match_league != league:
        return False

    side = niche['side']
    odds = features.get('home_odds') if side == 'home' else features.get('away_odds')
    if odds is None:
        return False
    if not (niche['lo'] <= odds < niche['hi']):
        return False

    xg = features.get(f'xg_ratio_{side}_5')
    if niche['xg'] is not None and (xg is None or xg < niche['xg']):
        return False

    elo_diff = features.get('elo_diff')  # home - away
    if niche['elo_ge'] is not None:
        val = elo_diff if side == 'home' else -elo_diff
        if elo_diff is None or val < niche['elo_ge']:
            return False
    if niche['elo_le'] is not None:
        val = elo_diff if side == 'home' else -elo_diff
        if elo_diff is None or val > niche['elo_le']:
            return False

    form = features.get(f'home_pts_5') if side == 'home' else features.get(f'away_pts_5')
    if niche['form'] is not None and (form is None or form < niche['form']):
        return False

    mkt = features.get('mkt_home_prob') if side == 'home' else features.get('mkt_away_prob')
    if niche['mkt'] is not None and (mkt is None or mkt < niche['mkt']):
        return False

    return True
