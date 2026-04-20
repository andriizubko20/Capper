"""
model/aquamarine/niches.py

Aquamarine model — niche definitions.
Reuses parse_niche / match_niche logic from Monster.
"""
from datetime import datetime

# Import shared parsing logic from Monster
from model.monster.niches import parse_niche, match_niche, HIGH_RISK_ODDS  # noqa: F401

OOS_START = datetime(2025, 8, 1)

MODELS = {
    'Premier League': [
        'home[1.7,2.0) xg>=1.8',
    ],
    'Bundesliga': [
        'home[2.2,2.8) xg>=1.2 form>=1.5',
        'home[1.8,2.2) elo>=75 form>=1.5 mkt>=0.5',
        'home[1.55,1.8) elo>=150 form>=1.5',
        'away[2.2,2.8) xg>=1.5 mkt>=0.4',
    ],
    'Serie A': [
        'away[1.8,2.2) xg>=1.5 form>=2.2',
        'home[2.0,2.5) form>=2.2 mkt>=0.45',
    ],
    'La Liga': [
        'home[1.8,2.2) xg>=1.0 elo>=30 mkt>=0.5',
        'home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5',
        'home[1.55,1.8) xg>=1.8',
        'home[1.8,2.2) xg>=1.2 form>=1.5',
    ],
    'Ligue 1': [
        'home[1.7,2.0) xg>=1.8',
        'home[1.7,2.0) xg>=1.5 elo>=30',
        'home[1.7,2.0) xg>=1.5 mkt>=0.5',
        'home[1.7,2.0) xg>=1.5 elo>=75',
        'home[1.55,1.8) xg>=1.2 elo>=75',
    ],
    'Primeira Liga': [
        'home[1.55,1.8) elo>=150',
        'away[2.2,2.8) xg>=1.2 mkt>=0.4',
    ],
    'Serie B': [
        'home[1.55,1.8) xg>=1.8',
        'home[1.55,1.8) xg>=1.5 elo>=75 form>=2.2',
        'home[1.55,1.8) xg>=1.2 elo>=150',
        'home[1.7,2.0) xg>=1.0 form>=2.2',
        'home[1.55,1.8) xg>=1.5',
        'home[2.0,2.5) elo>=75 form>=2.2',
    ],
    'Eredivisie': [
        'away[1.55,1.8) elo<=-150',
    ],
    'Jupiler Pro League': [
        'home[1.55,1.8) xg>=1.8',
        'home[1.3,1.55) xg>=1.5 elo>=150 form>=1.5',
        'away[2.0,2.5) xg>=1.5 form>=2.2',
        'home[1.55,1.8) form>=2.2',
        'home[1.8,2.2) xg>=1.2 mkt>=0.5',
        'home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5',
        'home[1.7,2.0) xg>=1.0 form>=1.5 mkt>=0.5',
    ],
    'Champions League': [
        'home[2.0,2.5) form>=1.5 mkt>=0.45',
        'home[1.7,2.0) xg>=1.5 mkt>=0.55',
        'home[1.7,2.0) xg>=1.5 elo>=75 mkt>=0.5',
        'home[1.55,1.8) elo>=30 form>=2.2',
        'home[1.55,1.8) elo>=150',
    ],
}
