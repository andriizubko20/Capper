"""
Утиліти для розрахунку результату та PnL по завершеним матчам.
Використовується у update_results.py та live_tracker.py.
"""
from __future__ import annotations


def calculate_result(market: str, outcome: str, home_score: int, away_score: int) -> str | None:
    """
    Повертає 'win' або 'loss' для prediction по завершеному матчу.
    Повертає None якщо ринок не розпізнано.
    """
    market = market.lower().strip()

    if market == '1x2':
        if home_score > away_score:
            actual = 'home'
        elif away_score > home_score:
            actual = 'away'
        else:
            actual = 'draw'
        return 'win' if outcome.lower() == actual else 'loss'

    if market == 'total':
        # outcome: 'Over 2.5' / 'Under 2.5'
        try:
            parts = outcome.strip().split()
            direction = parts[0].lower()   # 'over' | 'under'
            line = float(parts[1])
        except (IndexError, ValueError):
            return None
        total = home_score + away_score
        if direction == 'over':
            return 'win' if total > line else 'loss'
        elif direction == 'under':
            return 'win' if total < line else 'loss'
        return None

    if market == 'btts':
        both = home_score > 0 and away_score > 0
        if outcome.lower() == 'yes':
            return 'win' if both else 'loss'
        elif outcome.lower() == 'no':
            return 'win' if not both else 'loss'
        return None

    if market == 'double_chance':
        if home_score > away_score:
            match_res = 'home'
        elif away_score > home_score:
            match_res = 'away'
        else:
            match_res = 'draw'
        norm = outcome.lower().replace('/', '_')
        mapping = {
            'home_draw': ('home', 'draw'),
            'home_away': ('home', 'away'),
            'draw_away': ('draw', 'away'),
        }
        allowed = mapping.get(norm)
        if allowed is None:
            return None
        return 'win' if match_res in allowed else 'loss'

    if market == 'handicap':
        # outcome: 'Home -1.5' / 'Away +0.5' etc.
        try:
            parts = outcome.strip().split()
            team = parts[0].lower()    # 'home' | 'away'
            hcap = float(parts[1])
        except (IndexError, ValueError):
            return None
        team = team.lower()
        if team == 'home':
            return 'win' if (home_score + hcap) > away_score else 'loss'
        elif team == 'away':
            return 'win' if (away_score + hcap) > home_score else 'loss'
        return None

    return None


def calculate_pnl(result: str, stake: float, odds_used: float) -> float:
    """PnL в доларах: виграш = stake × (odds − 1), програш = −stake."""
    if result == 'win':
        return round(stake * (odds_used - 1), 2)
    return round(-stake, 2)
