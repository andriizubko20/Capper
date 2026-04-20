def implied_probability(odds: float) -> float:
    """Implied probability з десяткового коефіцієнта."""
    if odds <= 0:
        return 0.0
    return 1 / odds


def remove_overround(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    """
    Прибирає маржу букмекера, нормалізує implied probabilities до 1.
    Повертає (home_prob, draw_prob, away_prob).
    """
    raw = [implied_probability(o) for o in (home_odds, draw_odds, away_odds)]
    total = sum(raw)
    if total == 0:
        return 1/3, 1/3, 1/3
    return raw[0] / total, raw[1] / total, raw[2] / total


def odds_movement_features(opening_odds: float, current_odds: float) -> dict:
    """
    Рух лінії між відкриттям і поточним значенням.
    Різке падіння коефа = sharp money на цей вихід.
    """
    if opening_odds <= 0 or current_odds <= 0:
        return {"odds_movement": 0.0, "odds_movement_pct": 0.0, "sharp_signal": False}

    movement = current_odds - opening_odds
    movement_pct = movement / opening_odds

    return {
        "odds_movement": movement,
        "odds_movement_pct": movement_pct,
        "sharp_signal": movement_pct < -0.1,  # коеф впав більш ніж на 10%
    }


def market_implied_features(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """
    Ознаки на основі ринкових коефіцієнтів для 1x2.
    """
    home_prob, draw_prob, away_prob = remove_overround(home_odds, draw_odds, away_odds)
    overround = sum(implied_probability(o) for o in (home_odds, draw_odds, away_odds)) - 1

    return {
        "market_home_prob": home_prob,
        "market_draw_prob": draw_prob,
        "market_away_prob": away_prob,
        "market_overround": overround,
        "market_certainty": max(home_prob, away_prob),  # наскільки ринок впевнений в фавориті
        "market_home_edge": home_prob - away_prob,       # перевага хозяїна за ринком
    }
