import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from model.train import MODEL_DIR

# Синхронізовано з backtest.py
MIN_EV = 0.17
MIN_ODDS = 1.5
MAX_STAKE_PCT = 0.04
FRACTIONAL_KELLY = 0.25
MIN_SCENARIO_SCORE = 3


def load_model(version: str = "v1") -> tuple:
    model_path = MODEL_DIR / f"model_{version}.pkl"
    encoder_path = MODEL_DIR / f"encoder_{version}.pkl"
    features_path = MODEL_DIR / f"features_{version}.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    return model, encoder, features


def _scenario_score(features: dict, outcome: str) -> int:
    score = 0
    home_form = features.get("home_form_points", 0.5)
    away_form = features.get("away_form_points", 0.5)
    elo_diff = features.get("elo_diff", 0)
    home_xg_for = features.get("home_xg_for_avg_10", 1.3)
    away_xg_for = features.get("away_xg_for_avg_10", 1.3)
    home_xg_against = features.get("home_xg_against_avg_10", 1.3)
    away_xg_against = features.get("away_xg_against_avg_10", 1.3)
    market_home = features.get("market_home_prob", 0.33)
    market_away = features.get("market_away_prob", 0.33)
    home_rest = features.get("home_rest_days", 7)
    away_rest = features.get("away_rest_days", 7)
    home_injured = features.get("home_injured_count", 0)
    away_injured = features.get("away_injured_count", 0)

    if outcome == "home":
        if market_home > 0.50:             score += 1
        if home_form > 0.55:               score += 1
        if away_form < 0.45:               score += 1
        if elo_diff > 50:                  score += 1
        if home_xg_for > away_xg_against:  score += 1
        if home_rest >= 5:                 score += 1
        if away_injured > home_injured:    score += 1
        if elo_diff > 25:                  score += 1

    elif outcome == "away":
        if market_away > 0.38:             score += 1
        if away_form > 0.55:               score += 1
        if home_form < 0.45:               score += 1
        if elo_diff < -50:                 score += 1
        if away_xg_for > home_xg_against:  score += 1
        if away_rest >= 5:                 score += 1
        if home_injured > away_injured:    score += 1
        if elo_diff < -25:                 score += 1

    return score


def predict_match(
    features: dict,
    odds: dict,  # {"home": float, "draw": float, "away": float}
    bankroll: float,
    version: str = "v1",
) -> list[dict]:
    """
    Генерує pick для одного матчу.
    Повертає список ставок що пройшли scenario + EV фільтри.
    Максимум 1 ставка на матч (найвищий EV).
    """
    model, encoder, feature_cols = load_model(version)

    X = pd.DataFrame([features])[feature_cols].fillna(0)
    probs = model.predict_proba(X)[0]
    prob_map = dict(zip(encoder.classes_, probs))

    best_pick = None
    best_ev = -1

    for outcome in ("home", "away"):
        odd = odds.get(outcome, 0)
        if odd < MIN_ODDS:
            continue

        if _scenario_score(features, outcome) < MIN_SCENARIO_SCORE:
            continue

        our_prob = prob_map.get(outcome, 0)
        ev = our_prob * odd - 1
        if ev < MIN_EV:
            continue

        if ev > best_ev:
            best_ev = ev
            b = odd - 1
            q = 1 - our_prob
            kelly = max(0, (our_prob * b - q) / b) * FRACTIONAL_KELLY
            stake = round(min(bankroll * kelly, bankroll * MAX_STAKE_PCT), 2) if bankroll > 0 else 0

            best_pick = {
                "outcome": outcome,
                "probability": round(our_prob, 4),
                "odds": odd,
                "ev": round(ev, 4),
                "kelly_fraction": round(kelly, 4),
                "stake": stake,
            }

    if best_pick:
        logger.info(
            f"Pick: {best_pick['outcome']} | prob={best_pick['probability']:.3f} | "
            f"odds={best_pick['odds']} | EV={best_pick['ev']:.3f} | stake={best_pick['stake']}"
        )
        return [best_pick]

    return []
