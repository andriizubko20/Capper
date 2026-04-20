import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from model.train import MODEL_DIR
from model.weighted_score import compute_weighted_score, get_min_ev

MIN_ODDS = 1.5
MAX_STAKE_PCT = 0.04
FRACTIONAL_KELLY = 0.25


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


def predict_match(
    features: dict,
    odds: dict,  # {"home": float, "draw": float, "away": float}
    bankroll: float,
    version: str = "v1",
) -> list[dict]:
    """
    Генерує pick для одного матчу.
    Фільтри: weighted score (dyn_A) + EV.
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
        if not odd or odd < MIN_ODDS:
            continue

        ws = compute_weighted_score(features, outcome)
        min_ev = get_min_ev(ws)
        if min_ev is None:
            continue

        our_prob = prob_map.get(outcome, 0)
        ev = our_prob * odd - 1
        if ev < min_ev:
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
                "weighted_score": ws,
            }

    if best_pick:
        logger.info(
            f"Pick: {best_pick['outcome']} | prob={best_pick['probability']:.3f} | "
            f"odds={best_pick['odds']} | EV={best_pick['ev']:.3f} | "
            f"WS={best_pick['weighted_score']} | stake={best_pick['stake']}"
        )
        return [best_pick]

    return []
