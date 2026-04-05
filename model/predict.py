import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from model.train import MODEL_DIR
from model.features.odds_features import implied_probability

FRACTIONAL_KELLY = 0.25
MIN_EV = 0.05  # мінімальний EV для відправки пику


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
    Генерує пики для одного матчу.
    Повертає список ставок з позитивним EV.
    """
    model, encoder, feature_cols = load_model(version)

    X = pd.DataFrame([features])[feature_cols].fillna(0)
    probs = model.predict_proba(X)[0]
    classes = encoder.classes_  # ["away", "draw", "home"]

    prob_map = dict(zip(classes, probs))

    picks = []
    for outcome in ("home", "draw", "away"):
        our_prob = prob_map.get(outcome, 0)
        odd = odds.get(outcome, 0)
        if odd <= 1:
            continue

        ev = our_prob * odd - 1
        if ev < MIN_EV:
            continue

        # Fractional Kelly
        b = odd - 1
        q = 1 - our_prob
        kelly_full = (our_prob * b - q) / b
        kelly_fraction = max(0, kelly_full * FRACTIONAL_KELLY)
        stake = round(bankroll * kelly_fraction, 2)

        picks.append({
            "outcome": outcome,
            "probability": round(our_prob, 4),
            "odds": odd,
            "ev": round(ev, 4),
            "kelly_fraction": round(kelly_fraction, 4),
            "stake": stake,
        })
        logger.info(
            f"Pick: {outcome} | prob={our_prob:.3f} | odds={odd} | EV={ev:.3f} | stake={stake}"
        )

    return picks


def predict_day(
    matches: list[dict],
    bankroll: float,
    version: str = "v1",
) -> list[dict]:
    """
    Генерує пики для списку матчів на день.
    matches — список dict з ключами: match_id, features, odds
    """
    model, encoder, feature_cols = load_model(version)
    all_picks = []

    for match in matches:
        picks = predict_match(
            features=match["features"],
            odds=match["odds"],
            bankroll=bankroll,
            version=version,
        )
        for pick in picks:
            pick["match_id"] = match["match_id"]
            all_picks.append(pick)

    logger.info(f"Generated {len(all_picks)} picks for {len(matches)} matches")
    return all_picks
