import pandas as pd
from loguru import logger

from model.train import train, FEATURE_COLS, load_data_from_db
from model.evaluate import compute_roi, log_metrics
from model.features.builder import build_dataset

MIN_EV = 0.20
MIN_EV_DRAW = 0.25        # вищий поріг для нічиїх — найнепередбачуваніший вихід
MIN_ODDS = 1.5
MAX_STAKE_PCT = 0.03       # максимум 3% банкролу незалежно від Kelly
FRACTIONAL_KELLY = 0.25
INITIAL_BANKROLL = 1000.0


def backtest(
    dataset: pd.DataFrame,
    train_window: int = 500,
    step: int = 50,
    version: str = "backtest",
) -> dict:
    """
    Walk-forward бектест: навчаємо на train_window матчах,
    тестуємо на наступних step матчах, зсуваємо вікно.

    dataset — результат build_dataset() з усіма ознаками і таргетами.
    """
    dataset = dataset.sort_values("date").reset_index(drop=True)

    if len(dataset) < train_window + step:
        logger.error(f"Not enough data: need {train_window + step}, got {len(dataset)}")
        return {}

    all_results = []
    bankroll = INITIAL_BANKROLL
    start = train_window

    logger.info(f"Starting walk-forward backtest: {len(dataset)} matches, window={train_window}, step={step}")

    while start + step <= len(dataset):
        if bankroll <= 0 or (isinstance(bankroll, float) and bankroll != bankroll):  # NaN check
            logger.warning("Bankroll depleted or NaN — stopping backtest")
            break

        train_df = dataset.iloc[:start]
        test_df = dataset.iloc[start:start + step]

        # Навчаємо на тренувальному вікні
        metrics = train(train_df, version=version)

        from model.predict import load_model
        model, encoder, feature_cols = load_model(version)

        for _, row in test_df.iterrows():
            X = pd.DataFrame([row[feature_cols].fillna(0).to_dict()])
            probs = model.predict_proba(X)[0]
            prob_map = dict(zip(encoder.classes_, probs))

            h_prob = row.get("market_home_prob", 0)
            d_prob = row.get("market_draw_prob", 0)
            a_prob = row.get("market_away_prob", 0)
            if any(not p or pd.isna(p) or p <= 0 for p in (h_prob, d_prob, a_prob)):
                continue  # skip match without full odds

            # Тільки Home/Away — Draw виключено (найнепередбачуваніший результат)
            candidates = [
                # (market_name, our_prob, market_prob, winning_outcomes)
                ("home", prob_map.get("home", 0), h_prob, {"home"}),
                ("away", prob_map.get("away", 0), a_prob, {"away"}),
            ]

            for outcome, our_prob, market_prob, winning_set in candidates:
                odds = 1 / market_prob
                if odds < MIN_ODDS:
                    continue
                ev = our_prob * odds - 1
                if ev < MIN_EV:
                    continue

                b = odds - 1
                q = 1 - our_prob
                kelly = max(0, (our_prob * b - q) / b) * FRACTIONAL_KELLY
                stake = min(bankroll * kelly, bankroll * MAX_STAKE_PCT)

                actual = row["target"]
                profit = stake * (odds - 1) if actual in winning_set else -stake
                bankroll += profit

                all_results.append({
                    "date": row["date"],
                    "match_id": row.get("match_id"),
                    "outcome": outcome,
                    "actual_outcome": actual,
                    "probability": our_prob,
                    "odds": odds,
                    "stake": stake,
                    "profit": profit,
                    "bankroll": bankroll,
                    "clv": row.get(f"{outcome}_clv"),
                })

        start += step

    if not all_results:
        logger.warning("No bets placed during backtest")
        return {}

    results_df = pd.DataFrame(all_results)
    roi_metrics = compute_roi(results_df)
    roi_metrics["final_bankroll"] = round(bankroll, 2)
    roi_metrics["initial_bankroll"] = INITIAL_BANKROLL

    log_metrics(roi_metrics)
    return roi_metrics


if __name__ == "__main__":
    logger.info("Loading data from DB...")
    matches, stats, odds, teams, injuries = load_data_from_db()

    logger.info("Building features...")
    dataset = build_dataset(matches, stats, odds, teams, injuries_df=injuries)
    logger.info(f"Dataset: {len(dataset)} rows")

    backtest(dataset, train_window=1500)
