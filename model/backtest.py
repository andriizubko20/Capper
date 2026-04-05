import pandas as pd
from loguru import logger

from model.train import train, FEATURE_COLS, load_data_from_db
from model.evaluate import compute_roi, log_metrics
from model.features.builder import build_dataset

# 1X2 (Home/Away only)
MIN_EV = 0.17
MIN_ODDS = 1.5

# Double Chance (1X / 2X) — нижчі кефи, окремі пороги
MIN_EV_DC = 0.10
MIN_ODDS_DC = 1.35

MAX_STAKE_PCT = 0.04       # максимум 4% банкролу незалежно від Kelly
FRACTIONAL_KELLY = 0.25
INITIAL_BANKROLL = 1000.0

# Scenario filter — мінімум факторів для ставки
MIN_SCENARIO_SCORE = 3


def scenario_score(row: pd.Series, outcome: str) -> int:
    """
    Рахує кількість збіжних факторів для конкретного виходу.
    Ставимо тільки якщо score >= MIN_SCENARIO_SCORE.
    """
    score = 0

    home_form = row.get("home_form_points", 0.5)
    away_form = row.get("away_form_points", 0.5)
    elo_diff = row.get("elo_diff", 0)
    home_xg_for = row.get("home_xg_for_avg_10", 1.3)
    away_xg_for = row.get("away_xg_for_avg_10", 1.3)
    home_xg_against = row.get("home_xg_against_avg_10", 1.3)
    away_xg_against = row.get("away_xg_against_avg_10", 1.3)
    market_home = row.get("market_home_prob", 0.33)
    market_away = row.get("market_away_prob", 0.33)

    home_rest = row.get("home_rest_days", 7)
    away_rest = row.get("away_rest_days", 7)
    home_injured = row.get("home_injured_count", 0)
    away_injured = row.get("away_injured_count", 0)

    if outcome == "home":
        if market_home > 0.50:            score += 1  # ринок підтримує фаворита
        if home_form > 0.55:              score += 1  # домашня команда у формі
        if away_form < 0.45:              score += 1  # гості в поганій формі
        if elo_diff > 50:                 score += 1  # суттєва перевага в Elo
        if home_xg_for > away_xg_against: score += 1  # атака сильніша за захист гостей
        if home_rest >= 5:                score += 1  # відпочинок домашньої команди
        if away_injured > home_injured:   score += 1  # перевага по травмах
        if elo_diff > 25:                 score += 1  # помірна Elo перевага

    elif outcome == "away":
        if market_away > 0.38:            score += 1  # ринок бачить шанси гостей
        if away_form > 0.55:              score += 1  # гості у формі
        if home_form < 0.45:              score += 1  # домашня команда в поганій формі
        if elo_diff < -50:                score += 1  # гості сильніші за Elo
        if away_xg_for > home_xg_against: score += 1  # атака гостей сильніша за захист хозяїв
        if away_rest >= 5:                score += 1  # відпочинок гостей
        if home_injured > away_injured:   score += 1  # перевага по травмах
        if elo_diff < -25:                score += 1  # помірна Elo перевага гостей

    return score


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

            p_home = prob_map.get("home", 0)
            p_draw = prob_map.get("draw", 0)
            p_away = prob_map.get("away", 0)

            # 1X2: Home/Away
            candidates_1x2 = [
                ("home", p_home, h_prob, {"home"}, MIN_EV, MIN_ODDS),
                ("away", p_away, a_prob, {"away"}, MIN_EV, MIN_ODDS),
            ]

            # Double Chance: 1X і 2X — реальні odds з ринку
            dc_1x_odds = row.get("dc_1x_odds")
            dc_2x_odds = row.get("dc_2x_odds")
            candidates_dc = []
            if dc_1x_odds and not pd.isna(dc_1x_odds) and dc_1x_odds >= MIN_ODDS_DC:
                candidates_dc.append(("1X", p_home + p_draw, 1 / dc_1x_odds, {"home", "draw"}, MIN_EV_DC, MIN_ODDS_DC))
            if dc_2x_odds and not pd.isna(dc_2x_odds) and dc_2x_odds >= MIN_ODDS_DC:
                candidates_dc.append(("2X", p_away + p_draw, 1 / dc_2x_odds, {"away", "draw"}, MIN_EV_DC, MIN_ODDS_DC))

            # Обираємо найкращу ставку на матч (найвищий EV серед всіх кандидатів)
            best_bet = None
            best_ev = -1

            for outcome, our_prob, market_prob, winning_set, min_ev, min_odds in candidates_1x2 + candidates_dc:
                if not market_prob or market_prob <= 0:
                    continue
                odds = 1 / market_prob
                if odds < min_odds:
                    continue
                # Scenario filter — тільки для 1X2, DC пропускаємо
                if outcome in ("home", "away"):
                    if scenario_score(row, outcome) < MIN_SCENARIO_SCORE:
                        continue
                ev = our_prob * odds - 1
                if ev < min_ev:
                    continue
                if ev > best_ev:
                    best_ev = ev
                    best_bet = (outcome, our_prob, odds, winning_set)

            if best_bet is None:
                continue

            outcome, our_prob, odds, winning_set = best_bet
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
                "clv": row.get(f"{outcome}_clv") if outcome in ("home", "away") else None,
            })

        start += step

    if not all_results:
        logger.warning("No bets placed during backtest")
        return {}

    results_df = pd.DataFrame(all_results)
    roi_metrics = compute_roi(results_df)
    roi_metrics["final_bankroll"] = round(bankroll, 2)
    roi_metrics["initial_bankroll"] = INITIAL_BANKROLL

    logger.info(f"Scenario filter: MIN_SCORE={MIN_SCENARIO_SCORE}, MIN_EV={MIN_EV:.0%}")

    # Розбивка по типу ставки
    for market in ("home", "away", "1X", "2X"):
        subset = results_df[results_df["outcome"] == market]
        if not subset.empty:
            wr = subset.apply(lambda r: r["actual_outcome"] in (
                {"home"} if market == "home" else
                {"away"} if market == "away" else
                {"home", "draw"} if market == "1X" else {"away", "draw"}
            ), axis=1).mean()
            logger.info(f"  {market}: {len(subset)} bets, win_rate={wr:.1%}")

    log_metrics(roi_metrics)
    return roi_metrics


if __name__ == "__main__":
    logger.info("Loading data from DB...")
    matches, stats, odds, teams, injuries = load_data_from_db()

    logger.info("Building features...")
    dataset = build_dataset(matches, stats, odds, teams, injuries_df=injuries)
    logger.info(f"Dataset: {len(dataset)} rows")

    backtest(dataset, train_window=1500)
