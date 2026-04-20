"""
experiments/build_feature_store.py

Фаза 1: збираємо feature store одноразово.

Expanding window — для кожного тестового матчу зберігаємо:
  - OOS model prob (XGBoost з ринком, чесний out-of-sample)
  - Всі 54 бінарних WS фактори (True/False)
  - Ринкові odds, фактичний результат

Результат: feature_store.csv → вхід для ablation_runner.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from model.train import train, load_data_from_db
from model.features.builder import build_dataset
from model.predict import load_model

ALL_LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    2:   "Champions League",
    88:  "Eredivisie",
    144: "Jupiler Pro League",
    136: "Serie B",
    94:  "Primeira Liga",
}

MIN_ODDS      = 1.5
INITIAL_TRAIN = 800
STEP          = 100

# ─── Всі 54 командних фактора (без ринку) ────────────────────────────────────

HOME_FACTORS = {
    # Elo
    "elo_gap_large":            lambda r: r.get("elo_diff", 0) > 50,
    "elo_gap_moderate":         lambda r: r.get("elo_diff", 0) > 25,
    "elo_win_prob_high":        lambda r: r.get("elo_home_win_prob", 0.5) > 0.55,
    "home_elo_strong":          lambda r: r.get("home_elo", 1500) > 1600,
    "away_elo_weak":            lambda r: r.get("away_elo", 1500) < 1400,
    # Форма
    "home_in_form":             lambda r: r.get("home_form_points", 0.5) > 0.55,
    "home_strong_form":         lambda r: r.get("home_form_points", 0.5) > 0.65,
    "away_out_of_form":         lambda r: r.get("away_form_points", 0.5) < 0.45,
    "away_poor_form":           lambda r: r.get("away_form_points", 0.5) < 0.35,
    "home_home_form":           lambda r: r.get("home_home_form_points", 0.5) > 0.55,
    "home_home_wins":           lambda r: r.get("home_home_form_wins", 0.33) > 0.60,
    "away_away_poor":           lambda r: r.get("away_away_form_points", 0.5) < 0.40,
    "away_away_loses":          lambda r: r.get("away_away_form_losses", 0.33) > 0.50,
    # xG
    "xg_ratio_home":            lambda r: (r.get("home_xg_for_avg_10", 1) / max(r.get("home_xg_against_avg_10", 1), 0.1)) > 1.2,
    "xg_diff_positive":         lambda r: r.get("home_xg_diff_avg_10", 0) > 0.3,
    "xg_attack_edge":           lambda r: r.get("home_xg_for_avg_10", 1.3) > r.get("away_xg_against_avg_10", 1.3),
    "home_xg_regression":       lambda r: r.get("home_xg_overperformance", 0) < -0.2,
    "away_xg_overperforming":   lambda r: r.get("away_xg_overperformance", 0) > 0.3,
    "home_scoring_strong":      lambda r: r.get("home_form_goals_for_avg", 1.3) > 1.8,
    "away_conceding_lots":      lambda r: r.get("away_form_goals_against_avg", 1.3) > 1.8,
    "home_defense_solid":       lambda r: r.get("home_form_goals_against_avg", 1.3) < 1.2,
    # Таблиця
    "table_home_higher":        lambda r: r.get("table_position_diff", 0) < -3,
    "table_points_home_better": lambda r: r.get("table_points_diff", 0) > 5,
    # Відпочинок
    "home_rested":              lambda r: r.get("home_rest_days", 7) >= 5,
    "away_tired":               lambda r: r.get("away_rest_days", 7) <= 3,
    "rest_advantage":           lambda r: r.get("rest_days_diff", 0) >= 3,
    # Травми
    "injury_advantage":         lambda r: r.get("away_injured_count", 0) > r.get("home_injured_count", 0),
    "big_injury_advantage":     lambda r: r.get("away_injured_count", 0) - r.get("home_injured_count", 0) >= 3,
}

AWAY_FACTORS = {
    # Elo
    "elo_gap_away_large":       lambda r: r.get("elo_diff", 0) < -50,
    "elo_gap_away_moderate":    lambda r: r.get("elo_diff", 0) < -25,
    "elo_win_prob_low":         lambda r: r.get("elo_home_win_prob", 0.5) < 0.45,
    "away_elo_strong":          lambda r: r.get("away_elo", 1500) > 1600,
    "home_elo_weak":            lambda r: r.get("home_elo", 1500) < 1400,
    # Форма
    "away_in_form":             lambda r: r.get("away_form_points", 0.5) > 0.55,
    "away_strong_form":         lambda r: r.get("away_form_points", 0.5) > 0.65,
    "home_out_of_form":         lambda r: r.get("home_form_points", 0.5) < 0.45,
    "home_poor_form":           lambda r: r.get("home_form_points", 0.5) < 0.35,
    "away_away_form":           lambda r: r.get("away_away_form_points", 0.5) > 0.50,
    "away_away_wins":           lambda r: r.get("away_away_form_wins", 0.33) > 0.50,
    "home_home_poor":           lambda r: r.get("home_home_form_points", 0.5) < 0.45,
    "home_home_loses":          lambda r: r.get("home_home_form_losses", 0.33) > 0.40,
    # xG
    "xg_ratio_away":            lambda r: (r.get("away_xg_for_avg_10", 1) / max(r.get("away_xg_against_avg_10", 1), 0.1)) > 1.2,
    "xg_diff_away_positive":    lambda r: r.get("away_xg_diff_avg_10", 0) > 0.3,
    "xg_away_attack":           lambda r: r.get("away_xg_for_avg_10", 1.3) > r.get("home_xg_against_avg_10", 1.3),
    "away_xg_regression":       lambda r: r.get("away_xg_overperformance", 0) < -0.2,
    "home_xg_overperforming":   lambda r: r.get("home_xg_overperformance", 0) > 0.3,
    "away_scoring_strong":      lambda r: r.get("away_form_goals_for_avg", 1.3) > 1.8,
    "home_conceding_lots":      lambda r: r.get("home_form_goals_against_avg", 1.3) > 1.8,
    "away_defense_solid":       lambda r: r.get("away_form_goals_against_avg", 1.3) < 1.2,
    # Таблиця
    "table_away_higher":        lambda r: r.get("table_position_diff", 0) > 3,
    "table_points_away_better": lambda r: r.get("table_points_diff", 0) < -5,
    # Відпочинок
    "away_rested":              lambda r: r.get("away_rest_days", 7) >= 5,
    "home_tired":               lambda r: r.get("home_rest_days", 7) <= 3,
    "rest_advantage_away":      lambda r: r.get("rest_days_diff", 0) <= -3,
    # Травми
    "injury_adv_away":          lambda r: r.get("home_injured_count", 0) > r.get("away_injured_count", 0),
    "big_injury_adv_away":      lambda r: r.get("home_injured_count", 0) - r.get("away_injured_count", 0) >= 3,
}

# Групи для ablation
FACTOR_GROUPS = {
    "elo":    ["elo_gap_large", "elo_gap_moderate", "elo_win_prob_high", "home_elo_strong", "away_elo_weak",
               "elo_gap_away_large", "elo_gap_away_moderate", "elo_win_prob_low", "away_elo_strong", "home_elo_weak"],
    "form":   ["home_in_form", "home_strong_form", "away_out_of_form", "away_poor_form",
               "home_home_form", "home_home_wins", "away_away_poor", "away_away_loses",
               "away_in_form", "away_strong_form", "home_out_of_form", "home_poor_form",
               "away_away_form", "away_away_wins", "home_home_poor", "home_home_loses"],
    "xg":     ["xg_ratio_home", "xg_diff_positive", "xg_attack_edge", "home_xg_regression",
               "away_xg_overperforming", "home_scoring_strong", "away_conceding_lots", "home_defense_solid",
               "xg_ratio_away", "xg_diff_away_positive", "xg_away_attack", "away_xg_regression",
               "home_xg_overperforming", "away_scoring_strong", "home_conceding_lots", "away_defense_solid"],
    "table":  ["table_home_higher", "table_points_home_better", "table_away_higher", "table_points_away_better"],
    "rest":   ["home_rested", "away_tired", "rest_advantage", "away_rested", "home_tired", "rest_advantage_away"],
    "injury": ["injury_advantage", "big_injury_advantage", "injury_adv_away", "big_injury_adv_away"],
}


def get_league_db_ids(api_ids: dict) -> dict:
    from db.session import SessionLocal
    from db.models import League
    db = SessionLocal()
    try:
        result = {}
        for api_id, name in api_ids.items():
            for l in db.query(League).filter_by(api_id=api_id).all():
                result[l.id] = name
        return result
    finally:
        db.close()


def compute_factors(row: dict, outcome: str) -> dict:
    factors = HOME_FACTORS if outcome == "home" else AWAY_FACTORS
    result = {}
    for name, fn in factors.items():
        try:
            result[f"f_{outcome}_{name}"] = int(fn(row))
        except Exception:
            result[f"f_{outcome}_{name}"] = 0
    return result


def run():
    logger.info("=== Feature Store: expanding window ===")

    league_id_to_name = get_league_db_ids(ALL_LEAGUES)

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую фічі...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset = dataset[dataset["league_id"].isin(league_id_to_name.keys())].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    logger.info(f"Датасет: {len(dataset)} матчів")

    start   = INITIAL_TRAIN
    fold    = 0
    version = "feature_store"
    folds_total = (len(dataset) - INITIAL_TRAIN) // STEP
    logger.info(f"Expanding window: folds={folds_total}")

    store = []

    while start + STEP <= len(dataset):
        fold += 1
        train_df = dataset.iloc[:start]
        test_df  = dataset.iloc[start:start + STEP]

        train(train_df, version=version)
        model, encoder, feature_cols = load_model(version)

        for _, row in test_df.iterrows():
            h_prob = row.get("market_home_prob", 0)
            a_prob = row.get("market_away_prob", 0)
            if not h_prob or not a_prob or pd.isna(h_prob) or pd.isna(a_prob):
                continue

            X = pd.DataFrame([row[feature_cols].fillna(0).to_dict()])
            probs = model.predict_proba(X)[0]
            prob_map = dict(zip(encoder.classes_, probs))

            row_dict = row.to_dict()

            for outcome in ("home", "away"):
                market_prob = h_prob if outcome == "home" else a_prob
                if not market_prob or market_prob <= 0:
                    continue
                odds = 1 / market_prob
                if odds < MIN_ODDS:
                    continue

                our_prob = float(prob_map.get(outcome, 0))
                ev = our_prob * odds - 1

                record = {
                    "date":        row["date"],
                    "league":      league_id_to_name.get(row.get("league_id"), "Other"),
                    "outcome":     outcome,
                    "actual":      row["target"],
                    "model_prob":  round(our_prob, 4),
                    "market_prob": round(market_prob, 4),
                    "odds":        round(odds, 3),
                    "ev":          round(ev * 100, 2),
                    "won":         row["target"] == outcome,
                    "fold":        fold,
                }
                # Додаємо всі бінарні фактори
                record.update(compute_factors(row_dict, outcome))
                store.append(record)

        logger.info(f"  Fold {fold:02d} | train={start} | records={len(store)}")
        start += STEP

    df = pd.DataFrame(store)
    out = os.path.join(os.path.dirname(__file__), "results", "feature_store.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    factor_cols = [c for c in df.columns if c.startswith("f_")]
    logger.info(f"\nЗбережено {len(df)} записів → {out}")
    logger.info(f"Факторів: {len(factor_cols)}")
    logger.info(f"Ліги: {sorted(df['league'].unique())}")
    logger.info(f"EV range: {df['ev'].min():.1f}% – {df['ev'].max():.1f}%")


if __name__ == "__main__":
    run()
