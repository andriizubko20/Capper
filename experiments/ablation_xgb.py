"""
experiments/ablation_xgb.py

Grid search по фіч-групах + EV/odds порогах для XGBoost моделі.

Алгоритм:
  1. Будуємо датасет один раз
  2. Для кожної комбінації груп фіч (~64 комбо): тренуємо 2 фолди, зберігаємо ймовірності
  3. Для кожного EV-порогу і MAX_ODDS: фільтруємо ставки без ретренування
  4. Виводимо топ-30 конфігурацій по OOS ROI

Walk-forward: Fold1 (Jan–Jun 2025 OOS), Fold2 (Jul 2025–зараз OOS).

Запуск: python -m experiments.ablation_xgb
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import product
from loguru import logger
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from model.train import load_data_from_db
from model.features.builder import build_dataset

SPLIT_1 = pd.Timestamp("2025-01-01")
SPLIT_2 = pd.Timestamp("2025-07-01")

INITIAL_BR   = 1000.0
MAX_STAKE    = 0.04
FRACTIONAL   = 0.25
MIN_BETS     = 40        # мінімум ставок щоб результат мав сенс

LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    94:  "Primeira Liga",
}

# ─── Групи фіч ────────────────────────────────────────────────────────────────
# market_base завжди включений — це найсильніший сигнал
ALWAYS = [
    "market_home_prob", "market_away_prob", "market_draw_prob",
]

FEATURE_GROUPS = {
    "form_basic": [
        "home_form_points", "home_form_goals_for_avg", "home_form_goals_against_avg",
        "home_form_wins", "home_form_losses",
        "away_form_points", "away_form_goals_for_avg", "away_form_goals_against_avg",
        "away_form_wins", "away_form_losses",
        "home_home_form_wins", "home_home_form_goals_for_avg",
        "away_away_form_wins", "away_away_form_goals_for_avg",
    ],
    "form_adv": [
        "home_clean_sheet_rate", "home_btts_rate", "home_failed_to_score_rate",
        "home_win_streak", "home_loss_streak", "home_goals_for_std",
        "away_clean_sheet_rate", "away_btts_rate", "away_failed_to_score_rate",
        "away_win_streak", "away_loss_streak", "away_goals_for_std",
        "delta_clean_sheet_rate", "delta_btts_rate", "delta_failed_to_score_rate",
    ],
    "xg": [
        "home_xg_for_avg", "home_xg_against_avg", "home_xg_ratio",
        "home_xg_for_avg_10", "home_xg_against_avg_10", "home_xg_ratio_10",
        "away_xg_for_avg", "away_xg_against_avg", "away_xg_ratio",
        "away_xg_for_avg_10", "away_xg_against_avg_10", "away_xg_ratio_10",
        "home_xg_overperformance", "away_xg_overperformance",
    ],
    "elo": [
        "elo_diff", "elo_home_win_prob",
        "home_elo_momentum", "away_elo_momentum", "delta_elo_momentum",
    ],
    "standings": [
        "home_table_position", "home_table_points", "home_table_goal_diff",
        "away_table_position", "away_table_points", "away_table_goal_diff",
        "table_position_diff", "table_points_diff",
    ],
    "rest": [
        "home_rest_days", "away_rest_days", "rest_days_diff",
    ],
    "market_ext": [
        "market_certainty", "market_home_edge",
    ],
}

# Пороги для перебору (без ретренування)
EV_THRESHOLDS   = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
MAX_ODDS_VALUES = [2.2, 2.5, 3.0, 99.0]   # 99 = без обмеження
MIN_ODDS        = 1.5


def get_league_db_ids(db):
    from db.models import League
    result = {}
    for api_id, name in LEAGUES.items():
        for l in db.query(League).filter_by(api_id=api_id).all():
            result[l.id] = name
    return result


def train_model(X_tr, y_tr):
    split = int(len(X_tr) * 0.85)
    base = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    base.fit(X_tr.iloc[:split], y_tr[:split],
             eval_set=[(X_tr.iloc[split:], y_tr[split:])], verbose=False)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)
    return cal


def collect_probs(model, df, features, le):
    X = df[features].fillna(0)
    probs = model.predict_proba(X)
    class_idx = {c: i for i, c in enumerate(le.classes_)}
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        h_mkt = row.get("market_home_prob")
        a_mkt = row.get("market_away_prob")
        if not h_mkt or not a_mkt or pd.isna(h_mkt):
            continue
        for side, mkt_p in [("home", h_mkt), ("away", a_mkt)]:
            odds = 1 / mkt_p
            if odds < MIN_ODDS:
                continue
            records.append({
                "date":       row["date"],
                "league":     row.get("league", ""),
                "side":       side,
                "won":        row["target"] == side,
                "odds":       round(odds, 3),
                "model_prob": probs[i][class_idx[side]],
                "mkt_prob":   mkt_p,
            })
    return pd.DataFrame(records)


def compute_roi(bets: pd.DataFrame) -> tuple[float, float, float]:
    """Повертає (win_rate, flat_roi, kelly_roi)."""
    if len(bets) < MIN_BETS:
        return 0.0, -999.0, -999.0
    wr = bets["won"].mean()
    flat = ((bets["won"] * (bets["odds"] - 1)) - (~bets["won"])).sum() / len(bets) * 100

    bankroll, staked = INITIAL_BR, 0.0
    for _, b in bets.sort_values("date").iterrows():
        p, odd = b["model_prob"], b["odds"]
        kelly = max(0.0, (p * (odd - 1) - (1 - p)) / (odd - 1)) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * MAX_STAKE)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked   += stake
    kelly_roi = (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else -999.0
    return wr, flat, kelly_roi


def run():
    from db.session import SessionLocal
    db = SessionLocal()
    try:
        league_map = get_league_db_ids(db)
    finally:
        db.close()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую датасет...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset[dataset["league_id"].isin(league_map.keys())].copy()
    dataset = dataset[dataset["market_home_prob"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)
    logger.info(f"Датасет: {len(dataset)} матчів")

    le = LabelEncoder()
    le.fit(["away", "draw", "home"])

    fold1_tr = dataset[dataset["date"] < SPLIT_1]
    fold1_ts = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)]
    fold2_tr = dataset[dataset["date"] < SPLIT_2]
    fold2_ts = dataset[dataset["date"] >= SPLIT_2]
    logger.info(f"Fold1: train={len(fold1_tr)}, test={len(fold1_ts)}")
    logger.info(f"Fold2: train={len(fold2_tr)}, test={len(fold2_ts)}")

    # ─── Перебір комбінацій груп ──────────────────────────────────────────────
    group_names = list(FEATURE_GROUPS.keys())
    n_groups    = len(group_names)
    total_combos = 2 ** n_groups
    logger.info(f"Груп: {n_groups} → {total_combos} комбінацій фіч × {len(EV_THRESHOLDS)} EV × {len(MAX_ODDS_VALUES)} odds = {total_combos * len(EV_THRESHOLDS) * len(MAX_ODDS_VALUES)} конфігів")

    all_results = []
    processed   = 0

    for mask in range(total_combos):
        active = [group_names[i] for i in range(n_groups) if mask & (1 << i)]
        features = ALWAYS[:]
        for g in active:
            features += FEATURE_GROUPS[g]
        # Залишаємо тільки ті що є в датасеті
        features = [f for f in features if f in dataset.columns]

        if len(features) < 3:
            continue

        label = "+".join(active) if active else "market_only"

        try:
            y1 = le.transform(fold1_tr["target"])
            y2 = le.transform(fold2_tr["target"])
            m1 = train_model(fold1_tr[features].fillna(0), y1)
            m2 = train_model(fold2_tr[features].fillna(0), y2)
        except Exception as e:
            logger.warning(f"  {label}: помилка тренування — {e}")
            continue

        probs1 = collect_probs(m1, fold1_ts, features, le)
        probs2 = collect_probs(m2, fold2_ts, features, le)
        all_probs = pd.concat([probs1, probs2], ignore_index=True)

        if all_probs.empty:
            continue

        # Додаємо EV для всіх записів
        all_probs["ev"] = all_probs["model_prob"] * all_probs["odds"] - 1

        # Перебір порогів
        for min_ev, max_odds in product(EV_THRESHOLDS, MAX_ODDS_VALUES):
            bets = all_probs[
                (all_probs["ev"] >= min_ev) &
                (all_probs["odds"] <= max_odds)
            ]
            wr, flat_roi, kelly_roi = compute_roi(bets)
            if flat_roi == -999.0:
                continue
            all_results.append({
                "features":  label,
                "n_groups":  len(active),
                "n_feats":   len(features),
                "min_ev":    min_ev,
                "max_odds":  max_odds,
                "n_bets":    len(bets),
                "win_rate":  round(wr * 100, 1),
                "flat_roi":  round(flat_roi, 2),
                "kelly_roi": round(kelly_roi, 2),
                "bankroll":  round(INITIAL_BR + INITIAL_BR * kelly_roi / 100, 2),
            })

        processed += 1
        if processed % 10 == 0:
            logger.info(f"  {processed}/{total_combos} комбінацій...")

    logger.info(f"Готово. Всього результатів: {len(all_results)}")

    if not all_results:
        logger.error("Немає результатів!")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("flat_roi", ascending=False)

    # ─── Топ-30 конфігурацій ─────────────────────────────────────────────────
    print("\n" + "="*100)
    print(f"  ТОП-30 КОНФІГУРАЦІЙ (з {len(results_df)} валідних) — сортування по Flat ROI OOS")
    print("="*100)
    print(f"  {'Групи фіч':<45} {'EV≥':>5} {'Odds≤':>6} {'Ставок':>7} {'Win%':>6} {'FlatROI':>8} {'KellyROI':>9} {'Банкрол':>9}")
    print("  " + "-"*100)
    for _, r in results_df.head(30).iterrows():
        print(
            f"  {r['features']:<45} {r['min_ev']*100:>4.0f}% {r['max_odds']:>6.1f} "
            f"{r['n_bets']:>7} {r['win_rate']:>5.1f}% {r['flat_roi']:>+8.1f}% "
            f"{r['kelly_roi']:>+9.1f}% ${r['bankroll']:>8.0f}"
        )

    # ─── Найкраща конфігурація детально ──────────────────────────────────────
    best = results_df.iloc[0]
    print(f"\n{'='*100}")
    print(f"  НАЙКРАЩЕ: {best['features']}")
    print(f"  EV≥{best['min_ev']*100:.0f}%, Odds≤{best['max_odds']}, "
          f"{best['n_bets']} ставок, Win {best['win_rate']:.1f}%, "
          f"Flat ROI {best['flat_roi']:+.1f}%, Kelly ROI {best['kelly_roi']:+.1f}%, "
          f"Банкрол ${best['bankroll']:.0f}")

    # ─── Аналіз впливу кожної групи ──────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  ВПЛИВ КОЖНОЇ ГРУПИ (avg flat ROI при включенні vs виключенні)")
    print(f"  {'Група':<20} {'Avg ROI якщо ON':>16} {'Avg ROI якщо OFF':>16} {'Δ':>8}")
    print("  " + "-"*65)
    for g in group_names:
        on  = results_df[results_df["features"].str.contains(g)]["flat_roi"].mean()
        off = results_df[~results_df["features"].str.contains(g)]["flat_roi"].mean()
        marker = "✅" if (on - off) > 0.5 else ("❌" if (on - off) < -0.5 else "⬜")
        print(f"  {marker} {g:<20} {on:>+15.2f}% {off:>+15.2f}% {(on-off):>+8.2f}%")

    # ─── EV threshold аналіз ─────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  ВПЛИВ EV ПОРОГУ (avg flat ROI по всіх конфігураціях)")
    print(f"  {'EV≥':<8} {'Avg ROI':>10} {'Avg ставок':>12}")
    print("  " + "-"*35)
    for ev in sorted(results_df["min_ev"].unique()):
        g = results_df[results_df["min_ev"] == ev]
        print(f"  {ev*100:>4.0f}%   {g['flat_roi'].mean():>+10.2f}%   {g['n_bets'].mean():>10.0f}")

    # ─── MAX ODDS аналіз ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  ВПЛИВ MAX ODDS (avg flat ROI по всіх конфігураціях)")
    print(f"  {'Odds≤':<8} {'Avg ROI':>10} {'Avg ставок':>12}")
    print("  " + "-"*35)
    for mo in sorted(results_df["max_odds"].unique()):
        g = results_df[results_df["max_odds"] == mo]
        label = f"{mo:.1f}" if mo < 50 else "без ліміту"
        print(f"  {label:<8} {g['flat_roi'].mean():>+10.2f}%   {g['n_bets'].mean():>10.0f}")

    # ─── Зберігаємо ──────────────────────────────────────────────────────────
    os.makedirs("experiments/results", exist_ok=True)
    out = "experiments/results/ablation_xgb.csv"
    results_df.to_csv(out, index=False)
    logger.info(f"Збережено: {out}")


if __name__ == "__main__":
    run()
