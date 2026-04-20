"""
experiments/backtest_xgb_ws_features.py

XGBoost з 54 бінарними WS факторами як фічами моделі.
WS фільтр прибрано — тільки EV threshold.

Grid search: EV × Kelly cap (включно без cap).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from itertools import product
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset

ALL_LEAGUES = {
    39: "Premier League", 140: "La Liga", 78: "Bundesliga",
    135: "Serie A", 61: "Ligue 1", 2: "Champions League",
    88: "Eredivisie", 144: "Jupiler Pro League", 136: "Serie B", 94: "Primeira Liga",
}

INITIAL_BANKROLL = 1000.0
FRACTIONAL_KELLY = 0.25
MIN_ODDS         = 1.5
INITIAL_TRAIN    = 800
STEP             = 100
MIN_BETS         = 30

EV_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20]
KELLY_CAPS    = [0.02, 0.03, 0.04, 0.06, 0.08, 1.00]   # 1.00 = без cap (тільки fractional kelly)

# ─── Оригінальні фічі ─────────────────────────────────────────────────────────
BASE_FEATURES = [
    "home_form_points", "home_form_goals_for_avg", "home_form_goals_against_avg",
    "away_form_points", "away_form_goals_for_avg", "away_form_goals_against_avg",
    "home_xg_for_avg_10", "home_xg_against_avg_10",
    "away_xg_for_avg_10", "away_xg_against_avg_10",
    "home_xg_diff_avg_10", "away_xg_diff_avg_10",
    "home_xg_overperformance", "away_xg_overperformance",
    "elo_diff", "home_elo", "away_elo", "elo_home_win_prob",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_injured_count", "away_injured_count",
    "table_position_diff", "table_points_diff",
    "home_home_form_points", "home_home_form_wins",
    "away_away_form_points", "away_away_form_wins",
    "market_home_prob", "market_away_prob", "market_draw_prob",
]

# ─── Всі 54 бінарних WS факторів ─────────────────────────────────────────────
WS_FACTORS_HOME = {
    "elo_gap_large":            lambda r: int(r.get("elo_diff", 0) > 50),
    "elo_gap_moderate":         lambda r: int(r.get("elo_diff", 0) > 25),
    "elo_win_prob_high":        lambda r: int(r.get("elo_home_win_prob", 0.5) > 0.55),
    "home_elo_strong":          lambda r: int(r.get("home_elo", 1500) > 1600),
    "away_elo_weak":            lambda r: int(r.get("away_elo", 1500) < 1400),
    "home_in_form":             lambda r: int(r.get("home_form_points", 0.5) > 0.55),
    "home_strong_form":         lambda r: int(r.get("home_form_points", 0.5) > 0.65),
    "away_out_of_form":         lambda r: int(r.get("away_form_points", 0.5) < 0.45),
    "away_poor_form":           lambda r: int(r.get("away_form_points", 0.5) < 0.35),
    "home_home_form":           lambda r: int(r.get("home_home_form_points", 0.5) > 0.55),
    "home_home_wins":           lambda r: int(r.get("home_home_form_wins", 0.33) > 0.60),
    "away_away_poor":           lambda r: int(r.get("away_away_form_points", 0.5) < 0.40),
    "xg_ratio_home":            lambda r: int((r.get("home_xg_for_avg_10", 1) / max(r.get("home_xg_against_avg_10", 1), 0.1)) > 1.2),
    "xg_diff_positive":         lambda r: int(r.get("home_xg_diff_avg_10", 0) > 0.3),
    "xg_attack_edge":           lambda r: int(r.get("home_xg_for_avg_10", 1.3) > r.get("away_xg_against_avg_10", 1.3)),
    "home_scoring_strong":      lambda r: int(r.get("home_form_goals_for_avg", 1.3) > 1.8),
    "away_conceding_lots":      lambda r: int(r.get("away_form_goals_against_avg", 1.3) > 1.8),
    "home_defense_solid":       lambda r: int(r.get("home_form_goals_against_avg", 1.3) < 1.2),
    "table_home_higher":        lambda r: int(r.get("table_position_diff", 0) < -3),
    "table_points_home_better": lambda r: int(r.get("table_points_diff", 0) > 5),
    "home_rested":              lambda r: int(r.get("home_rest_days", 7) >= 5),
    "away_tired":               lambda r: int(r.get("away_rest_days", 7) <= 3),
    "injury_advantage":         lambda r: int(r.get("away_injured_count", 0) > r.get("home_injured_count", 0)),
    "market_favors_home":       lambda r: int(r.get("market_home_prob", 0) > 0.50),
    "market_strong_home":       lambda r: int(r.get("market_home_prob", 0) > 0.60),
}

WS_FACTORS_AWAY = {
    "elo_gap_away_large":       lambda r: int(r.get("elo_diff", 0) < -50),
    "elo_gap_away_moderate":    lambda r: int(r.get("elo_diff", 0) < -25),
    "elo_win_prob_low":         lambda r: int(r.get("elo_home_win_prob", 0.5) < 0.45),
    "away_elo_strong":          lambda r: int(r.get("away_elo", 1500) > 1600),
    "home_elo_weak":            lambda r: int(r.get("home_elo", 1500) < 1400),
    "away_in_form":             lambda r: int(r.get("away_form_points", 0.5) > 0.55),
    "away_strong_form":         lambda r: int(r.get("away_form_points", 0.5) > 0.65),
    "home_out_of_form":         lambda r: int(r.get("home_form_points", 0.5) < 0.45),
    "home_poor_form":           lambda r: int(r.get("home_form_points", 0.5) < 0.35),
    "away_away_form":           lambda r: int(r.get("away_away_form_points", 0.5) > 0.50),
    "away_away_wins":           lambda r: int(r.get("away_away_form_wins", 0.33) > 0.50),
    "home_home_poor":           lambda r: int(r.get("home_home_form_points", 0.5) < 0.45),
    "xg_ratio_away":            lambda r: int((r.get("away_xg_for_avg_10", 1) / max(r.get("away_xg_against_avg_10", 1), 0.1)) > 1.2),
    "xg_diff_away_positive":    lambda r: int(r.get("away_xg_diff_avg_10", 0) > 0.3),
    "xg_away_attack":           lambda r: int(r.get("away_xg_for_avg_10", 1.3) > r.get("home_xg_against_avg_10", 1.3)),
    "away_scoring_strong":      lambda r: int(r.get("away_form_goals_for_avg", 1.3) > 1.8),
    "home_conceding_lots":      lambda r: int(r.get("home_form_goals_against_avg", 1.3) > 1.8),
    "away_defense_solid":       lambda r: int(r.get("away_form_goals_against_avg", 1.3) < 1.2),
    "table_away_higher":        lambda r: int(r.get("table_position_diff", 0) > 3),
    "table_points_away_better": lambda r: int(r.get("table_points_diff", 0) < -5),
    "away_rested":              lambda r: int(r.get("away_rest_days", 7) >= 5),
    "home_tired":               lambda r: int(r.get("home_rest_days", 7) <= 3),
    "injury_adv_away":          lambda r: int(r.get("home_injured_count", 0) > r.get("away_injured_count", 0)),
    "market_sees_away":         lambda r: int(r.get("market_away_prob", 0) > 0.38),
    "market_strong_away":       lambda r: int(r.get("market_away_prob", 0) > 0.50),
}


def compute_ws_features(row: dict) -> dict:
    result = {}
    for name, fn in WS_FACTORS_HOME.items():
        result[f"ws_h_{name}"] = fn(row)
    for name, fn in WS_FACTORS_AWAY.items():
        result[f"ws_a_{name}"] = fn(row)
    return result


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


def train_model(train_df: pd.DataFrame) -> tuple:
    df = train_df[train_df["market_home_prob"].notna()].copy()
    available = [f for f in BASE_FEATURES if f in df.columns]
    ws_cols   = [c for c in df.columns if c.startswith("ws_h_") or c.startswith("ws_a_")]
    all_feats = available + ws_cols

    X = df[all_feats].fillna(0)
    le = LabelEncoder()
    le.fit(["away", "draw", "home"])
    y = le.transform(df["target"])

    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y[:split], y[split:]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    acc = (model.predict(X_val) == y_val).mean() if len(X_val) else 0
    logger.debug(f"    val_acc={acc:.3f} | feats={len(all_feats)}")
    return model, le, all_feats


def simulate_kelly(df: pd.DataFrame, kelly_cap: float) -> tuple[float, float, float]:
    bankroll = INITIAL_BANKROLL
    staked   = 0.0
    for _, row in df.sort_values("date").iterrows():
        p    = row["prob"]
        odds = row["odds"]
        b    = odds - 1
        kelly = max(0.0, (p * b - (1 - p)) / b) * FRACTIONAL_KELLY
        cap   = bankroll * kelly_cap if kelly_cap < 1.0 else bankroll  # no cap = fractional kelly only
        stake = min(bankroll * kelly, cap)
        if stake <= 0 or bankroll <= 0:
            continue
        bankroll += stake * b if row["won"] else -stake
        staked   += stake
    profit = bankroll - INITIAL_BANKROLL
    roi = profit / staked * 100 if staked > 0 else 0.0
    return round(bankroll, 2), round(profit, 2), round(roi, 2)


def run():
    logger.info("=== Backtest: XGBoost + WS factors as features (no WS filter) ===")

    league_id_to_name = get_league_db_ids(ALL_LEAGUES)

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую фічі...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset = dataset[dataset["league_id"].isin(league_id_to_name.keys())].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    logger.info(f"Датасет: {len(dataset)} матчів")

    # Додаємо WS бінарні фічі до датасету
    logger.info("Рахую WS фічі...")
    ws_rows = [compute_ws_features(r) for r in dataset.to_dict("records")]
    ws_df = pd.DataFrame(ws_rows, index=dataset.index)
    dataset = pd.concat([dataset, ws_df], axis=1)
    logger.info(f"Додано {len(ws_df.columns)} WS бінарних фіч")

    start = INITIAL_TRAIN
    fold  = 0
    folds_total = (len(dataset) - INITIAL_TRAIN) // STEP
    logger.info(f"Expanding window: folds={folds_total}")

    all_candidates = []

    while start + STEP <= len(dataset):
        fold += 1
        train_df = dataset.iloc[:start].copy()
        test_df  = dataset.iloc[start:start + STEP].copy()

        model, le, feature_cols = train_model(train_df)

        fold_count = 0
        for _, row in test_df.iterrows():
            h_prob = row.get("market_home_prob", 0)
            a_prob = row.get("market_away_prob", 0)
            if not h_prob or not a_prob or pd.isna(h_prob) or pd.isna(a_prob):
                continue

            X = pd.DataFrame([row[feature_cols].fillna(0).to_dict()])
            probs = model.predict_proba(X)[0]
            prob_map = dict(zip(le.classes_, probs))

            best_bet = None
            best_ev  = -999

            for outcome in ("home", "away"):
                market_prob = h_prob if outcome == "home" else a_prob
                if not market_prob or market_prob <= 0:
                    continue
                odds = 1 / market_prob
                if odds < MIN_ODDS:
                    continue

                our_prob = float(prob_map.get(outcome, 0))
                ev = our_prob * odds - 1

                if ev > best_ev:
                    best_ev  = ev
                    best_bet = (outcome, our_prob, odds)

            if best_bet is None:
                continue

            outcome, our_prob, odds = best_bet
            all_candidates.append({
                "date":    row["date"],
                "league":  league_id_to_name.get(row.get("league_id"), "Other"),
                "outcome": outcome,
                "actual":  row["target"],
                "prob":    round(our_prob, 4),
                "odds":    round(odds, 2),
                "ev":      round(best_ev * 100, 2),
                "won":     row["target"] == outcome,
                "fold":    fold,
            })
            fold_count += 1

        logger.info(f"  Fold {fold:02d} | train={start} | candidates={fold_count}")
        start += STEP

    if not all_candidates:
        logger.warning("Жодного кандидата")
        return

    cdf = pd.DataFrame(all_candidates)
    cdf["date"] = pd.to_datetime(cdf["date"])
    logger.info(f"\nКандидатів: {len(cdf)} | win (all): {cdf['won'].mean():.1%}")
    logger.info(f"EV: {cdf['ev'].min():.1f}% – {cdf['ev'].max():.1f}% | avg odds: {cdf['odds'].mean():.2f}")

    # ─── Grid search ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("GRID SEARCH: EV threshold × Kelly cap")
    logger.info(f"{'='*80}")
    logger.info(f"{'EV≥':>5} {'Cap':>6} {'Ставок':>8} {'Win%':>7} {'Avg odds':>9} {'ROI':>8} {'P&L':>10} {'Банкрол':>10}")
    logger.info("-"*80)

    results = []
    for ev_thr, kelly_cap in product(EV_THRESHOLDS, KELLY_CAPS):
        filtered = (
            cdf[cdf["ev"] >= ev_thr * 100]
            .sort_values("ev", ascending=False)
            .drop_duplicates(subset=["date", "fold"])
            .copy()
        )
        if len(filtered) < MIN_BETS:
            continue

        final_br, profit, roi = simulate_kelly(filtered, kelly_cap)
        cap_label = f"{kelly_cap*100:.0f}%" if kelly_cap < 1.0 else "none"
        results.append({
            "ev_thr": ev_thr, "kelly_cap": kelly_cap, "cap_label": cap_label,
            "bets": len(filtered), "win_rate": filtered["won"].mean(),
            "avg_odds": filtered["odds"].mean(),
            "roi": roi, "profit": profit, "bankroll": final_br,
        })
        logger.info(
            f"  {ev_thr*100:>4.0f}% {cap_label:>6} {len(filtered):>8}"
            f" {filtered['won'].mean():>7.1%} {filtered['odds'].mean():>9.2f}"
            f" {roi:>+8.1f}% {profit:>+10.2f} ${final_br:>9.2f}"
        )

    if not results:
        logger.warning("Жодної валідної комбінації")
        return

    res_df = pd.DataFrame(results).sort_values("roi", ascending=False)
    best = res_df.iloc[0]

    logger.info(f"\n{'='*65}")
    logger.info(f"НАЙКРАЩИЙ: EV≥{best['ev_thr']*100:.0f}% | Cap {best['cap_label']}")
    logger.info(f"ROI={best['roi']:+.1f}% | {int(best['bets'])} ставок | avg odds={best['avg_odds']:.2f} | ${best['bankroll']:.2f}")

    best_df = (
        cdf[cdf["ev"] >= best["ev_thr"] * 100]
        .sort_values("ev", ascending=False)
        .drop_duplicates(subset=["date", "fold"])
        .copy()
    )

    logger.info(f"\n{'Ліга':<25} {'Ставок':>7} {'Win%':>7} {'ROI':>8}")
    logger.info("-"*50)
    for name, grp in sorted(best_df.groupby("league"), key=lambda x: x[1]["won"].mean(), reverse=True):
        _, lp, lr = simulate_kelly(grp, best["kelly_cap"])
        logger.info(f"  {name:<23} {len(grp):>7} {grp['won'].mean():>7.1%} {lr:>+8.1f}%")

    logger.info(f"\nHome / Away:")
    for o in ("home", "away"):
        g = best_df[best_df["outcome"] == o]
        if g.empty: continue
        _, lp, lr = simulate_kelly(g, best["kelly_cap"])
        logger.info(f"  {o}: {len(g)} ставок | win {g['won'].mean():.1%} | ROI {lr:+.1f}%")

    # Порівняння
    orig_path = os.path.join(os.path.dirname(__file__), "results", "backtest_expanding.csv")
    if os.path.exists(orig_path):
        orig = pd.read_csv(orig_path)
        orig_roi = orig["profit"].sum() / orig["stake"].sum() * 100
        logger.info(f"\n{'='*65}")
        logger.info("ПОРІВНЯННЯ")
        logger.info(f"{'='*65}")
        logger.info(f"  Оригінал (WS filter, cap 4%): {len(orig)} ставок | win {orig['won'].mean():.1%} | ROI {orig_roi:+.1f}% | ${orig['bankroll'].iloc[-1]:.2f}")
        logger.info(f"  XGBoost+WS feats (best):      {int(best['bets'])} ставок | win {best['win_rate']:.1%} | ROI {best['roi']:+.1f}% | ${best['bankroll']:.2f}")

    out = os.path.join(os.path.dirname(__file__), "results", "backtest_xgb_ws_features.csv")
    res_df.to_csv(out, index=False)
    logger.info(f"\nЗбережено: {out}")


if __name__ == "__main__":
    run()
