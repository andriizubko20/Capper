"""
experiments/ablation_runner.py

Фаза 2: швидкий ablation на feature_store.csv.

Тестує тисячі конфігурацій без ретренінгу:
  - Групова абляція (яку групу факторів прибрати)
  - Вагове масштабування груп (0.5× / 1× / 2× / 3×)
  - Пороги WS і EV

WS = sum(group_weight × factor_lift × factor_value)
Фільтр: WS >= ws_min AND EV >= ev_min
Один матч — одна ставка (найвищий EV).
Kelly betting з model_prob.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from itertools import product

INITIAL_BANKROLL = 1000.0
MAX_STAKE_PCT    = 0.04
FRACTIONAL_KELLY = 0.25
MIN_BETS         = 30

# Ваги факторів з factor_analysis (lift - 1) × 100
FACTOR_LIFTS_HOME = {
    "elo_gap_large": 73.5, "elo_gap_moderate": 69.6, "elo_win_prob_high": 67.5,
    "home_elo_strong": 68.1, "away_elo_weak": 28.4,
    "home_in_form": 47.7, "home_strong_form": 50.8, "away_out_of_form": 24.1,
    "away_poor_form": 25.0, "home_home_form": 37.4, "home_home_wins": 43.6,
    "away_away_poor": 24.5, "away_away_loses": 0,
    "xg_ratio_home": 74.1, "xg_diff_positive": 76.6, "xg_attack_edge": 15.7,
    "home_xg_regression": 0, "away_xg_overperforming": 0,  # шкідливий
    "home_scoring_strong": 47.3, "away_conceding_lots": 21.2, "home_defense_solid": 25.5,
    "table_home_higher": 66.1, "table_points_home_better": 68.6,
    "home_rested": 0, "away_tired": 0, "rest_advantage": 0,
    "injury_advantage": 0, "big_injury_advantage": 0,
}

FACTOR_LIFTS_AWAY = {
    "elo_gap_away_large": 107.0, "elo_gap_away_moderate": 98.7, "elo_win_prob_low": 100.4,
    "away_elo_strong": 87.6, "home_elo_weak": 57.5,
    "away_in_form": 45.5, "away_strong_form": 51.5, "home_out_of_form": 40.8,
    "home_poor_form": 34.1, "away_away_form": 46.0, "away_away_wins": 41.0,
    "home_home_poor": 37.4, "home_home_loses": 0,
    "xg_ratio_away": 90.2, "xg_diff_away_positive": 97.8, "xg_away_attack": 20.1,
    "away_xg_regression": 0,  # шкідливий
    "home_xg_overperforming": 0,  # шкідливий
    "away_scoring_strong": 57.8, "home_conceding_lots": 19.3, "away_defense_solid": 31.1,
    "table_away_higher": 81.2, "table_points_away_better": 101.1,
    "away_rested": 0, "home_tired": 0, "rest_advantage_away": 0,
    "injury_adv_away": 10.5, "big_injury_adv_away": 0,
}

FACTOR_GROUPS = {
    "elo":    ["elo_gap_large", "elo_gap_moderate", "elo_win_prob_high", "home_elo_strong", "away_elo_weak",
               "elo_gap_away_large", "elo_gap_away_moderate", "elo_win_prob_low", "away_elo_strong", "home_elo_weak"],
    "form":   ["home_in_form", "home_strong_form", "away_out_of_form", "away_poor_form",
               "home_home_form", "home_home_wins", "away_away_poor", "away_away_loses",
               "away_in_form", "away_strong_form", "home_out_of_form", "home_poor_form",
               "away_away_form", "away_away_wins", "home_home_poor", "home_home_loses"],
    "xg":     ["xg_ratio_home", "xg_diff_positive", "xg_attack_edge",
               "home_scoring_strong", "away_conceding_lots", "home_defense_solid",
               "xg_ratio_away", "xg_diff_away_positive", "xg_away_attack",
               "away_scoring_strong", "home_conceding_lots", "away_defense_solid"],
    "table":  ["table_home_higher", "table_points_home_better",
               "table_away_higher", "table_points_away_better"],
    "rest":   ["home_rested", "away_tired", "rest_advantage",
               "away_rested", "home_tired", "rest_advantage_away"],
    "injury": ["injury_advantage", "big_injury_advantage",
               "injury_adv_away", "big_injury_adv_away"],
}

WS_THRESHOLDS = [0, 30, 50, 75, 100, 125, 150]
EV_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20]
GROUP_SCALES  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def compute_ws(df: pd.DataFrame, group_weights: dict[str, float], outcome: str) -> pd.Series:
    """Рахує WS для всіх рядків з заданими group_weights."""
    lifts = FACTOR_LIFTS_HOME if outcome == "home" else FACTOR_LIFTS_AWAY
    ws = pd.Series(0.0, index=df.index)
    for group, scale in group_weights.items():
        if scale == 0:
            continue
        for fname in FACTOR_GROUPS.get(group, []):
            col = f"f_{outcome}_{fname}"
            if col not in df.columns:
                continue
            lift = lifts.get(fname, 0)
            if lift > 0:
                ws += df[col] * lift * scale
    return ws


def simulate_kelly(df: pd.DataFrame) -> tuple[float, float, float]:
    bankroll = INITIAL_BANKROLL
    staked   = 0.0
    for _, row in df.sort_values("date").iterrows():
        p    = row["model_prob"]
        odds = row["odds"]
        b    = odds - 1
        kelly = max(0.0, (p * b - (1 - p)) / b) * FRACTIONAL_KELLY
        stake = min(bankroll * kelly, bankroll * MAX_STAKE_PCT)
        if stake <= 0 or bankroll <= 0:
            continue
        bankroll += stake * b if row["won"] else -stake
        staked   += stake
    profit = bankroll - INITIAL_BANKROLL
    roi    = profit / staked * 100 if staked > 0 else 0.0
    return round(bankroll, 2), round(profit, 2), round(roi, 2)


def run_config(store: pd.DataFrame, group_weights: dict, ws_min: float, ev_min: float) -> dict | None:
    """Одна конфігурація → результат."""
    # Рахуємо WS окремо для home і away
    home_mask = store["outcome"] == "home"
    away_mask = store["outcome"] == "away"

    ws = pd.Series(0.0, index=store.index)
    if home_mask.any():
        ws[home_mask] = compute_ws(store[home_mask], group_weights, "home")
    if away_mask.any():
        ws[away_mask] = compute_ws(store[away_mask], group_weights, "away")

    filtered = store[(ws >= ws_min) & (store["ev"] >= ev_min * 100)].copy()
    filtered = (filtered
                .assign(ws_val=ws[filtered.index])
                .sort_values("ev", ascending=False)
                .drop_duplicates(subset=["date", "fold"]))

    if len(filtered) < MIN_BETS:
        return None

    final_br, profit, roi = simulate_kelly(filtered)
    return {
        "bets":     len(filtered),
        "win_rate": round(filtered["won"].mean() * 100, 1),
        "avg_odds": round(filtered["odds"].mean(), 2),
        "roi":      roi,
        "profit":   profit,
        "bankroll": final_br,
    }


def run():
    store_path = os.path.join(os.path.dirname(__file__), "results", "feature_store.csv")
    if not os.path.exists(store_path):
        print("❌ feature_store.csv не знайдено. Спочатку запусти build_feature_store.py")
        return

    store = pd.read_csv(store_path)
    store["date"] = pd.to_datetime(store["date"])
    leagues = sorted(store["league"].unique())

    factor_cols = [c for c in store.columns if c.startswith("f_")]
    print(f"Feature store: {len(store)} записів | {len(factor_cols)} факторів | {len(leagues)} ліг")

    # ─── 1. Baseline (всі групи × 1.0) ───────────────────────────────────────
    baseline_weights = {g: 1.0 for g in FACTOR_GROUPS}
    print("\n" + "="*70)
    print("1. BASELINE + GRID (всі групи × 1.0, різні пороги)")
    print("="*70)
    print(f"{'WS≥':>6} {'EV≥':>5} {'Ставок':>8} {'Win%':>7} {'Avg odds':>9} {'ROI':>8} {'P&L':>10}")
    print("-"*60)
    baseline_best = {"roi": -999}
    for ws_min, ev_min in product(WS_THRESHOLDS, EV_THRESHOLDS):
        r = run_config(store, baseline_weights, ws_min, ev_min)
        if r is None:
            continue
        print(f"  {ws_min:>5} {ev_min*100:>5.0f}% {r['bets']:>8} {r['win_rate']:>7.1f}% {r['avg_odds']:>9.2f} {r['roi']:>+8.1f}% {r['profit']:>+10.2f}")
        if r["roi"] > baseline_best["roi"]:
            baseline_best = {**r, "ws_min": ws_min, "ev_min": ev_min}

    print(f"\n  ★ Baseline best: WS≥{baseline_best['ws_min']} EV≥{baseline_best['ev_min']*100:.0f}% → ROI {baseline_best['roi']:+.1f}% ({baseline_best['bets']} ставок)")

    # ─── 2. Групова абляція (прибираємо одну групу) ───────────────────────────
    print("\n" + "="*70)
    print("2. ГРУПОВА АБЛЯЦІЯ (одна група = 0, решта × 1.0)")
    print("="*70)
    ws_b, ev_b = baseline_best["ws_min"], baseline_best["ev_min"]
    ablation_results = []
    for removed_group in FACTOR_GROUPS:
        weights = {g: (0.0 if g == removed_group else 1.0) for g in FACTOR_GROUPS}
        r = run_config(store, weights, ws_b, ev_b)
        if r is None:
            print(f"  -{removed_group:<8}: замало ставок")
            continue
        delta = r["roi"] - baseline_best["roi"]
        marker = "✅" if delta > 0 else ("❌" if delta < -2 else "⬜")
        print(f"  {marker} -{removed_group:<10}: ROI {r['roi']:>+7.1f}% (Δ{delta:>+6.1f}%) | {r['bets']} ставок | {r['avg_odds']:.2f} avg odds")
        ablation_results.append({"group": removed_group, "roi": r["roi"], "delta": delta})

    # ─── 3. Вагове масштабування кожної групи ────────────────────────────────
    print("\n" + "="*70)
    print("3. ВАГОВЕ МАСШТАБУВАННЯ ГРУП (одна група, решта × 1.0)")
    print("="*70)
    for group in FACTOR_GROUPS:
        print(f"\n  Група '{group}':")
        for scale in GROUP_SCALES:
            weights = {g: (scale if g == group else 1.0) for g in FACTOR_GROUPS}
            r = run_config(store, weights, ws_b, ev_b)
            if r is None:
                continue
            delta = r["roi"] - baseline_best["roi"]
            marker = "★" if delta > 1 else ("▼" if delta < -1 else " ")
            print(f"    {marker} ×{scale:.1f}: ROI {r['roi']:>+7.1f}% (Δ{delta:>+6.1f}%) | {r['bets']} ставок")

    # ─── 4. Комбінований grid найкращих груп ─────────────────────────────────
    print("\n" + "="*70)
    print("4. КОМБІНОВАНИЙ GRID (top групи × різні ваги)")
    print("="*70)

    # Визначаємо групи що допомагають і шкодять
    helpful = [a["group"] for a in ablation_results if a["delta"] > 0]
    harmful = [a["group"] for a in ablation_results if a["delta"] < -2]
    print(f"  Корисні групи: {helpful}")
    print(f"  Шкідливі групи: {harmful}")

    best_combo = {"roi": baseline_best["roi"], "label": "baseline"}
    combo_results = []

    scales_to_try = [0.0, 0.5, 1.0, 2.0, 3.0]
    # Тестуємо: harmful=0, helpful варіює, решта=1.0
    for helpful_scale in scales_to_try:
        weights = {}
        for g in FACTOR_GROUPS:
            if g in harmful:
                weights[g] = 0.0
            elif g in helpful:
                weights[g] = helpful_scale
            else:
                weights[g] = 1.0

        for ws_min, ev_min in product([ws_b, max(0, ws_b - 25), ws_b + 25], [ev_b, max(0.05, ev_b - 0.03), ev_b + 0.03]):
            r = run_config(store, weights, ws_min, ev_min)
            if r is None:
                continue
            label = f"helpful×{helpful_scale:.1f} harmful=0 WS≥{ws_min} EV≥{ev_min*100:.0f}%"
            combo_results.append({**r, "label": label, "weights": weights, "ws_min": ws_min, "ev_min": ev_min})
            if r["roi"] > best_combo["roi"]:
                best_combo = {**r, "label": label, "weights": weights, "ws_min": ws_min, "ev_min": ev_min}

    combo_results.sort(key=lambda x: x["roi"], reverse=True)
    print(f"\n  Топ-10 комбінацій:")
    print(f"  {'Конфігурація':<50} {'Ставок':>8} {'Win%':>7} {'ROI':>8}")
    print("  " + "-"*80)
    for r in combo_results[:10]:
        print(f"  {r['label']:<50} {r['bets']:>8} {r['win_rate']:>7.1f}% {r['roi']:>+8.1f}%")

    # ─── 5. Підсумок ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"  Baseline:      ROI {baseline_best['roi']:>+7.1f}% | {baseline_best['bets']} ставок")
    print(f"  Найкраще:      ROI {best_combo['roi']:>+7.1f}% | {best_combo.get('bets', '?')} ставок")
    print(f"  Конфігурація:  {best_combo['label']}")

    # Зберігаємо всі результати
    all_results = []
    for ws_min, ev_min in product(WS_THRESHOLDS, EV_THRESHOLDS):
        for gw_combo in combo_results[:20]:
            all_results.append({**gw_combo, "ws_min": ws_min, "ev_min": ev_min})

    out = os.path.join(os.path.dirname(__file__), "results", "ablation_results.csv")
    if combo_results:
        pd.DataFrame(combo_results).drop(columns=["weights"], errors="ignore").to_csv(out, index=False)
        print(f"\n  Збережено: {out}")


if __name__ == "__main__":
    run()
