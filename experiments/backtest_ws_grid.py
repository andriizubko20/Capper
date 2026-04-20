"""
experiments/backtest_ws_grid.py

WS Gap модель — 2-fold OOS бектест з:
  1. Factor lift-аналіз: які фактори реально допомагають
  2. Threshold grid: WS_GAP × WS_DOM × ODDS_MIN × ODDS_MAX
  3. Weight group sweep: множники для груп факторів

Запуск: python -m experiments.backtest_ws_grid
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import pandas as pd
import numpy as np
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

SPLIT_1 = pd.Timestamp("2025-01-01")
SPLIT_2 = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0
FRACTIONAL = 0.25
MAX_STAKE   = 0.04

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}

# ─── Грід параметрів ──────────────────────────────────────────────────────────
WS_GAP_VALS  = [50, 60, 70, 80, 90, 100]
WS_DOM_VALS  = [65, 70, 75, 80, 85]
ODDS_MIN_VALS = [1.5, 1.7, 2.0]
ODDS_MAX_VALS = [2.5, 3.0, 4.0, 99.0]

# ─── Групи факторів для weight sweep ─────────────────────────────────────────
FACTOR_GROUPS = {
    "market":   {"market_favors_home", "market_strong_home", "market_sees_away", "market_strong_away"},
    "elo":      {"elo_gap_large", "elo_gap_moderate", "elo_win_prob_high", "home_elo_strong",
                 "away_elo_weak", "elo_gap_away_large", "elo_gap_away_moderate",
                 "elo_win_prob_low", "away_elo_strong", "home_elo_weak"},
    "form":     {"home_in_form", "home_strong_form", "away_out_of_form", "away_poor_form",
                 "home_home_form", "home_home_wins", "away_away_poor", "away_away_loses",
                 "home_out_of_form", "home_poor_form", "home_home_poor", "home_home_loses",
                 "away_in_form", "away_strong_form", "away_away_form", "away_away_wins"},
    "xg":       {"xg_ratio_home", "xg_diff_positive", "xg_attack_edge", "home_xg_regression",
                 "away_xg_overperforming", "xg_ratio_away", "xg_diff_away_positive",
                 "xg_away_attack", "away_xg_regression", "home_xg_overperforming"},
    "table":    {"table_home_higher", "table_points_home_better",
                 "table_away_higher", "table_points_away_better"},
    "new":      {"home_win_streak_3", "away_loss_streak_3", "home_clean_sheet_strong",
                 "away_failed_to_score", "home_elo_rising", "away_elo_falling",
                 "away_win_streak_3", "home_loss_streak_3", "away_clean_sheet_strong",
                 "home_failed_to_score", "away_elo_rising", "home_elo_falling"},
}

# Weight sweep: множники по групах (кожен варіант — dict group→multiplier)
WEIGHT_VARIANTS = {
    "baseline":      {g: 1.0 for g in FACTOR_GROUPS},
    "boost_new":     {**{g: 1.0 for g in FACTOR_GROUPS}, "new": 2.0},
    "boost_elo":     {**{g: 1.0 for g in FACTOR_GROUPS}, "elo": 1.5, "new": 1.5},
    "no_form":       {**{g: 1.0 for g in FACTOR_GROUPS}, "form": 0.0},
    "no_xg":         {**{g: 1.0 for g in FACTOR_GROUPS}, "xg": 0.0},
    "elo_only":      {g: 0.0 for g in FACTOR_GROUPS} | {"market": 1.0, "elo": 1.0, "new": 1.0, "table": 1.0},
}


def make_weights(multipliers: dict) -> tuple[dict, dict]:
    """Повертає (home_w, away_w) з застосованими множниками по групах."""
    def group_of(factor_name):
        for g, members in FACTOR_GROUPS.items():
            if factor_name in members:
                return g
        return None

    hw = {}
    for name, base in HOME_WEIGHTS.items():
        g = group_of(name)
        mult = multipliers.get(g, 1.0) if g else 1.0
        hw[name] = base * mult

    aw = {}
    for name, base in AWAY_WEIGHTS.items():
        g = group_of(name)
        mult = multipliers.get(g, 1.0) if g else 1.0
        aw[name] = base * mult

    return hw, aw


def compute_ws(features: dict, outcome: str, hw: dict, aw: dict) -> float:
    weights = hw if outcome == "home" else aw
    total = 0.0
    for name, active in _get_factors(features, outcome):
        if active:
            total += weights.get(name, 1)
    return total


def collect_bets(dataset: pd.DataFrame, ws_gap_min, ws_dom_min,
                 odds_min, odds_max, hw, aw) -> pd.DataFrame:
    records = []
    for _, row in dataset.iterrows():
        h_odds = row.get("home_odds")
        a_odds = row.get("away_odds")
        if not h_odds or not a_odds:
            continue

        feats = row.to_dict()
        ws_h = compute_ws(feats, "home", hw, aw)
        ws_a = compute_ws(feats, "away", hw, aw)

        if ws_h >= ws_a:
            side, ws_dom, ws_weak, odds = "home", ws_h, ws_a, h_odds
            p_elo = float(row.get("elo_home_win_prob", 0.5))
        else:
            side, ws_dom, ws_weak, odds = "away", ws_a, ws_h, a_odds
            p_elo = 1.0 - float(row.get("elo_home_win_prob", 0.5))

        ws_gap = ws_dom - ws_weak
        if ws_gap < ws_gap_min or ws_dom < ws_dom_min:
            continue
        if not (odds_min <= odds <= odds_max):
            continue

        records.append({
            "date":    row["date"],
            "league":  row.get("league", ""),
            "side":    side,
            "won":     row["target"] == side,
            "odds":    round(odds, 2),
            "ws_dom":  ws_dom,
            "ws_gap":  ws_gap,
            "p_elo":   p_elo,
        })
    return pd.DataFrame(records)


def simulate(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n": 0, "wins": 0, "win_rate": 0, "flat_roi": 0, "kelly_roi": 0, "bankroll": INITIAL_BR}
    bets = bets.sort_values("date")
    bankroll = INITIAL_BR
    staked = 0.0
    flat_pnl = 0.0
    for _, b in bets.iterrows():
        p, odd = b["p_elo"], b["odds"]
        q = 1 - p
        kelly = max(0.0, (p * (odd - 1) - q) / (odd - 1)) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * MAX_STAKE)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked += stake
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n":        n,
        "wins":     int(bets["won"].sum()),
        "win_rate": bets["won"].mean(),
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0,
        "bankroll": round(bankroll, 2),
    }


def show_by_league(bets: pd.DataFrame):
    if bets.empty or "league" not in bets.columns:
        return
    logger.info(f"\n  {'Ліга':<23} {'N':>5} {'WR%':>7} {'FlatROI':>9}")
    logger.info(f"  {'-'*48}")
    by_league = bets.groupby("league").apply(
        lambda g: pd.Series({
            "n": len(g),
            "wr": g["won"].mean(),
            "roi": ((g["won"] * (g["odds"] - 1)) - (~g["won"])).sum() / len(g) * 100,
        })
    ).sort_values("roi", ascending=False)
    for league, row in by_league.iterrows():
        logger.info(f"  {league:<23} {int(row['n']):>5} {row['wr']:>7.1%} {row['roi']:>+9.1f}%")


def factor_lift_analysis(dataset: pd.DataFrame):
    """Для кожного фактору: % спрацювань, win rate та flat ROI коли спрацював."""
    hw, aw = make_weights({g: 1.0 for g in FACTOR_GROUPS})
    h_factor_names = [name for name, _ in _get_factors({}, "home")]
    a_factor_names = [name for name, _ in _get_factors({}, "away")]

    base_wr = dataset["target"].isin(["home", "away"]).mean()  # base (без нічиїх не рахуємо)

    results = []
    for outcome, factor_names in [("home", h_factor_names), ("away", a_factor_names)]:
        for fname in factor_names:
            fired_wins = 0
            fired_total = 0
            flat_pnl = 0.0
            for _, row in dataset.iterrows():
                feats = row.to_dict()
                factors = dict(_get_factors(feats, outcome))
                if not factors.get(fname, False):
                    continue
                h_odds = row.get("home_odds")
                a_odds = row.get("away_odds")
                odds = h_odds if outcome == "home" else a_odds
                if not odds:
                    continue
                fired_total += 1
                won = row["target"] == outcome
                if won:
                    fired_wins += 1
                    flat_pnl += odds - 1
                else:
                    flat_pnl -= 1.0

            if fired_total < 10:
                continue
            wr = fired_wins / fired_total
            roi = flat_pnl / fired_total * 100
            results.append({
                "factor": fname,
                "outcome": outcome,
                "n": fired_total,
                "win_rate": wr,
                "flat_roi": roi,
            })

    df = pd.DataFrame(results).sort_values("flat_roi", ascending=False)
    print("\n" + "="*72)
    print(f"FACTOR LIFT ANALYSIS (топ позитивні та негативні)")
    print("="*72)
    print(f"  {'Фактор':<42} {'Side':>5} {'N':>6} {'WR%':>7} {'ROI%':>9}")
    print(f"  {'-'*70}")
    top = pd.concat([df.head(15), df.tail(10)]).drop_duplicates()
    for _, row in top.iterrows():
        marker = "+" if row["flat_roi"] > 0 else "-"
        print(f"  {marker} {row['factor']:<41} {row['outcome']:>5} {int(row['n']):>6} "
              f"{row['win_rate']:>7.1%} {row['flat_roi']:>+9.1f}%")

    out = "experiments/results/ws_factor_lift.csv"
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Factor lift збережено: {out}")
    return df


def run():
    from db.session import SessionLocal
    from db.models import League

    db = SessionLocal()
    try:
        league_map = {
            l.id: LEAGUES[l.api_id]
            for l in db.query(League).all()
            if l.api_id in LEAGUES
        }
    finally:
        db.close()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Будую датасет...")
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset[dataset["league_id"].isin(league_map)].copy()
    dataset = dataset[dataset["home_odds"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)

    logger.info(f"Датасет: {len(dataset)} матчів з odds")

    fold1 = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    fold2 = dataset[dataset["date"] >= SPLIT_2].copy()
    fold_all = dataset.copy()

    # ─── 1. Factor lift ───────────────────────────────────────────────────────
    logger.info("\nРахую factor lift...")
    factor_lift_analysis(fold_all)

    # ─── 2. Threshold grid (baseline weights) ─────────────────────────────────
    logger.info("\nThreshold grid (baseline weights)...")
    hw_base, aw_base = make_weights({g: 1.0 for g in FACTOR_GROUPS})

    grid_results = []
    total = len(WS_GAP_VALS) * len(WS_DOM_VALS) * len(ODDS_MIN_VALS) * len(ODDS_MAX_VALS)
    done = 0
    for gap, dom, omin, omax in itertools.product(WS_GAP_VALS, WS_DOM_VALS, ODDS_MIN_VALS, ODDS_MAX_VALS):
        if omin >= omax:
            continue
        b1 = collect_bets(fold1, gap, dom, omin, omax, hw_base, aw_base)
        b2 = collect_bets(fold2, gap, dom, omin, omax, hw_base, aw_base)
        r1, r2 = simulate(b1), simulate(b2)
        grid_results.append({
            "ws_gap": gap, "ws_dom": dom, "odds_min": omin, "odds_max": omax,
            "f1_n": r1["n"], "f1_wr": r1["win_rate"], "f1_roi": r1["flat_roi"],
            "f2_n": r2["n"], "f2_wr": r2["win_rate"], "f2_roi": r2["flat_roi"],
            "avg_roi": (r1["flat_roi"] + r2["flat_roi"]) / 2,
        })
        done += 1
        if done % 50 == 0:
            logger.info(f"  {done}/{total}...")

    grid_df = pd.DataFrame(grid_results).sort_values("avg_roi", ascending=False)

    print("\n" + "="*80)
    print("THRESHOLD GRID — TOP 20 (baseline weights)")
    print("="*80)
    print(f"  {'Gap':>5} {'Dom':>5} {'OMin':>6} {'OMax':>6} | "
          f"{'F1 N':>6} {'F1 WR':>7} {'F1 ROI':>8} | "
          f"{'F2 N':>6} {'F2 WR':>7} {'F2 ROI':>8} | {'AvgROI':>8}")
    print(f"  {'-'*80}")
    for _, r in grid_df.head(20).iterrows():
        print(f"  {int(r.ws_gap):>5} {int(r.ws_dom):>5} {r.odds_min:>6.1f} {r.odds_max:>6.1f} | "
              f"{int(r.f1_n):>6} {r.f1_wr:>7.1%} {r.f1_roi:>+8.1f}% | "
              f"{int(r.f2_n):>6} {r.f2_wr:>7.1%} {r.f2_roi:>+8.1f}% | {r.avg_roi:>+8.1f}%")

    # ─── 3. Weight group sweep (на найкращих threshold) ───────────────────────
    best = grid_df.iloc[0]
    gap_best   = int(best["ws_gap"])
    dom_best   = int(best["ws_dom"])
    omin_best  = best["odds_min"]
    omax_best  = best["odds_max"]
    logger.info(f"\nWeight sweep на найкращому threshold: GAP={gap_best} DOM={dom_best} "
                f"ODDS={omin_best}-{omax_best}")

    print("\n" + "="*80)
    print(f"WEIGHT VARIANTS — GAP={gap_best} DOM={dom_best} ODDS={omin_best}-{omax_best}")
    print("="*80)
    print(f"  {'Варіант':<18} | {'F1 N':>6} {'F1 WR':>7} {'F1 ROI':>8} | "
          f"{'F2 N':>6} {'F2 WR':>7} {'F2 ROI':>8} | {'AvgROI':>8}")
    print(f"  {'-'*75}")

    weight_results = []
    for vname, multipliers in WEIGHT_VARIANTS.items():
        hw, aw = make_weights(multipliers)
        b1 = collect_bets(fold1, gap_best, dom_best, omin_best, omax_best, hw, aw)
        b2 = collect_bets(fold2, gap_best, dom_best, omin_best, omax_best, hw, aw)
        r1, r2 = simulate(b1), simulate(b2)
        avg = (r1["flat_roi"] + r2["flat_roi"]) / 2
        weight_results.append({"variant": vname, **r1, **{f"f2_{k}": v for k, v in r2.items()}, "avg_roi": avg})
        print(f"  {vname:<18} | {int(r1['n']):>6} {r1['win_rate']:>7.1%} {r1['flat_roi']:>+8.1f}% | "
              f"{int(r2['n']):>6} {r2['win_rate']:>7.1%} {r2['flat_roi']:>+8.1f}% | {avg:>+8.1f}%")

    # ─── Best overall ─────────────────────────────────────────────────────────
    best_row = grid_df.iloc[0]
    hw, aw = make_weights({g: 1.0 for g in FACTOR_GROUPS})
    b1 = collect_bets(fold1, gap_best, dom_best, omin_best, omax_best, hw, aw)
    b2 = collect_bets(fold2, gap_best, dom_best, omin_best, omax_best, hw, aw)

    logger.info(f"\n{'='*65}\n  BEST CONFIG: GAP={gap_best} DOM={dom_best} ODDS={omin_best}-{omax_best}\n{'='*65}")
    for label, bets in [("FOLD 1 (Jan–Jun 2025)", b1), ("FOLD 2 (Jul 2025+)", b2)]:
        r = simulate(bets)
        logger.info(f"\n  {label}: {r['n']} ставок | WR {r['win_rate']:.1%} | Flat ROI {r['flat_roi']:+.1f}%")
        show_by_league(bets)

    # ─── Збереження ───────────────────────────────────────────────────────────
    os.makedirs("experiments/results", exist_ok=True)
    grid_df.to_csv("experiments/results/ws_threshold_grid.csv", index=False)
    logger.info("\nЗбережено: experiments/results/ws_threshold_grid.csv")


if __name__ == "__main__":
    run()
