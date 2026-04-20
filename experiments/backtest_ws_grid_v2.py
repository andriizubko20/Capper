"""
experiments/backtest_ws_grid_v2.py

Розширений WS Gap грід:
  - Без DOM (прибрано)
  - WS_GAP × ODDS_MIN × ODDS_MAX × EV_MIN × KELLY_CAP × FRACTIONAL
  - Tiered WS (багаторівнева система порогів)
  - Home vs Away split
  - Ліги як фільтр (з/без Ligue 1)
  - EV filter через Elo probability

Запуск: python -m experiments.backtest_ws_grid_v2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}

LEAGUES_NO_L1 = {k: v for k, v in LEAGUES.items() if k != 61}

# ─── Грід параметрів ──────────────────────────────────────────────────────────
WS_GAP_VALS   = [40, 50, 60, 70, 80, 90, 100, 120]
ODDS_MIN_VALS = [1.5, 1.65, 1.75, 2.0, 2.2]
ODDS_MAX_VALS = [2.5, 2.8, 3.0, 3.5, 4.0]
EV_MIN_VALS   = [0.0, 0.03, 0.05, 0.08, 0.10]   # p_elo * odds - 1 >= ev_min
KELLY_CAP_VALS    = [0.02, 0.03, 0.04, 0.05]
FRACTIONAL_VALS   = [0.15, 0.25, 0.33]

# ─── Tiered WS конфіги ────────────────────────────────────────────────────────
# Список рівнів: (min_gap, odds_min, odds_max). Беремо перший рівень що підходить.
TIER_CONFIGS = {
    "flat_70_2.0-2.5":  [(70,  2.0, 2.5)],
    "flat_80_1.9-2.8":  [(80,  1.9, 2.8)],
    "tiered_A": [
        (100, 1.65, 3.5),
        (70,  1.90, 2.8),
        (50,  2.10, 2.5),
    ],
    "tiered_B": [
        (90,  1.75, 3.0),
        (70,  2.00, 2.5),
    ],
    "tiered_C": [
        (100, 1.65, 3.0),
        (80,  1.80, 2.8),
        (60,  2.00, 2.5),
    ],
    "tiered_D": [
        (90,  1.70, 3.5),
        (70,  1.90, 2.8),
        (50,  2.20, 2.5),
    ],
    "wide_60_1.65-3.5": [(60, 1.65, 3.5)],
    "wide_70_1.75-3.0": [(70, 1.75, 3.0)],
}


def compute_ws(features: dict, outcome: str) -> float:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    total = 0.0
    for name, active in _get_factors(features, outcome):
        if active:
            total += weights.get(name, 1)
    return total


def pick_for_row(row: dict, ws_gap_min: float, odds_min: float, odds_max: float,
                 ev_min: float = 0.0) -> dict | None:
    h_odds = row.get("home_odds")
    a_odds = row.get("away_odds")
    if not h_odds or not a_odds:
        return None

    ws_h = compute_ws(row, "home")
    ws_a = compute_ws(row, "away")

    if ws_h >= ws_a:
        side, ws_dom, ws_weak, odds = "home", ws_h, ws_a, h_odds
        p_elo = float(row.get("elo_home_win_prob", 0.5))
    else:
        side, ws_dom, ws_weak, odds = "away", ws_a, ws_h, a_odds
        p_elo = 1.0 - float(row.get("elo_home_win_prob", 0.5))

    ws_gap = ws_dom - ws_weak
    if ws_gap < ws_gap_min:
        return None
    if not (odds_min <= odds <= odds_max):
        return None
    ev = p_elo * odds - 1.0
    if ev < ev_min:
        return None

    return {"side": side, "odds": round(odds, 2), "ws_gap": ws_gap,
            "ws_dom": ws_dom, "p_elo": p_elo, "ev": ev,
            "won": row["target"] == side,
            "date": row["date"], "league": row.get("league", "")}


def pick_tiered(row: dict, tiers: list, ev_min: float = 0.0) -> dict | None:
    """Тестує рівні від найвищого GAP до найнижчого, бере перший що підходить."""
    for gap_min, omin, omax in sorted(tiers, key=lambda x: -x[0]):
        p = pick_for_row(row, gap_min, omin, omax, ev_min)
        if p:
            return p
    return None


def collect_bets_flat(dataset: pd.DataFrame, ws_gap_min, odds_min, odds_max,
                      ev_min=0.0) -> pd.DataFrame:
    rows = [r for _, row in dataset.iterrows()
            if (r := pick_for_row(row.to_dict(), ws_gap_min, odds_min, odds_max, ev_min))]
    return pd.DataFrame(rows)


def collect_bets_tiered(dataset: pd.DataFrame, tiers: list, ev_min=0.0) -> pd.DataFrame:
    rows = [r for _, row in dataset.iterrows()
            if (r := pick_tiered(row.to_dict(), tiers, ev_min))]
    return pd.DataFrame(rows)


def simulate(bets: pd.DataFrame, kelly_cap: float = 0.04,
             fractional: float = 0.25) -> dict:
    if bets.empty:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "flat_roi": 0.0,
                "kelly_roi": 0.0, "bankroll": INITIAL_BR}
    bets = bets.sort_values("date")
    bankroll = INITIAL_BR
    staked = 0.0
    flat_pnl = 0.0
    for _, b in bets.iterrows():
        p, odd = b["p_elo"], b["odds"]
        q = 1.0 - p
        kelly = max(0.0, (p * (odd - 1) - q) / (odd - 1)) * fractional
        stake = min(bankroll * kelly, bankroll * kelly_cap)
        if stake > 0 and bankroll > 0:
            bankroll += stake * (odd - 1) if b["won"] else -stake
            staked += stake
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wins": int(bets["won"].sum()),
        "win_rate": bets["won"].mean(),
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0,
        "bankroll": round(bankroll, 2),
    }


def show_by_league(bets: pd.DataFrame, label: str = ""):
    if bets.empty:
        return
    if label:
        logger.info(f"\n  {label}")
    logger.info(f"  {'Ліга':<23} {'N':>5} {'WR%':>7} {'FlatROI':>9} {'KellyROI':>10}")
    logger.info(f"  {'-'*56}")
    for league, g in bets.groupby("league"):
        n = len(g)
        wr = g["won"].mean()
        roi = ((g["won"] * (g["odds"] - 1)) - (~g["won"])).sum() / n * 100
        logger.info(f"  {league:<23} {n:>5} {wr:>7.1%} {roi:>+9.1f}%")


def print_grid_table(results: list[dict], title: str, top_n: int = 25):
    df = pd.DataFrame(results).sort_values("avg_roi", ascending=False)
    print(f"\n{'='*92}")
    print(f"  {title}")
    print(f"{'='*92}")
    print(f"  {'Gap':>5} {'OMin':>6} {'OMax':>6} {'EV':>5} | "
          f"{'F1 N':>5} {'F1 WR':>6} {'F1 ROI':>7} | "
          f"{'F2 N':>5} {'F2 WR':>6} {'F2 ROI':>7} | {'Avg':>7}")
    print(f"  {'-'*88}")
    for _, r in df.head(top_n).iterrows():
        ev_str = f"{r.get('ev_min', 0):.2f}"
        print(f"  {int(r['ws_gap']):>5} {r['odds_min']:>6.2f} {r['odds_max']:>6.1f} {ev_str:>5} | "
              f"{int(r['f1_n']):>5} {r['f1_wr']:>6.1%} {r['f1_roi']:>+7.1f}% | "
              f"{int(r['f2_n']):>5} {r['f2_wr']:>6.1%} {r['f2_roi']:>+7.1f}% | "
              f"{r['avg_roi']:>+7.1f}%")
    return df


def run():
    from db.session import SessionLocal
    from db.models import League

    db = SessionLocal()
    try:
        league_map = {l.id: LEAGUES[l.api_id] for l in db.query(League).all() if l.api_id in LEAGUES}
        league_map_no_l1 = {l.id: LEAGUES_NO_L1[l.api_id] for l in db.query(League).all() if l.api_id in LEAGUES_NO_L1}
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

    fold1     = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    fold2     = dataset[dataset["date"] >= SPLIT_2].copy()
    fold1_no  = fold1[fold1["league"] != "Ligue 1"].copy()
    fold2_no  = fold2[fold2["league"] != "Ligue 1"].copy()

    os.makedirs("experiments/results", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. THRESHOLD + EV GRID (flat, no DOM, Kelly cap=4%, frac=0.25)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[1] Threshold + EV grid...")
    grid1 = []
    combos = [(g, omin, omax, ev)
              for g, omin, omax, ev
              in itertools.product(WS_GAP_VALS, ODDS_MIN_VALS, ODDS_MAX_VALS, EV_MIN_VALS)
              if omin < omax]
    for i, (gap, omin, omax, ev) in enumerate(combos):
        b1 = collect_bets_flat(fold1, gap, omin, omax, ev)
        b2 = collect_bets_flat(fold2, gap, omin, omax, ev)
        r1, r2 = simulate(b1), simulate(b2)
        grid1.append({"ws_gap": gap, "odds_min": omin, "odds_max": omax, "ev_min": ev,
                      "f1_n": r1["n"], "f1_wr": r1["win_rate"], "f1_roi": r1["flat_roi"],
                      "f2_n": r2["n"], "f2_wr": r2["win_rate"], "f2_roi": r2["flat_roi"],
                      "avg_roi": (r1["flat_roi"] + r2["flat_roi"]) / 2})
        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(combos)}...")

    df1 = print_grid_table(grid1, "THRESHOLD + EV GRID (flat, top-25 avg ROI)")
    df1.to_csv("experiments/results/ws_v2_threshold_ev.csv", index=False)

    # Визначаємо best threshold для подальших секцій
    best = df1.iloc[0]
    gap_b, omin_b, omax_b, ev_b = int(best.ws_gap), best.odds_min, best.odds_max, best.ev_min
    logger.info(f"\nBest flat: GAP={gap_b} ODDS={omin_b}-{omax_b} EV>={ev_b:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. KELLY CAP × FRACTIONAL SWEEP (на best threshold)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[2] Kelly cap × fractional sweep...")
    b1_base = collect_bets_flat(fold1, gap_b, omin_b, omax_b, ev_b)
    b2_base = collect_bets_flat(fold2, gap_b, omin_b, omax_b, ev_b)

    print(f"\n{'='*75}")
    print(f"  KELLY SWEEP — GAP={gap_b} ODDS={omin_b}-{omax_b} EV>={ev_b:.2f}")
    print(f"{'='*75}")
    print(f"  {'Cap%':>5} {'Frac':>6} | {'F1 KRoi':>9} {'F1 BK':>9} | {'F2 KRoi':>9} {'F2 BK':>9}")
    print(f"  {'-'*60}")
    kelly_results = []
    for cap, frac in itertools.product(KELLY_CAP_VALS, FRACTIONAL_VALS):
        r1 = simulate(b1_base, kelly_cap=cap, fractional=frac)
        r2 = simulate(b2_base, kelly_cap=cap, fractional=frac)
        kelly_results.append({"cap": cap, "frac": frac,
                               "f1_kroi": r1["kelly_roi"], "f1_bk": r1["bankroll"],
                               "f2_kroi": r2["kelly_roi"], "f2_bk": r2["bankroll"],
                               "avg_kroi": (r1["kelly_roi"] + r2["kelly_roi"]) / 2})
        print(f"  {cap*100:>4.0f}% {frac:>6.2f} | "
              f"{r1['kelly_roi']:>+9.1f}% ${r1['bankroll']:>8.0f} | "
              f"{r2['kelly_roi']:>+9.1f}% ${r2['bankroll']:>8.0f}")
    pd.DataFrame(kelly_results).sort_values("avg_kroi", ascending=False).to_csv(
        "experiments/results/ws_v2_kelly_sweep.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TIERED WS CONFIGS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[3] Tiered WS configs...")
    print(f"\n{'='*85}")
    print(f"  TIERED WS — різні рівні EV")
    print(f"{'='*85}")
    print(f"  {'Config':<22} {'EV':>5} | {'F1 N':>5} {'F1 WR':>6} {'F1 ROI':>7} | "
          f"{'F2 N':>5} {'F2 WR':>6} {'F2 ROI':>7} | {'Avg':>7}")
    print(f"  {'-'*82}")
    tier_results = []
    for cname, tiers in TIER_CONFIGS.items():
        for ev in [0.0, 0.03, 0.05, 0.08]:
            b1 = collect_bets_tiered(fold1, tiers, ev)
            b2 = collect_bets_tiered(fold2, tiers, ev)
            r1, r2 = simulate(b1), simulate(b2)
            avg = (r1["flat_roi"] + r2["flat_roi"]) / 2
            tier_results.append({"config": cname, "ev_min": ev,
                                  "f1_n": r1["n"], "f1_wr": r1["win_rate"], "f1_roi": r1["flat_roi"],
                                  "f2_n": r2["n"], "f2_wr": r2["win_rate"], "f2_roi": r2["flat_roi"],
                                  "avg_roi": avg})
            print(f"  {cname:<22} {ev:>5.2f} | "
                  f"{r1['n']:>5} {r1['win_rate']:>6.1%} {r1['flat_roi']:>+7.1f}% | "
                  f"{r2['n']:>5} {r2['win_rate']:>6.1%} {r2['flat_roi']:>+7.1f}% | "
                  f"{avg:>+7.1f}%")
    pd.DataFrame(tier_results).sort_values("avg_roi", ascending=False).to_csv(
        "experiments/results/ws_v2_tiered.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. HOME vs AWAY SPLIT
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[4] Home vs Away split...")
    print(f"\n{'='*75}")
    print(f"  HOME vs AWAY — GAP={gap_b} ODDS={omin_b}-{omax_b} EV>={ev_b:.2f}")
    print(f"{'='*75}")
    for label, fold, fname in [("F1", b1_base, "Fold1"), ("F2", b2_base, "Fold2")]:
        for side in ["home", "away"]:
            sub = fold[fold["side"] == side] if not fold.empty else fold
            r = simulate(sub)
            logger.info(f"  {fname} {side.upper():>5}: {r['n']:>4} ставок | "
                        f"WR {r['win_rate']:.1%} | Flat ROI {r['flat_roi']:>+.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. З LIGUE 1 та БЕЗ
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n[5] З Ligue 1 vs без...")
    print(f"\n{'='*75}")
    print(f"  LIGUE 1 IMPACT — GAP={gap_b} ODDS={omin_b}-{omax_b} EV>={ev_b:.2f}")
    print(f"{'='*75}")
    for label, f1, f2 in [("з Ligue 1", fold1, fold2), ("без Ligue 1", fold1_no, fold2_no)]:
        b1 = collect_bets_flat(f1, gap_b, omin_b, omax_b, ev_b)
        b2 = collect_bets_flat(f2, gap_b, omin_b, omax_b, ev_b)
        r1, r2 = simulate(b1), simulate(b2)
        logger.info(f"  {label}: F1={r1['flat_roi']:>+.1f}% ({r1['n']} bets) | "
                    f"F2={r2['flat_roi']:>+.1f}% ({r2['n']} bets) | "
                    f"Avg={(r1['flat_roi']+r2['flat_roi'])/2:>+.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. BEST CONFIG — деталізований результат
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n[6] Best config деталі: GAP={gap_b} ODDS={omin_b}-{omax_b} EV>={ev_b:.2f}")
    for label, bets in [("FOLD 1", b1_base), ("FOLD 2", b2_base)]:
        r = simulate(bets)
        logger.info(f"\n{'='*60}\n  {label}: {r['n']} ставок | WR {r['win_rate']:.1%} | "
                    f"Flat ROI {r['flat_roi']:>+.1f}% | Kelly ROI {r['kelly_roi']:>+.1f}%")
        show_by_league(bets)

    logger.info("\nВсі результати збережено в experiments/results/ws_v2_*.csv")


if __name__ == "__main__":
    run()
