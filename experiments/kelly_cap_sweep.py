"""
experiments/kelly_cap_sweep.py

Порівнює різні Kelly caps на повному датасеті WS Gap (Gap≥80, Dom≥80, Odds≥1.7).
Caps: 4%, 6%, 8%, 10%, 12%, 15%, 17%, 20%, 23%, та чистий Kelly 25% (без кепу).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import compute_weighted_score

GAP_MIN   = 80
DOM_MIN   = 80
ODDS_MIN  = 1.7
FRACTIONAL = 0.25
INITIAL   = 1000.0

TOP5_API  = {39, 140, 78, 135, 61}
CAPS      = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.23, None]  # None = pure Kelly


def get_top5_league_ids():
    from db.session import SessionLocal
    from db.models import League
    db = SessionLocal()
    try:
        return {l.id for l in db.query(League).all() if l.api_id in TOP5_API}
    finally:
        db.close()


def build_bets():
    logger.info("Loading data...")
    matches, stats, odds_data, teams_df, injuries = load_data_from_db()
    dataset = build_dataset(matches, stats, odds_data, teams_df, injuries_df=injuries)

    top5_ids = get_top5_league_ids()
    dataset  = dataset[dataset["league_id"].isin(top5_ids)].copy()
    dataset  = dataset[dataset["target"].isin(["home", "away"])].copy()
    dataset  = dataset.sort_values("date").reset_index(drop=True)

    bets = []
    for _, row in dataset.iterrows():
        h_prob = row.get("market_home_prob", 0)
        a_prob = row.get("market_away_prob", 0)
        if not h_prob or not a_prob or pd.isna(h_prob) or pd.isna(a_prob):
            continue
        try:
            ws_h = compute_weighted_score(row, "home")
            ws_a = compute_weighted_score(row, "away")
        except Exception:
            continue

        elo_home = float(row.get("elo_home_win_prob", 0.5) or 0.5)
        if ws_h >= ws_a:
            dom, ws_dom, ws_wk = "home", ws_h, ws_a
            odds, p_elo = round(1 / h_prob, 2), elo_home
        else:
            dom, ws_dom, ws_wk = "away", ws_a, ws_h
            odds, p_elo = round(1 / a_prob, 2), 1.0 - elo_home

        ws_gap = ws_dom - ws_wk
        if ws_gap < GAP_MIN or ws_dom < DOM_MIN or odds < ODDS_MIN:
            continue

        b = odds - 1
        q = 1 - p_elo
        kelly_frac = max(0.0, (p_elo * b - q) / b) * FRACTIONAL

        bets.append({
            "date":      row["date"],
            "odds":      odds,
            "won":       row["target"] == dom,
            "p_elo":     p_elo,
            "kelly_frac": kelly_frac,
        })

    logger.info(f"Total bets: {len(bets)}")
    return bets


def simulate(bets: list, cap: float | None) -> dict:
    bankroll = INITIAL
    peak     = INITIAL
    max_dd   = 0.0

    for bet in bets:
        kf    = bet["kelly_frac"]
        stake = bankroll * kf if cap is None else min(bankroll * kf, bankroll * cap)
        stake = round(stake, 2)

        if bet["won"]:
            bankroll += stake * (bet["odds"] - 1)
        else:
            bankroll -= stake

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak * 100
        if dd > max_dd:
            max_dd = dd

    profit = bankroll - INITIAL
    roi    = profit / INITIAL * 100
    return {
        "final":    round(bankroll, 0),
        "profit":   round(profit, 0),
        "roi":      round(roi, 1),
        "max_dd":   round(max_dd, 1),
        "peak":     round(peak, 0),
    }


if __name__ == "__main__":
    bets = build_bets()
    n    = len(bets)
    wins = sum(1 for b in bets if b["won"])

    print(f"\nДатасет: {n} ставок | Win: {wins/n:.1%} | Avg odds: {sum(b['odds'] for b in bets)/n:.2f}")
    print(f"{'Cap':<12} {'Final $':>10} {'Profit':>10} {'ROI':>8} {'Max DD':>8} {'Peak $':>10}")
    print("─" * 62)

    results = []
    for cap in CAPS:
        label = f"Pure Kelly" if cap is None else f"{cap*100:.0f}%"
        r = simulate(bets, cap)
        results.append((label, r))
        sign = "+" if r['profit'] >= 0 else ""
        print(f"{label:<12} ${r['final']:>9,.0f} {sign}${abs(r['profit']):>8,.0f} {r['roi']:>7.1f}% {r['max_dd']:>7.1f}% ${r['peak']:>9,.0f}")

    # Найкращий по фіналу
    best = max(results, key=lambda x: x[1]["final"])
    print(f"\n🏆 Найприбутковіший: {best[0]} → ${best[1]['final']:,.0f}")

    # Найкращий ROI/DD (Calmar ratio)
    calmar = max(results, key=lambda x: x[1]["roi"] / max(x[1]["max_dd"], 1))
    print(f"⚖️  Найкращий Calmar (ROI/DD): {calmar[0]} → ROI {calmar[1]['roi']:.1f}% / DD {calmar[1]['max_dd']:.1f}%")
