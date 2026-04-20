"""
experiments/pattern_analysis.py

Аналіз паттернів: коли модель виграє vs програє.

Для кожного in-range бету (GAP>=80, odds 2.2-2.5):
  1. Factor lift — які фактори корелюють з перемогою
  2. Breakdown по лізі, odds bucket, WS gap, місяці
  3. Небезпечні паттерни — коли програємо
  4. Топ комбінації факторів

Запуск: python -m experiments.pattern_analysis
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

GAP_MIN  = 80
ODDS_MIN = 2.20
ODDS_MAX = 2.50

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}


def compute_ws(features, outcome):
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    return sum(weights.get(n, 1) for n, active in _get_factors(features, outcome) if active)


def collect_bets(dataset):
    """Collect all in-range bets with full factor info."""
    rows = []
    for _, row in dataset.iterrows():
        r = row.to_dict()
        h_odds = r.get("home_odds")
        a_odds = r.get("away_odds")
        if not h_odds or not a_odds:
            continue

        ws_h = compute_ws(r, "home")
        ws_a = compute_ws(r, "away")

        if ws_h >= ws_a:
            side, ws_dom, ws_weak, odds = "home", ws_h, ws_a, h_odds
        else:
            side, ws_dom, ws_weak, odds = "away", ws_a, ws_h, a_odds

        gap = ws_dom - ws_weak
        if gap < GAP_MIN:
            continue
        if not (ODDS_MIN <= odds <= ODDS_MAX):
            continue

        # Active factors for dominant side
        active_factors = [n for n, active in _get_factors(r, side) if active]

        won = r["target"] == side
        p_elo = float(r.get("elo_home_win_prob", 0.5))
        if side == "away":
            p_elo = 1.0 - p_elo

        rows.append({
            "date":   row["date"],
            "league": row.get("league", "?"),
            "side":   side,
            "odds":   round(odds, 2),
            "ws_dom": ws_dom,
            "ws_weak": ws_weak,
            "gap":    gap,
            "won":    won,
            "p_elo":  p_elo,
            "n_factors": len(active_factors),
            "factors": active_factors,
            # Key dimensions
            "home_odds_raw": h_odds,
            "away_odds_raw": a_odds,
            "market_home_prob": r.get("market_home_prob", 0),
            "elo_home_win_prob": r.get("elo_home_win_prob", 0.5),
        })

    return pd.DataFrame(rows)


def section(title):
    logger.info(f"\n{'='*65}")
    logger.info(f"  {title}")
    logger.info(f"{'='*65}")


def show_breakdown(df, col, label, min_n=3):
    """Show WR + FlatROI breakdown by a categorical column."""
    logger.info(f"\n  {'Segment':<28} {'N':>5}  {'WR%':>7}  {'FlatROI':>9}")
    logger.info(f"  {'-'*55}")
    grp = df.groupby(col).agg(
        n=("won", "count"),
        wins=("won", "sum"),
        avg_odds=("odds", "mean"),
    ).reset_index()
    grp["wr"] = grp["wins"] / grp["n"]
    grp["flat_roi"] = grp.apply(
        lambda r: df[df[col] == r[col]].apply(
            lambda b: (b["odds"] - 1) if b["won"] else -1, axis=1
        ).sum() / r["n"] * 100, axis=1
    )
    grp = grp[grp["n"] >= min_n].sort_values("flat_roi", ascending=False)
    for _, r in grp.iterrows():
        logger.info(f"  {str(r[col]):<28} {int(r['n']):>5}  {r['wr']:>7.1%}  {r['flat_roi']:>+9.1f}%")


def factor_lift(df, min_n=5):
    """Compute WR for each factor vs baseline."""
    baseline_wr = df["won"].mean()
    results = []

    # Collect all unique factors
    all_factors = set()
    for flist in df["factors"]:
        all_factors.update(flist)

    for factor in sorted(all_factors):
        has_factor = df["factors"].apply(lambda fl: factor in fl)
        sub = df[has_factor]
        n = len(sub)
        if n < min_n:
            continue
        wr = sub["won"].mean()
        lift = wr / baseline_wr - 1
        flat_roi = sub.apply(
            lambda b: (b["odds"] - 1) if b["won"] else -1, axis=1
        ).sum() / n * 100
        results.append({
            "factor": factor, "n": n, "wr": wr,
            "lift": lift, "flat_roi": flat_roi
        })

    return pd.DataFrame(results).sort_values("flat_roi", ascending=False)


def anti_factor_analysis(df, min_n=5):
    """Find factors that appear more in losses than wins."""
    baseline_wr = df["won"].mean()
    all_factors = set()
    for flist in df["factors"]:
        all_factors.update(flist)

    results = []
    for factor in sorted(all_factors):
        # WR with vs without factor
        has = df["factors"].apply(lambda fl: factor in fl)
        n_with = has.sum()
        n_without = (~has).sum()
        if n_with < min_n or n_without < min_n:
            continue
        wr_with = df[has]["won"].mean()
        wr_without = df[~has]["won"].mean()
        diff = wr_with - wr_without
        results.append({
            "factor": factor,
            "n_with": n_with, "wr_with": wr_with,
            "n_without": n_without, "wr_without": wr_without,
            "diff": diff,
        })

    return pd.DataFrame(results).sort_values("diff")


def combo_analysis(df, top_n=10, min_n=4):
    """Find 2-factor combinations with best/worst WR."""
    from itertools import combinations

    # Use only frequent factors (appear in >= 10% of bets)
    threshold = max(min_n, len(df) * 0.05)
    all_factors = set()
    for flist in df["factors"]:
        all_factors.update(flist)
    freq_factors = [f for f in all_factors
                    if df["factors"].apply(lambda fl: f in fl).sum() >= threshold]

    results = []
    for f1, f2 in combinations(freq_factors, 2):
        mask = df["factors"].apply(lambda fl: f1 in fl and f2 in fl)
        n = mask.sum()
        if n < min_n:
            continue
        wr = df[mask]["won"].mean()
        flat_roi = df[mask].apply(
            lambda b: (b["odds"] - 1) if b["won"] else -1, axis=1
        ).sum() / n * 100
        results.append({"f1": f1, "f2": f2, "n": n, "wr": wr, "flat_roi": flat_roi})

    return pd.DataFrame(results).sort_values("flat_roi", ascending=False) if results else pd.DataFrame()


def dangerous_patterns(df):
    """Identify specific conditions where WR drops below 40%."""
    patterns = []

    # Low Elo probability
    mask = df["p_elo"] < 0.40
    if mask.sum() >= 4:
        sub = df[mask]
        patterns.append(("p_elo < 40%", len(sub), sub["won"].mean()))

    # Very high odds
    mask = df["odds"] >= 2.40
    if mask.sum() >= 4:
        sub = df[mask]
        patterns.append(("odds >= 2.40", len(sub), sub["won"].mean()))

    # Low WS gap (barely above threshold)
    mask = df["gap"] < 95
    if mask.sum() >= 4:
        sub = df[mask]
        patterns.append((f"gap 80-95", len(sub), sub["won"].mean()))

    mask = df["gap"] >= 95
    if mask.sum() >= 4:
        sub = df[mask]
        patterns.append((f"gap >= 95", len(sub), sub["won"].mean()))

    # Market disagrees (high market_home_prob when betting away)
    away_bets = df[df["side"] == "away"]
    if len(away_bets) >= 4:
        mask_mkt = away_bets["market_home_prob"] > 0.55
        if mask_mkt.sum() >= 4:
            sub = away_bets[mask_mkt]
            patterns.append(("away bet, mkt_home>55%", len(sub), sub["won"].mean()))
        mask_mkt2 = away_bets["market_home_prob"] <= 0.55
        if mask_mkt2.sum() >= 4:
            sub = away_bets[mask_mkt2]
            patterns.append(("away bet, mkt_home<=55%", len(sub), sub["won"].mean()))

    # Few active factors (weak signal)
    mask = df["n_factors"] <= 8
    if mask.sum() >= 4:
        patterns.append((f"n_factors <= 8", df[mask].shape[0], df[mask]["won"].mean()))
    mask = df["n_factors"] >= 12
    if mask.sum() >= 4:
        patterns.append((f"n_factors >= 12", df[mask].shape[0], df[mask]["won"].mean()))

    return patterns


def run():
    logger.info("Loading data...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()
    dataset = build_dataset(matches, stats, odds_data, teams, injuries_df=injuries)
    dataset["date"] = pd.to_datetime(dataset["date"])

    from db.session import SessionLocal
    from db.models import League
    db = SessionLocal()
    try:
        league_map = {l.id: LEAGUES[l.api_id]
                      for l in db.query(League).all() if l.api_id in LEAGUES}
    finally:
        db.close()

    dataset = dataset[dataset["league_id"].isin(league_map)].copy()
    dataset = dataset[dataset["market_home_prob"].notna()].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset["league"] = dataset["league_id"].map(league_map)

    logger.info(f"Dataset: {len(dataset)} matches  ({dataset['date'].min().date()} – {dataset['date'].max().date()})")

    # ─── Collect bets ────────────────────────────────────────────────────────
    bets = collect_bets(dataset)
    if bets.empty:
        logger.error("No bets found!")
        return

    bets["month"] = bets["date"].dt.to_period("M")
    bets["odds_bucket"] = pd.cut(bets["odds"],
                                  bins=[2.19, 2.30, 2.40, 2.51],
                                  labels=["2.20-2.30", "2.30-2.40", "2.40-2.50"])
    bets["gap_bucket"] = pd.cut(bets["gap"],
                                 bins=[79, 90, 100, 110, 999],
                                 labels=["80-90", "90-100", "100-110", "110+"])

    overall_wr = bets["won"].mean()
    n_total = len(bets)

    section(f"OVERVIEW  (GAP≥{GAP_MIN}, odds {ODDS_MIN}-{ODDS_MAX})")
    logger.info(f"\n  Total bets: {n_total}")
    logger.info(f"  Win Rate:   {overall_wr:.1%}  ({int(bets['won'].sum())}W / {int((~bets['won']).sum())}L)")
    flat_pnl = bets.apply(lambda b: (b["odds"] - 1) if b["won"] else -1, axis=1).sum()
    logger.info(f"  Flat ROI:   {flat_pnl/n_total*100:+.1f}%  (total P&L: {flat_pnl:+.1f} units)")
    logger.info(f"  Side split: {(bets['side']=='home').sum()} home / {(bets['side']=='away').sum()} away")
    logger.info(f"  Avg odds:   {bets['odds'].mean():.2f}")
    logger.info(f"  Date range: {bets['date'].min().date()} – {bets['date'].max().date()}")

    # ─── By League ───────────────────────────────────────────────────────────
    section("BREAKDOWN BY LEAGUE")
    show_breakdown(bets, "league", "League", min_n=2)

    # ─── By Odds Bucket ──────────────────────────────────────────────────────
    section("BREAKDOWN BY ODDS BUCKET")
    show_breakdown(bets, "odds_bucket", "Odds", min_n=2)

    # ─── By WS Gap ───────────────────────────────────────────────────────────
    section("BREAKDOWN BY WS GAP")
    show_breakdown(bets, "gap_bucket", "Gap", min_n=2)

    # ─── By Side ─────────────────────────────────────────────────────────────
    section("HOME vs AWAY")
    show_breakdown(bets, "side", "Side", min_n=2)

    # ─── Monthly WR ──────────────────────────────────────────────────────────
    section("MONTHLY WIN RATE")
    logger.info(f"\n  {'Month':<12} {'N':>4}  {'WR%':>7}  {'FlatROI':>9}")
    logger.info(f"  {'-'*36}")
    monthly = bets.groupby("month").apply(lambda g: pd.Series({
        "n": len(g),
        "wr": g["won"].mean(),
        "roi": g.apply(lambda b: (b["odds"]-1) if b["won"] else -1, axis=1).sum() / len(g) * 100,
    })).reset_index()
    for _, r in monthly.iterrows():
        bar = "█" * int(r["wr"] * 20)
        logger.info(f"  {str(r['month']):<12} {int(r['n']):>4}  {r['wr']:>7.1%}  {r['roi']:>+9.1f}%  {bar}")

    # ─── Factor Lift ─────────────────────────────────────────────────────────
    section("FACTOR LIFT (sorted by FlatROI when factor is active)")
    lift_df = factor_lift(bets, min_n=5)
    if not lift_df.empty:
        logger.info(f"\n  {'Factor':<35} {'N':>5}  {'WR%':>7}  {'Lift':>7}  {'ROI':>9}")
        logger.info(f"  {'-'*68}")
        # Top 15 best
        logger.info("  --- TOP 15 (best ROI when active) ---")
        for _, r in lift_df.head(15).iterrows():
            logger.info(f"  {r['factor']:<35} {int(r['n']):>5}  {r['wr']:>7.1%}  {r['lift']:>+7.1%}  {r['flat_roi']:>+9.1f}%")
        # Bottom 10 worst
        logger.info("\n  --- BOTTOM 10 (worst ROI when active) ---")
        for _, r in lift_df.tail(10).iterrows():
            logger.info(f"  {r['factor']:<35} {int(r['n']):>5}  {r['wr']:>7.1%}  {r['lift']:>+7.1%}  {r['flat_roi']:>+9.1f}%")

    # ─── Anti-factors ─────────────────────────────────────────────────────────
    section("FACTORS THAT HURT (WR with vs without)")
    anti_df = anti_factor_analysis(bets, min_n=5)
    if not anti_df.empty:
        logger.info(f"\n  {'Factor':<35} {'N(w)':>6}  {'WR(w)':>7}  {'N(wo)':>6}  {'WR(wo)':>7}  {'Diff':>7}")
        logger.info(f"  {'-'*75}")
        # Worst 10 (biggest negative diff = factor hurts)
        for _, r in anti_df.head(10).iterrows():
            logger.info(
                f"  {r['factor']:<35} {int(r['n_with']):>6}  {r['wr_with']:>7.1%}  "
                f"{int(r['n_without']):>6}  {r['wr_without']:>7.1%}  {r['diff']:>+7.1%}"
            )

    # ─── Dangerous patterns ───────────────────────────────────────────────────
    section("DANGEROUS PATTERNS (low WR conditions)")
    for name, n, wr in dangerous_patterns(bets):
        flat = bets[bets.apply(
            lambda b: (b["odds"] >= 2.40 if name == "odds >= 2.40"
                       else b["gap"] < 95 if "80-95" in name
                       else b["gap"] >= 95 if ">= 95" in name
                       else b["p_elo"] < 0.40 if "p_elo" in name
                       else b["n_factors"] <= 8 if "<= 8" in name
                       else b["n_factors"] >= 12 if ">= 12" in name
                       else False), axis=1
        )]["won"].mean() if n > 0 else 0
        marker = "⚠️ " if wr < overall_wr - 0.05 else ("✓ " if wr > overall_wr + 0.05 else "  ")
        logger.info(f"  {marker} {name:<35}  N={n:>4}  WR={wr:.1%}  (baseline {overall_wr:.1%})")

    # ─── Top factor combos ───────────────────────────────────────────────────
    section("BEST 2-FACTOR COMBINATIONS (min 4 bets)")
    combo_df = combo_analysis(bets, min_n=4)
    if not combo_df.empty:
        logger.info(f"\n  {'F1':<30} {'F2':<30} {'N':>4}  {'WR%':>7}  {'ROI':>9}")
        logger.info(f"  {'-'*83}")
        logger.info("  --- TOP 10 ---")
        for _, r in combo_df.head(10).iterrows():
            logger.info(f"  {r['f1']:<30} {r['f2']:<30} {int(r['n']):>4}  {r['wr']:>7.1%}  {r['flat_roi']:>+9.1f}%")
        logger.info("\n  --- BOTTOM 5 ---")
        for _, r in combo_df.tail(5).iterrows():
            logger.info(f"  {r['f1']:<30} {r['f2']:<30} {int(r['n']):>4}  {r['wr']:>7.1%}  {r['flat_roi']:>+9.1f}%")

    # ─── WS gap vs ELO probability matrix ─────────────────────────────────────
    section("GAP × ELO PROBABILITY MATRIX")
    bets["elo_bucket"] = pd.cut(bets["p_elo"],
                                 bins=[0, 0.40, 0.50, 0.60, 1.0],
                                 labels=["<40%", "40-50%", "50-60%", ">60%"])
    matrix = bets.groupby(["gap_bucket", "elo_bucket"]).apply(
        lambda g: f"N={len(g)} WR={g['won'].mean():.0%}" if len(g) >= 3 else f"N={len(g)}"
    ).unstack()
    logger.info(f"\n{matrix.to_string()}")

    # ─── Save bets CSV ────────────────────────────────────────────────────────
    os.makedirs("experiments/results", exist_ok=True)
    out = bets[["date", "league", "side", "odds", "gap", "n_factors",
                "won", "p_elo", "odds_bucket", "gap_bucket"]].copy()
    out.to_csv("experiments/results/pattern_analysis_bets.csv", index=False)
    logger.info(f"\nSaved: experiments/results/pattern_analysis_bets.csv")


if __name__ == "__main__":
    run()
