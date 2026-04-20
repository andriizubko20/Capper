"""
experiments/backtest_structural_gate.py

Тест структурного гейту: вимагаємо мінімальний "structural_ws" —
суму ваг лише структурних факторів (Elo, таблиця, ринок).

Мета: відсіяти матчі де GAP великий лише через форму/серії/match stats,
але Elo і таблиця не підтверджують перевагу (Chelsea-Everton проблема).

Запуск: python -m experiments.backtest_structural_gate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATABASE_URL', 'postgresql://capper:capper@localhost:5432/capper')

import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, AWAY_WEIGHTS, HOME_WEIGHTS

SPLIT_1    = pd.Timestamp("2025-01-01")
SPLIT_2    = pd.Timestamp("2025-07-01")
INITIAL_BR = 1000.0
FRAC       = 0.25
CAP        = 0.04
ODDS_MAX   = 3.0

TOP5 = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

# Структурні фактори — ТІЛЬКИ Elo + таблиця (без ринку!)
# Ринок (market_sees_away +16) завжди спрацьовує при odds≥2.20,
# тому виключаємо щоб фільтр мав реальну дискримінаційну силу.
STRUCTURAL_AWAY = {
    "elo_gap_away_large",       # elo_diff < -50        (+12)
    "elo_gap_away_moderate",    # elo_diff < -25        (+12)
    "elo_win_prob_low",         # elo_home_win_prob < 0.45  (+12)
    "table_points_away_better", # table_points_diff < -5    (+11)
    "table_away_higher",        # table_position_diff > 3   (+9)
    "home_elo_weak",            # home_elo < 1400           (+15)
    "away_elo_strong",          # away_elo > 1600           (+2)
}

# Ці параметри фіксовані — перевіряємо тільки ефект structural_min
GRID_CONFIGS = [
    # (label, odds_min, gap_min)
    ("ODDS≥2.20 GAP≥90",  2.20, 90),
    ("ODDS≥2.10 GAP≥90",  2.10, 90),
    ("ODDS≥2.20 GAP≥80",  2.20, 80),
    ("ODDS≥2.10 GAP≥110", 2.10, 110),
    ("ODDS≥2.00 GAP≥110", 2.00, 110),
]

STRUCTURAL_MIN_VALS = [0, 12, 20, 25, 30, 35, 40]


def compute_ws_split(features: dict, outcome: str) -> tuple[float, float]:
    """Повертає (structural_ws, total_ws)."""
    structural_set = STRUCTURAL_AWAY  # наразі тільки AWAY
    weights = AWAY_WEIGHTS if outcome == "away" else HOME_WEIGHTS
    total = 0.0
    structural = 0.0
    for name, active in _get_factors(features, outcome):
        if active:
            w = weights.get(name, 1)
            total += w
            if name in structural_set:
                structural += w
    return structural, total


def simulate(bets: list) -> dict:
    if not bets:
        return {"n": 0, "wins": 0, "wr": 0.0, "flat_roi": 0.0,
                "kelly_roi": 0.0, "bankroll": INITIAL_BR, "max_dd": 0.0}
    bankroll = INITIAL_BR
    peak = INITIAL_BR
    max_dd = 0.0
    staked = 0.0
    flat_pnl = 0.0
    wins = 0
    for b in sorted(bets, key=lambda x: x["date"]):
        p, odd = b["p_elo"], b["odds"]
        q = 1 - p
        denom = odd - 1
        kelly = max(0.0, (p * denom - q) / denom) * FRAC if denom > 0 else 0
        stake = min(bankroll * kelly, bankroll * CAP)
        if stake > 0 and bankroll > 0:
            if b["won"]:
                bankroll += stake * (odd - 1); wins += 1
            else:
                bankroll -= stake
            staked += stake
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak * 100)
        flat_pnl += (odd - 1) if b["won"] else -1.0
    n = len(bets)
    return {
        "n": n, "wins": wins, "wr": wins / n,
        "flat_roi": flat_pnl / n * 100,
        "kelly_roi": (bankroll - INITIAL_BR) / staked * 100 if staked > 0 else 0.0,
        "bankroll": round(bankroll, 2),
        "max_dd": round(max_dd, 1),
    }


def run():
    logger.info("Loading data...")
    matches_df, stats_df, odds_df, teams, injuries_df = load_data_from_db()
    logger.info("Building dataset...")
    df = build_dataset(matches_df, stats_df, odds_df, teams, injuries_df)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Dataset: {len(df)} rows")

    from db.session import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT m.id, l.name FROM matches m JOIN leagues l ON l.id = m.league_id"
        )).fetchall()
    league_map = {r[0]: r[1] for r in rows}

    logger.info("Pre-computing WS scores (structural + total)...")
    records = []
    for _, row in df.iterrows():
        r = row.to_dict()
        ws_h_struct, ws_h_total = compute_ws_split(r, "home")
        ws_a_struct, ws_a_total = compute_ws_split(r, "away")

        if ws_a_total <= ws_h_total:
            continue  # AWAY dominant only

        gap = ws_a_total - ws_h_total
        a_odds = r.get("away_odds")
        if not a_odds or a_odds > ODDS_MAX:
            continue

        mid = int(r.get("match_id", 0))
        league = league_map.get(mid, "")
        if league not in TOP5:
            continue

        p_elo = 1.0 - float(r.get("elo_home_win_prob", 0.5))

        records.append({
            "date":         row["date"],
            "league":       league,
            "odds":         round(a_odds, 2),
            "ws_gap":       gap,
            "ws_structural": ws_a_struct,  # структурний скор AWAY команди
            "p_elo":        p_elo,
            "won":          r.get("target") == "away",
        })

    all_df = pd.DataFrame(records).sort_values("date")
    f1_df  = all_df[(all_df["date"] >= SPLIT_1) & (all_df["date"] < SPLIT_2)]
    f2_df  = all_df[all_df["date"] >= SPLIT_2]
    logger.info(f"Base pool: {len(all_df)} records (AWAY dominant, Top-5, odds≤{ODDS_MAX})")

    # Розподіл структурного скору
    logger.info("Structural WS distribution:")
    for threshold in [0, 10, 12, 20, 25, 30, 35, 40]:
        n = (all_df["ws_structural"] >= threshold).sum()
        pct = n / len(all_df) * 100
        logger.info(f"  structural >= {threshold:3d}: {n:5d} ({pct:.1f}%)")

    # --- GRID ---
    results = []
    for label, odds_min, gap_min in GRID_CONFIGS:
        for struct_min in STRUCTURAL_MIN_VALS:

            def filt(d):
                return d[
                    (d["ws_gap"]       >= gap_min) &
                    (d["odds"]         >= odds_min) &
                    (d["ws_structural"] >= struct_min)
                ].to_dict("records")

            bets_all = filt(all_df)
            bets_f1  = filt(f1_df)
            bets_f2  = filt(f2_df)

            s_all = simulate(bets_all)
            s_f1  = simulate(bets_f1)
            s_f2  = simulate(bets_f2)

            avg_roi = (s_f1["flat_roi"] + s_f2["flat_roi"]) / 2

            results.append({
                "config":       label,
                "odds_min":     odds_min,
                "gap_min":      gap_min,
                "struct_min":   struct_min,
                "N_all":        s_all["n"],
                "N_F1":         s_f1["n"],
                "N_F2":         s_f2["n"],
                "WR_all_%":     round(s_all["wr"] * 100, 1),
                "FlatROI_all_%": round(s_all["flat_roi"], 1),
                "FlatROI_F1_%": round(s_f1["flat_roi"], 1),
                "FlatROI_F2_%": round(s_f2["flat_roi"], 1),
                "KellyROI_F2_%": round(s_f2["kelly_roi"], 1),
                "MaxDD_F2_%":   s_f2["max_dd"],
                "Avg_F1_F2_%":  round(avg_roi, 1),
            })

    res_df = pd.DataFrame(results)

    # --- Print per config ---
    for label, odds_min, gap_min in GRID_CONFIGS:
        sub = res_df[res_df["config"] == label]
        print(f"\n{'='*95}")
        print(f"  {label}  (struct_min sweep)")
        print(f"  {'Struct≥':>8} | {'N_all':>6} {'N_F1':>5} {'N_F2':>5} | "
              f"{'F1_ROI':>8} {'F2_ROI':>8} {'Avg':>8} | "
              f"{'KellyF2':>8} {'MaxDD':>7} {'BK_all':>8}")
        print(f"  {'-'*90}")
        for _, r in sub.iterrows():
            marker = " ←" if r["FlatROI_F2_%"] >= 10 and r["N_F2"] >= 20 else ""
            print(f"  {int(r['struct_min']):>8} | "
                  f"{int(r['N_all']):>6} {int(r['N_F1']):>5} {int(r['N_F2']):>5} | "
                  f"{r['FlatROI_F1_%']:>+8.1f}% {r['FlatROI_F2_%']:>+8.1f}% {r['Avg_F1_F2_%']:>+8.1f}% | "
                  f"{r['KellyROI_F2_%']:>+8.1f}% {r['MaxDD_F2_%']:>6.1f}% ${r['BK_all_$'] if 'BK_all_$' in r else 0:>7,.0f}"
                  f"{marker}")

    # --- Save CSV ---
    csv_path = "experiments/results/structural_gate.csv"
    res_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # --- Best overall ---
    good = res_df[
        (res_df["N_F2"] >= 20) & (res_df["N_F1"] >= 20) &
        (res_df["FlatROI_F1_%"] > 0) & (res_df["FlatROI_F2_%"] > 0)
    ].sort_values("Avg_F1_F2_%", ascending=False)

    print(f"\n{'='*95}")
    print("  BEST: both F1>0 AND F2>0, N_F2>=20")
    print(f"  {'Config':<22} {'Struct≥':>8} | {'N_all':>6} {'N_F2':>5} | "
          f"{'F1_ROI':>8} {'F2_ROI':>8} {'Avg':>8} | {'KellyF2':>8} {'MaxDD':>7}")
    print(f"  {'-'*90}")
    for _, r in good.head(15).iterrows():
        print(f"  {r['config']:<22} {int(r['struct_min']):>8} | "
              f"{int(r['N_all']):>6} {int(r['N_F2']):>5} | "
              f"{r['FlatROI_F1_%']:>+8.1f}% {r['FlatROI_F2_%']:>+8.1f}% {r['Avg_F1_F2_%']:>+8.1f}% | "
              f"{r['KellyROI_F2_%']:>+8.1f}% {r['MaxDD_F2_%']:>6.1f}%")


if __name__ == "__main__":
    run()
