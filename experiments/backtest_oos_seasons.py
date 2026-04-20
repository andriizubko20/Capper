"""
experiments/backtest_oos_seasons.py

Out-of-sample тест по сезонах:
  Train: сезон 2024-25 (до 2025-07-01)
  Test:  сезон 2025-26 (з 2025-07-01)

Нічиї включені як програші (реалістично).
Flat betting для чистого вимірювання edge.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import compute_weighted_score

LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    88:  "Eredivisie",
    144: "Jupiler Pro League",
    94:  "Primeira Liga",
}

SPLIT_DATE = pd.Timestamp("2025-07-01")

# Параметри що показали найкращий результат в in-sample
WS_GAP_MIN  = 80
WS_DOM_MIN  = 60
ODDS_MIN    = 1.5
KELLY_CAP   = 0.02
FRACTIONAL  = 0.25
INITIAL_BR  = 1000.0


def get_league_db_ids():
    from db.session import SessionLocal
    from db.models import League
    db = SessionLocal()
    try:
        result = {}
        for api_id, name in LEAGUES.items():
            for l in db.query(League).filter_by(api_id=api_id).all():
                result[l.id] = name
        return result
    finally:
        db.close()


def build_candidates(dataset: pd.DataFrame, league_map: dict) -> pd.DataFrame:
    """Рахує WS для кожного матчу, повертає кандидатів зі всіма метриками."""
    records = []
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
            side, ws_dom, ws_weak = "home", ws_h, ws_a
            odds, p_elo = round(1 / h_prob, 2), elo_home
        else:
            side, ws_dom, ws_weak = "away", ws_a, ws_h
            odds, p_elo = round(1 / a_prob, 2), 1.0 - elo_home

        records.append({
            "date":      pd.Timestamp(row["date"]),
            "league":    league_map.get(row.get("league_id"), "Other"),
            "outcome":   side,
            "actual":    row["target"],
            "ws_dom":    ws_dom,
            "ws_gap":    ws_dom - ws_weak,
            "odds":      odds,
            "p_elo":     p_elo,
            "won":       row["target"] == side,
        })

    return pd.DataFrame(records)


def simulate_kelly(df: pd.DataFrame) -> dict:
    bankroll = INITIAL_BR
    staked   = 0.0
    for _, bet in df.sort_values("date").iterrows():
        p = bet["p_elo"]
        b = bet["odds"] - 1
        kelly = max(0.0, (p * b - (1 - p)) / b) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * KELLY_CAP)
        if stake <= 0 or bankroll <= 0:
            continue
        bankroll += stake * b if bet["won"] else -stake
        staked   += stake
    profit = bankroll - INITIAL_BR
    roi    = profit / staked * 100 if staked > 0 else 0.0
    return {"bankroll": round(bankroll, 2), "profit": round(profit, 2), "roi": round(roi, 2)}


def show_split(label: str, df: pd.DataFrame, league_map: dict):
    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}  ({len(df)} матчів з odds)")
    logger.info(f"{'='*70}")

    filtered = df[
        (df["ws_gap"] >= WS_GAP_MIN) &
        (df["ws_dom"] >= WS_DOM_MIN) &
        (df["odds"]   >= ODDS_MIN)
    ].copy()

    if filtered.empty:
        logger.info("  Немає ставок з поточними фільтрами")
        return

    n     = len(filtered)
    wins  = filtered["won"].sum()
    flat  = (filtered["won"] * (filtered["odds"] - 1) - ~filtered["won"]).sum()
    kelly = simulate_kelly(filtered)

    logger.info(
        f"  ВСЬОГО:  {n} ставок | win {wins/n:.1%} ({wins}/{n}) | "
        f"avg odds {filtered['odds'].mean():.2f} | avg gap {filtered['ws_gap'].mean():.1f}"
    )
    logger.info(
        f"  Flat ROI:  {flat/n*100:+.1f}%  (P&L {flat:+.1f} units)"
    )
    logger.info(
        f"  Kelly ROI: {kelly['roi']:+.1f}%  "
        f"(P&L ${kelly['profit']:+.2f} | банкрол ${kelly['bankroll']:.2f})"
    )

    # По лігах
    logger.info(f"\n  {'Ліга':<23} {'Ставок':>7} {'Win%':>7} {'Flat ROI':>10}")
    logger.info(f"  {'-'*50}")
    for league, g in sorted(
        filtered.groupby("league"),
        key=lambda x: (x[1]["won"] * (x[1]["odds"] - 1) - ~x[1]["won"]).sum() / len(x[1]),
        reverse=True
    ):
        fp = (g["won"] * (g["odds"] - 1) - ~g["won"]).sum()
        logger.info(f"  {league:<23} {len(g):>7} {g['won'].mean():>7.1%} {fp/len(g)*100:>+10.1f}%")

    # По Gap buckets
    logger.info(f"\n  WS Gap buckets:")
    for lo, hi in [(80, 100), (100, 130), (130, 200)]:
        g = filtered[(filtered["ws_gap"] >= lo) & (filtered["ws_gap"] < hi)]
        if len(g) < 5:
            continue
        fp = (g["won"] * (g["odds"] - 1) - ~g["won"]).sum()
        logger.info(f"    Gap {lo}-{hi}: {len(g):>4} ставок | win {g['won'].mean():.1%} | flat ROI {fp/len(g)*100:+.1f}%")


def run():
    logger.info("=" * 70)
    logger.info("OOS BACKTEST: сезон 2024-25 (train) vs 2025-26 (test)")
    logger.info(f"Фільтри: WS Gap≥{WS_GAP_MIN}, Dom≥{WS_DOM_MIN}, Odds≥{ODDS_MIN}")
    logger.info(f"Kelly: fractional {FRACTIONAL*100:.0f}%, cap {KELLY_CAP*100:.0f}%")
    logger.info("=" * 70)

    league_map = get_league_db_ids()

    logger.info("Завантажую дані...")
    matches, stats, odds_data, teams_df, injuries = load_data_from_db()

    logger.info("Будую фічі...")
    dataset = build_dataset(matches, stats, odds_data, teams_df, injuries_df=injuries)
    dataset = dataset[dataset["league_id"].isin(league_map.keys())].copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values("date").reset_index(drop=True)

    train = dataset[dataset["date"] < SPLIT_DATE]
    test  = dataset[dataset["date"] >= SPLIT_DATE]

    logger.info(f"Train (до {SPLIT_DATE.date()}): {len(train)} матчів")
    logger.info(f"Test  (з {SPLIT_DATE.date()}): {len(test)} матчів")

    logger.info("\nБудую кандидатів...")
    train_df = build_candidates(train, league_map)
    test_df  = build_candidates(test,  league_map)

    show_split(f"IN-SAMPLE  (2024-25)", train_df, league_map)
    show_split(f"OUT-OF-SAMPLE (2025-26)", test_df, league_map)

    # Порівняння Serie A окремо
    logger.info(f"\n{'='*70}")
    logger.info("  SERIE A — детальне порівняння")
    logger.info(f"{'='*70}")
    for label, df in [("IN-SAMPLE  ", train_df), ("OOS        ", test_df)]:
        g = df[
            (df["league"] == "Serie A") &
            (df["ws_gap"] >= WS_GAP_MIN) &
            (df["ws_dom"] >= WS_DOM_MIN) &
            (df["odds"]   >= ODDS_MIN)
        ]
        if g.empty:
            logger.info(f"  {label}: немає ставок")
            continue
        fp = (g["won"] * (g["odds"] - 1) - ~g["won"]).sum()
        logger.info(
            f"  {label}: {len(g)} ставок | win {g['won'].mean():.1%} | "
            f"avg odds {g['odds'].mean():.2f} | flat ROI {fp/len(g)*100:+.1f}%"
        )

    # Збереження
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "results", "oos_seasons_test.csv")
    test_df.to_csv(out, index=False)
    logger.info(f"\nЗбережено test дані: {out}")


if __name__ == "__main__":
    run()
