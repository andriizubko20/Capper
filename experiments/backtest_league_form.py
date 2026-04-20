"""
experiments/backtest_league_form.py

Порівняння двох підходів до форми команди у WS Gap моделі:
  A) ALL  — форма з усіх турнірів (поточна поведінка)
  B) LEAGUE — форма тільки з матчів тієї ж ліги

Решта логіки ідентична backtest_ws_gap.py:
  WS Gap >= 80, WS Dom >= 80, Odds >= 1.70, Kelly 25% з cap 4%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger

from model.train import load_data_from_db
from model.features.builder import build_match_features
from model.features.elo import build_elo_snapshots, elo_features
from model.features.form import compute_form, compute_home_away_form, compute_rest_days
from model.features.xg import compute_xg_features, compute_xg_overperformance
from model.features.standings import build_standings_snapshots
from model.features.odds_features import market_implied_features
from model.weighted_score import compute_weighted_score

INITIAL_BANKROLL = 1000.0
FRACTIONAL       = 0.25
KELLY_CAP        = 0.04
WS_GAP_MIN       = 80
WS_DOM_MIN       = 80
ODDS_MIN         = 1.70

TOP5_API_IDS = {39, 140, 78, 135, 61}  # EPL, La Liga, Bundesliga, Serie A, Ligue 1


def get_league_map():
    from db.session import SessionLocal
    from db.models import League
    db = SessionLocal()
    try:
        return {l.id: (l.api_id, l.name) for l in db.query(League).all()}
    finally:
        db.close()


def build_features_variant(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    teams_elo_snaps: dict,
    standings_snaps: dict,
    league_form: bool,
) -> pd.DataFrame:
    """
    Будує датасет з фічами.
    league_form=True  → форма тільки з матчів своєї ліги
    league_form=False → форма з усіх турнірів (поточна поведінка)
    """
    finished = matches_df[matches_df["home_score"].notna()].copy()
    finished = finished.sort_values("date").reset_index(drop=True)

    rows = []
    for _, match in finished.iterrows():
        match_id  = int(match["id"])
        home_id   = int(match["home_team_id"])
        away_id   = int(match["away_team_id"])
        date      = pd.Timestamp(match["date"])
        league_id = int(match["league_id"])

        lf = league_id if league_form else None  # фільтр по лізі

        features = {}

        # Форма (загальна, останні 5)
        hf = compute_form(matches_df, home_id, date, n=5, league_id=lf)
        af = compute_form(matches_df, away_id, date, n=5, league_id=lf)
        features.update({f"home_{k}": v for k, v in hf.items()})
        features.update({f"away_{k}": v for k, v in af.items()})

        # Форма вдома/на виїзді
        hhf = compute_home_away_form(matches_df, home_id, date, side="home", n=5, league_id=lf)
        aaf = compute_home_away_form(matches_df, away_id, date, side="away", n=5, league_id=lf)
        features.update({f"home_{k}": v for k, v in hhf.items()})
        features.update({f"away_{k}": v for k, v in aaf.items()})

        # xG (завжди всі турніри — нема сенсу фільтрувати, xG і так рідко є для CL)
        hxg = compute_xg_features(stats_df, home_id, date, n=5)
        axg = compute_xg_features(stats_df, away_id, date, n=5)
        features.update({f"home_{k}": v for k, v in hxg.items()})
        features.update({f"away_{k}": v for k, v in axg.items()})

        hxg_op = compute_xg_overperformance(stats_df, home_id, date)
        axg_op = compute_xg_overperformance(stats_df, away_id, date)
        features["home_xg_overperformance"] = hxg_op["xg_overperformance"]
        features["away_xg_overperformance"] = axg_op["xg_overperformance"]

        # Elo (завжди всі турніри — загальна сила)
        elo_snap = teams_elo_snaps.get(match_id, {})
        home_elo = elo_snap.get(home_id, 1500.0)
        away_elo = elo_snap.get(away_id, 1500.0)
        features.update(elo_features(home_elo, away_elo))

        # Дні відпочинку (завжди всі турніри — реальна втома)
        hr = compute_rest_days(matches_df, home_id, date)
        ar = compute_rest_days(matches_df, away_id, date)
        features["home_rest_days"] = hr["rest_days"]
        features["away_rest_days"] = ar["rest_days"]
        features["rest_days_diff"] = hr["rest_days"] - ar["rest_days"]

        # Таблиця
        snap = standings_snaps.get(match_id)
        if snap:
            features.update(snap)
        else:
            from model.features.standings import compute_standings_features
            features.update(compute_standings_features(
                matches_df, league_id, home_id, away_id, date
            ))

        features["league_id"] = league_id
        features["match_id"]  = match_id
        features["date"]      = match["date"]

        if match["home_score"] > match["away_score"]:
            features["target"] = "home"
        elif match["home_score"] == match["away_score"]:
            features["target"] = "draw"
        else:
            features["target"] = "away"

        rows.append(features)

    return pd.DataFrame(rows)


def ws_gap_simulate(dataset: pd.DataFrame, odds_df: pd.DataFrame) -> dict:
    """Запускає WS Gap симуляцію на датасеті, повертає статистику."""
    odds_1x2 = odds_df[odds_df["market"] == "1x2"]
    odds_by_match = {int(mid): grp for mid, grp in odds_1x2.groupby("match_id")}

    # Додаємо market odds до датасету
    records = []
    for _, row in dataset.iterrows():
        match_id = int(row.get("match_id", 0))
        odds_grp = odds_by_match.get(match_id)
        if odds_grp is None or odds_grp.empty:
            continue

        closing = odds_grp[odds_grp["is_closing"] == True]
        src = closing if not closing.empty else odds_grp
        best = {}
        for _, o in src.iterrows():
            out = o["outcome"]
            if out not in best or o["value"] > best[out]:
                best[out] = o["value"]

        if not {"home", "draw", "away"}.issubset(best):
            continue

        h_odds, a_odds = best["home"], best["away"]
        h_prob = 1 / h_odds if h_odds > 0 else 0
        a_prob = 1 / a_odds if a_odds > 0 else 0

        try:
            ws_h = compute_weighted_score(row, "home")
            ws_a = compute_weighted_score(row, "away")
        except Exception:
            continue

        elo_home = float(row.get("elo_home_win_prob") or 0.5)
        if pd.isna(elo_home): elo_home = 0.5

        if ws_h >= ws_a:
            dominant, ws_dom, ws_weak = "home", ws_h, ws_a
            odds_val, p_elo = h_odds, elo_home
        else:
            dominant, ws_dom, ws_weak = "away", ws_a, ws_h
            odds_val, p_elo = a_odds, 1 - elo_home

        ws_gap = ws_dom - ws_weak

        records.append({
            "date":     row["date"],
            "match_id": match_id,
            "outcome":  dominant,
            "actual":   row["target"],
            "ws_dom":   ws_dom,
            "ws_gap":   ws_gap,
            "odds":     odds_val,
            "p_elo":    p_elo,
            "won":      row["target"] == dominant,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Фільтр WS Gap
    filtered = df[
        (df["ws_gap"] >= WS_GAP_MIN) &
        (df["ws_dom"] >= WS_DOM_MIN) &
        (df["odds"]   >= ODDS_MIN)
    ].sort_values("date").copy()

    if filtered.empty:
        return {"bets": 0}

    # Симуляція bankroll
    bankroll = INITIAL_BANKROLL
    total_staked = 0.0
    for _, bet in filtered.iterrows():
        p = bet["p_elo"]
        b = bet["odds"] - 1
        kelly = max(0.0, (p * b - (1 - p)) / b) * FRACTIONAL
        stake = min(bankroll * kelly, bankroll * KELLY_CAP)
        if stake <= 0 or bankroll <= 0:
            continue
        profit = stake * b if bet["won"] else -stake
        bankroll     += profit
        total_staked += stake

    wins   = filtered["won"].sum()
    n      = len(filtered)
    profit = bankroll - INITIAL_BANKROLL
    roi    = profit / total_staked * 100 if total_staked > 0 else 0.0

    return {
        "bets":      n,
        "wins":      int(wins),
        "win_rate":  wins / n if n > 0 else 0,
        "avg_odds":  filtered["odds"].mean(),
        "avg_ws_gap": filtered["ws_gap"].mean(),
        "profit":    round(profit, 2),
        "roi":       round(roi, 2),
        "bankroll":  round(bankroll, 2),
    }


def print_result(label: str, r: dict):
    if r["bets"] == 0:
        logger.info(f"  {label}: немає ставок")
        return
    logger.info(
        f"  {label:<25} | {r['bets']:>5} ставок | "
        f"win {r['win_rate']:>5.1%} ({r['wins']}/{r['bets']}) | "
        f"avg odds {r['avg_odds']:>5.2f} | avg gap {r['avg_ws_gap']:>5.1f} | "
        f"ROI {r['roi']:>+7.1f}% | P&L {r['profit']:>+8.2f} | "
        f"Банкрол ${r['bankroll']:>8.2f}"
    )


def run():
    logger.info("=" * 90)
    logger.info("BACKTEST: League-only форма vs All-competitions форма (WS Gap)")
    logger.info(f"Параметри: WS Gap ≥ {WS_GAP_MIN}, WS Dom ≥ {WS_DOM_MIN}, "
                f"Odds ≥ {ODDS_MIN}, Kelly {FRACTIONAL*100:.0f}% cap {KELLY_CAP*100:.0f}%")
    logger.info("=" * 90)

    league_map = get_league_map()  # {db_id: (api_id, name)}
    top5_db_ids = {db_id for db_id, (api_id, _) in league_map.items() if api_id in TOP5_API_IDS}

    logger.info("Завантажую дані з БД...")
    matches, stats, odds_data, teams_df, injuries = load_data_from_db()

    logger.info("Pre-compute: Elo snapshots + standings snapshots...")
    elo_snaps        = build_elo_snapshots(matches)
    standings_snaps  = build_standings_snapshots(matches)

    # Тільки Top-5 (де є реальні кросс-турнірні матчі CL vs Ліга)
    matches_top5 = matches[matches["league_id"].isin(top5_db_ids)].copy()
    logger.info(f"Top-5 матчів (завершених): {matches_top5['home_score'].notna().sum()}")

    logger.info("\nБудую датасет A: ALL competitions форма...")
    ds_all = build_features_variant(matches_top5, stats, elo_snaps, standings_snaps, league_form=False)
    logger.info(f"  Рядків: {len(ds_all)}")

    logger.info("Будую датасет B: LEAGUE-ONLY форма...")
    ds_league = build_features_variant(matches_top5, stats, elo_snaps, standings_snaps, league_form=True)
    logger.info(f"  Рядків: {len(ds_league)}")

    logger.info("\nЗапускаю WS Gap симуляцію...")
    r_all    = ws_gap_simulate(ds_all,    odds_data)
    r_league = ws_gap_simulate(ds_league, odds_data)

    logger.info("\n" + "=" * 90)
    logger.info("РЕЗУЛЬТАТИ:")
    logger.info("-" * 90)
    print_result("A) All competitions", r_all)
    print_result("B) League-only",      r_league)
    logger.info("=" * 90)

    # Різниця
    if r_all["bets"] > 0 and r_league["bets"] > 0:
        logger.info(f"\nВідмінність (B - A):")
        logger.info(f"  Ставок:    {r_league['bets'] - r_all['bets']:+d} "
                    f"({'більше' if r_league['bets'] > r_all['bets'] else 'менше'})")
        logger.info(f"  Win rate:  {(r_league['win_rate'] - r_all['win_rate'])*100:+.2f}%")
        logger.info(f"  ROI:       {r_league['roi'] - r_all['roi']:+.2f}%")
        logger.info(f"  P&L:       {r_league['profit'] - r_all['profit']:+.2f}")
        logger.info(f"  Avg gap:   {r_league['avg_ws_gap'] - r_all['avg_ws_gap']:+.2f} "
                    f"({'більший' if r_league['avg_ws_gap'] > r_all['avg_ws_gap'] else 'менший'} домінування)")

    # Зберігаємо деталі
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "results", "backtest_league_form.txt")
    with open(out, "w") as f:
        f.write(f"A) All competitions:\n  {r_all}\n\n")
        f.write(f"B) League-only:\n  {r_league}\n")
    logger.info(f"\nЗбережено: {out}")


if __name__ == "__main__":
    run()
