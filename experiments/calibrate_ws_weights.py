"""
experiments/calibrate_ws_weights.py

LogReg calibration of WS factor weights.
Uses all available seasons from DB.

For each in-range match (GAP >= min, odds in range):
  - Build binary factor vector (41 features)
  - Target: did the dominant side win?
  - Train LogisticRegression(L2) separately for home and away
  - Extract coefficients → new integer weights

Запуск: python -m experiments.calibrate_ws_weights
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model.train import load_data_from_db
from model.features.builder import build_dataset
from model.weighted_score import _get_factors, HOME_WEIGHTS, AWAY_WEIGHTS

# ─── Filter params (best config from backtest grid) ──────────────────────────
GAP_MIN   = 80
ODDS_MIN  = 2.20
ODDS_MAX  = 2.50
SPLIT_1   = pd.Timestamp("2025-01-01")
SPLIT_2   = pd.Timestamp("2025-07-01")

LEAGUES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga",
           135: "Serie A", 61: "Ligue 1", 94: "Primeira Liga"}


# ─── Factor names in order (must match _get_factors order) ───────────────────

HOME_FACTOR_NAMES = [
    "market_favors_home", "market_strong_home",
    "home_in_form", "home_strong_form",
    "away_out_of_form", "away_poor_form",
    "home_home_form", "home_home_wins",
    "away_away_poor", "away_away_loses",
    "elo_gap_large", "elo_gap_moderate", "elo_win_prob_high",
    "home_elo_strong", "away_elo_weak",
    "xg_attack_edge", "xg_ratio_home", "xg_diff_positive",
    "home_xg_regression", "away_xg_overperforming",
    "home_scoring_strong", "away_conceding_lots", "home_defense_solid",
    "table_home_higher", "table_points_home_better",
    "home_rested", "away_tired", "rest_advantage",
    "injury_advantage", "big_injury_advantage",
    "home_win_streak_3", "away_loss_streak_3",
    "home_clean_sheet_strong", "away_failed_to_score",
    "home_elo_rising", "away_elo_falling",
    "home_shots_edge", "home_possession_edge", "home_passing_edge",
    "away_gk_busy", "home_conversion_edge",
]

AWAY_FACTOR_NAMES = [
    "market_sees_away", "market_strong_away",
    "away_in_form", "away_strong_form",
    "home_out_of_form", "home_poor_form",
    "away_away_form", "away_away_wins",
    "home_home_poor", "home_home_loses",
    "elo_gap_away_large", "elo_gap_away_moderate", "elo_win_prob_low",
    "away_elo_strong", "home_elo_weak",
    "xg_away_attack", "xg_ratio_away", "xg_diff_away_positive",
    "away_xg_regression", "home_xg_overperforming",
    "away_scoring_strong", "home_conceding_lots", "away_defense_solid",
    "table_away_higher", "table_points_away_better",
    "away_rested", "home_tired", "rest_advantage_away",
    "injury_adv_away", "big_injury_adv_away",
    "away_win_streak_3", "home_loss_streak_3",
    "away_clean_sheet_strong", "home_failed_to_score",
    "away_elo_rising", "home_elo_falling",
    "away_shots_edge", "away_possession_edge", "away_passing_edge",
    "home_gk_busy", "away_conversion_edge",
]


def factor_vector(features: dict, outcome: str) -> np.ndarray:
    """Returns binary factor vector for a given outcome side."""
    factors = _get_factors(features, outcome)
    return np.array([1.0 if active else 0.0 for _, active in factors])


def compute_ws(features: dict, outcome: str) -> float:
    weights = HOME_WEIGHTS if outcome == "home" else AWAY_WEIGHTS
    total = 0.0
    for name, active in _get_factors(features, outcome):
        if active:
            total += weights.get(name, 1)
    return total


def build_logreg_dataset(dataset: pd.DataFrame, gap_min: float,
                          odds_min: float, odds_max: float):
    """
    Returns (X_home, y_home, X_away, y_away) — factor matrices for
    matches where the dominant side falls in the filter window.
    """
    home_X, home_y = [], []
    away_X, away_y = [], []

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
        if gap < gap_min:
            continue
        if not (odds_min <= odds <= odds_max):
            continue

        won = int(r["target"] == side)
        fv = factor_vector(r, side)

        if side == "home":
            home_X.append(fv)
            home_y.append(won)
        else:
            away_X.append(fv)
            away_y.append(won)

    return (np.array(home_X) if home_X else np.zeros((0, len(HOME_FACTOR_NAMES))),
            np.array(home_y),
            np.array(away_X) if away_X else np.zeros((0, len(AWAY_FACTOR_NAMES))),
            np.array(away_y))


def train_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    if len(X) < 20 or y.sum() < 5:
        return None, None
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                class_weight="balanced", random_state=42)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    return model, acc


def coefs_to_weights(coefs: np.ndarray, factor_names: list,
                     min_w: int = 1, max_w: int = 20) -> dict:
    """
    Map positive LogReg coefficients to integer weights in [min_w, max_w].
    Negative/zero coefficients → weight 0 (factor removed).
    """
    pos_mask = coefs > 0
    if not pos_mask.any():
        return {}

    pos_coefs = coefs.copy()
    pos_coefs[~pos_mask] = 0.0

    # Scale to [min_w, max_w]
    c_max = pos_coefs.max()
    if c_max <= 0:
        return {}

    weights = {}
    for name, c in zip(factor_names, pos_coefs):
        if c > 0:
            w = max(min_w, round((c / c_max) * max_w))
            weights[name] = w
    return weights


def print_factor_table(coefs: np.ndarray, factor_names: list, label: str):
    logger.info(f"\n{label}")
    logger.info(f"  {'Factor':<35} {'Coef':>8}  {'Sign':>6}")
    logger.info(f"  {'-'*54}")
    for name, c in sorted(zip(factor_names, coefs), key=lambda x: -x[1]):
        sign = "+" if c > 0 else "-" if c < 0 else "0"
        logger.info(f"  {name:<35} {c:>8.3f}  {sign:>6}")


def backtest_with_weights(dataset: pd.DataFrame, home_w: dict, away_w: dict,
                           gap_min, odds_min, odds_max, label: str):
    """Quick backtest using custom weights."""
    bets = []
    for _, row in dataset.iterrows():
        r = row.to_dict()
        h_odds = r.get("home_odds")
        a_odds = r.get("away_odds")
        if not h_odds or not a_odds:
            continue

        # Compute WS with new weights
        ws_h = sum((home_w.get(n, 0) if active else 0)
                   for n, active in _get_factors(r, "home"))
        ws_a = sum((away_w.get(n, 0) if active else 0)
                   for n, active in _get_factors(r, "away"))

        if ws_h >= ws_a:
            side, ws_dom, ws_weak, odds = "home", ws_h, ws_a, h_odds
        else:
            side, ws_dom, ws_weak, odds = "away", ws_a, ws_h, a_odds

        gap = ws_dom - ws_weak
        if gap < gap_min:
            continue
        if not (odds_min <= odds <= odds_max):
            continue

        bets.append({
            "date": row["date"],
            "won": row["target"] == side,
            "odds": odds,
        })

    if not bets:
        logger.info(f"  {label}: 0 bets")
        return

    df = pd.DataFrame(bets).sort_values("date")
    n = len(df)
    wr = df["won"].mean()
    flat_pnl = sum((o - 1 if w else -1) for w, o in zip(df["won"], df["odds"]))
    roi = flat_pnl / n * 100
    logger.info(f"  {label}: N={n}  WR={wr:.1%}  FlatROI={roi:+.1f}%")


def run():
    logger.info("Loading data from DB...")
    matches, stats, odds_data, teams, injuries = load_data_from_db()

    logger.info("Building dataset...")
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

    # ─── Train / test split ──────────────────────────────────────────────────
    train_df = dataset[dataset["date"] < SPLIT_1].copy()
    test1_df = dataset[(dataset["date"] >= SPLIT_1) & (dataset["date"] < SPLIT_2)].copy()
    test2_df = dataset[dataset["date"] >= SPLIT_2].copy()

    logger.info(f"Train: {len(train_df)}  |  Test1 (F1): {len(test1_df)}  |  Test2 (F2): {len(test2_df)}")

    # ─── Build LogReg datasets ───────────────────────────────────────────────
    logger.info(f"\nFilter: GAP>={GAP_MIN}, odds {ODDS_MIN}-{ODDS_MAX}")
    hX, hy, aX, ay = build_logreg_dataset(train_df, GAP_MIN, ODDS_MIN, ODDS_MAX)
    logger.info(f"In-range training: home={len(hX)} (win={hy.sum()}), away={len(aX)} (win={ay.sum()})")

    all_X = np.vstack([hX, aX]) if len(hX) and len(aX) else (hX if len(hX) else aX)
    all_y = np.concatenate([hy, ay])
    logger.info(f"Combined: {len(all_X)} samples, baseline WR={all_y.mean():.1%}")

    if len(all_X) < 30:
        logger.warning("Too few samples for LogReg. Load more historical data.")
        return

    # ─── Train models ────────────────────────────────────────────────────────
    logger.info("\nTraining LogReg (C=1.0)...")

    # Combined model (same weights for home/away when dominant)
    # Use HOME factors for home bets, AWAY factors for away bets
    h_model, h_acc = train_logreg(hX, hy, C=1.0)
    a_model, a_acc = train_logreg(aX, ay, C=1.0)

    if h_model:
        logger.info(f"  Home model: acc={h_acc:.3f} (n={len(hX)})")
        print_factor_table(h_model.coef_[0], HOME_FACTOR_NAMES, "HOME factor coefficients")
    else:
        logger.warning(f"  Home model: insufficient data (n={len(hX)})")

    if a_model:
        logger.info(f"  Away model: acc={a_acc:.3f} (n={len(aX)})")
        print_factor_table(a_model.coef_[0], AWAY_FACTOR_NAMES, "AWAY factor coefficients")
    else:
        logger.warning(f"  Away model: insufficient data (n={len(aX)})")

    # ─── New weights ─────────────────────────────────────────────────────────
    if h_model:
        new_home_w = coefs_to_weights(h_model.coef_[0], HOME_FACTOR_NAMES)
    else:
        new_home_w = HOME_WEIGHTS.copy()

    if a_model:
        new_away_w = coefs_to_weights(a_model.coef_[0], AWAY_FACTOR_NAMES)
    else:
        new_away_w = AWAY_WEIGHTS.copy()

    # ─── Negative factors (to consider zeroing out) ──────────────────────────
    if h_model:
        neg_home = [(n, c) for n, c in zip(HOME_FACTOR_NAMES, h_model.coef_[0]) if c < -0.1]
        if neg_home:
            logger.info(f"\nNegative HOME factors (candidates to zero out):")
            for n, c in sorted(neg_home, key=lambda x: x[1]):
                cur_w = HOME_WEIGHTS.get(n, 0)
                logger.info(f"  {n:<35} coef={c:+.3f}  current_weight={cur_w}")

    if a_model:
        neg_away = [(n, c) for n, c in zip(AWAY_FACTOR_NAMES, a_model.coef_[0]) if c < -0.1]
        if neg_away:
            logger.info(f"\nNegative AWAY factors (candidates to zero out):")
            for n, c in sorted(neg_away, key=lambda x: x[1]):
                cur_w = AWAY_WEIGHTS.get(n, 0)
                logger.info(f"  {n:<35} coef={c:+.3f}  current_weight={cur_w}")

    # ─── Backtest comparison ─────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(f"  BACKTEST: GAP={GAP_MIN} ODDS={ODDS_MIN}-{ODDS_MAX}")
    logger.info(f"{'='*65}")

    for label, df in [("TRAIN (before 2025-01)", train_df),
                      ("TEST1 (F1, Jan-Jun 2025)", test1_df),
                      ("TEST2 (F2, Jul 2025+)", test2_df)]:
        logger.info(f"\n--- {label} ---")
        backtest_with_weights(df, HOME_WEIGHTS, AWAY_WEIGHTS, GAP_MIN, ODDS_MIN, ODDS_MAX,
                              "Current weights")
        if h_model or a_model:
            backtest_with_weights(df, new_home_w, new_away_w, GAP_MIN, ODDS_MIN, ODDS_MAX,
                                  "LogReg weights")

    # ─── Print new weights for copy-paste ────────────────────────────────────
    if h_model:
        logger.info("\nNEW HOME_WEIGHTS (for copy-paste):")
        logger.info(repr(new_home_w))
    if a_model:
        logger.info("\nNEW AWAY_WEIGHTS (for copy-paste):")
        logger.info(repr(new_away_w))

    # ─── Also try C=0.1 (stronger regularization) ───────────────────────────
    logger.info("\n--- C=0.1 (stronger regularization) ---")
    h2, h2_acc = train_logreg(hX, hy, C=0.1)
    a2, a2_acc = train_logreg(aX, ay, C=0.1)

    if h2:
        new_home_w2 = coefs_to_weights(h2.coef_[0], HOME_FACTOR_NAMES)
        new_away_w2 = coefs_to_weights(a2.coef_[0], AWAY_FACTOR_NAMES) if a2 else AWAY_WEIGHTS
        logger.info(f"  Home acc={h2_acc:.3f}  Away acc={a2_acc:.3f}" if a2 else f"  Home acc={h2_acc:.3f}")
        for label, df in [("TEST1", test1_df), ("TEST2", test2_df)]:
            logger.info(f"\n{label}:")
            backtest_with_weights(df, new_home_w2, new_away_w2, GAP_MIN, ODDS_MIN, ODDS_MAX,
                                  "C=0.1 weights")

    logger.info("\nDone.")


if __name__ == "__main__":
    run()
