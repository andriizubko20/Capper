"""
model/gem/analyze.py

Post-training analyses for the Gem model. Loads OOF predictions + match
info from artifacts/ and answers:

  1. gem_score analysis  — does (our_prob - market_prob) predict WIN better
                            than just our_prob alone?
  2. filter sweep        — Pareto frontier over (P_thr, MAX_DRAW, MIN_ODDS,
                            gem_score_thr) → (yield, WR, ROI)
  3. per-cluster + per-league breakdown — where is the alpha?
  4. feature signatures  — what do WINNING gem picks look like vs losers?
  5. market vs model     — direct calibration comparison

NOTE on odds: 94% of historical odds are post-match closing snapshots.
gem_score computed here uses CLOSING-derived market_prob, so the absolute
edge is conservative-biased relative to opening odds. Trends and rankings
remain valid; absolute ROI is optimistic by ~2-3pp.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.gem.features import market_probs_from_odds
from model.gem.niches import (
    FLAT_STAKE_FRAC, MAX_DRAW_PROB, MAX_ODDS, MIN_BET_PROB, MIN_ODDS,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
REPORTS_DIR   = Path(__file__).parent / "reports"


# ── Load OOF data ────────────────────────────────────────────────────────────

def load_oof(directory: Path = ARTIFACTS_DIR) -> dict:
    npz = np.load(directory / "oof.npz")
    info = pd.read_parquet(directory / "info.parquet")
    with open(directory / "feature_names.json") as f:
        feature_names = json.load(f)
    return {
        "oof_xgb": npz["oof_xgb"],
        "oof_lgb": npz["oof_lgb"],
        "oof_cat": npz["oof_cat"],
        "oof_raw": npz["oof_ensemble_raw"],
        "oof_cal": npz["oof_ensemble_calibrated"],
        "y":       npz["y"],
        "covered": npz["covered"],
        "info":    info,
        "feature_names": feature_names,
    }


# ── Build pick table ─────────────────────────────────────────────────────────

def build_picks(
    proba: np.ndarray,
    info: pd.DataFrame,
    y: np.ndarray,
    covered: np.ndarray,
    max_draw: float = MAX_DRAW_PROB,
    min_bet:  float = MIN_BET_PROB,
    min_odds: float = MIN_ODDS,
    max_odds: float = MAX_ODDS,
    gem_score_thr: float | None = None,  # None = ignore market gap filter
) -> pd.DataFrame:
    """
    Apply gem filter to ALL covered rows; return a DataFrame with one row per
    candidate pick (passes filter), enriched with market_prob, gem_score,
    won/lost, etc.
    """
    df = info.reset_index(drop=True).copy()
    df["pH"] = proba[:, 0]
    df["pD"] = proba[:, 1]
    df["pA"] = proba[:, 2]
    df["covered"] = covered
    df["y"] = y
    df = df[df["covered"]].reset_index(drop=True)

    # Side selection: argmax(P_H, P_A) — never bet draw
    side = np.where(df["pH"].values >= df["pA"].values, "H", "A")
    p_side = np.where(side == "H", df["pH"], df["pA"])
    side_odds = np.where(side == "H", df["home_odds"], df["away_odds"])

    # Market probabilities (de-vigged)
    market_p = []
    for h, d, a in zip(df["home_odds"], df["draw_odds"], df["away_odds"]):
        mp = market_probs_from_odds(h, d, a)
        market_p.append(mp)
    market_p = pd.DataFrame(market_p)
    market_side_p = np.where(side == "H", market_p["home"], market_p["away"])

    df["side"]         = side
    df["p_side"]       = p_side
    df["side_odds"]    = side_odds
    df["market_p_side"] = market_side_p
    df["gem_score"]    = df["p_side"] - df["market_p_side"]
    df["won"] = (
        ((df["side"] == "H") & (df["result"] == "H")) |
        ((df["side"] == "A") & (df["result"] == "A"))
    ).astype(int)

    # Apply filter
    mask = (
        (df["pD"] < max_draw) &
        (df["p_side"] > min_bet) &
        (df["side_odds"].notna()) &
        (df["side_odds"] >= min_odds) &
        (df["side_odds"] <= max_odds)
    )
    if gem_score_thr is not None:
        mask = mask & (df["gem_score"] > gem_score_thr)
    return df[mask].reset_index(drop=True)


# ── 1. gem_score analysis ────────────────────────────────────────────────────

def gem_score_analysis(picks: pd.DataFrame) -> dict:
    """
    For all candidate picks (P_side > min_bet etc), check if gem_score buckets
    have different WR. If positive correlation → use gem_score as filter.
    """
    if picks.empty:
        return {"n_picks": 0}

    # Bucket by gem_score
    buckets = pd.cut(
        picks["gem_score"],
        bins=[-1, -0.05, 0.0, 0.05, 0.10, 0.20, 1.0],
        labels=["<-5%", "-5..0", "0..5%", "5..10%", "10..20%", ">20%"],
    )
    bucket_stats = picks.groupby(buckets, observed=True).agg(
        n=("won", "size"),
        wr=("won", "mean"),
        avg_odds=("side_odds", "mean"),
        avg_p_side=("p_side", "mean"),
        avg_market_p=("market_p_side", "mean"),
    ).reset_index().rename(columns={"gem_score": "bucket"})

    # Correlation
    if len(picks) >= 30:
        corr = float(picks[["gem_score", "won"]].corr().iloc[0, 1])
    else:
        corr = None

    # Compare: top quartile gem_score vs bottom quartile WR
    if len(picks) >= 40:
        q75 = picks["gem_score"].quantile(0.75)
        q25 = picks["gem_score"].quantile(0.25)
        top = picks[picks["gem_score"] >= q75]
        bot = picks[picks["gem_score"] <= q25]
        top_wr = float(top["won"].mean())
        bot_wr = float(bot["won"].mean())
    else:
        top_wr = bot_wr = None

    return {
        "n_picks": len(picks),
        "buckets": bucket_stats.to_dict(orient="records"),
        "corr_gem_score_won": corr,
        "wr_top_quartile_gem": top_wr,
        "wr_bottom_quartile_gem": bot_wr,
    }


# ── 2. filter sweep ──────────────────────────────────────────────────────────

def sweep_filter(
    proba: np.ndarray,
    info: pd.DataFrame,
    y: np.ndarray,
    covered: np.ndarray,
    p_bet_grid:    list[float] = [0.60, 0.65, 0.70, 0.72, 0.75, 0.80],
    max_draw_grid: list[float] = [0.25, 0.28, 0.30, 0.32, 0.35],
    min_odds_grid: list[float] = [1.40, 1.50, 1.60],
    gem_score_grid: list[float] = [None, 0.0, 0.05, 0.10],
) -> pd.DataFrame:
    """
    Cartesian sweep. Returns a DataFrame of (config, n_bets, wr, roi).
    Each ROI uses simple flat-stake (not bankroll-compounding) for speed.
    """
    rows = []
    for p_bet in p_bet_grid:
        for mxd in max_draw_grid:
            for mino in min_odds_grid:
                for gs in gem_score_grid:
                    picks = build_picks(
                        proba, info, y, covered,
                        max_draw=mxd, min_bet=p_bet, min_odds=mino,
                        gem_score_thr=gs,
                    )
                    n = len(picks)
                    if n == 0:
                        continue
                    wr = float(picks["won"].mean())
                    pnl = float(((picks["side_odds"] - 1) * picks["won"] - (1 - picks["won"])).sum())
                    roi = pnl / n
                    rows.append({
                        "p_bet": p_bet, "max_draw": mxd, "min_odds": mino,
                        "gem_score_thr": gs if gs is not None else "off",
                        "n_bets": n, "wr": round(wr, 3), "roi": round(roi, 4),
                        "avg_odds": round(float(picks["side_odds"].mean()), 2),
                    })
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)


# ── 3. per-cluster + per-league breakdown ────────────────────────────────────

# Picks DF carries canonical "Country: Name" league_name (see niches.to_canonical).
# Use the shared classifier so we don't drift from TARGET_LEAGUES / TOP5_UCL.
from model.gem.niches import league_cluster as _league_cluster_fn  # noqa: E402


def per_cluster(picks: pd.DataFrame) -> dict:
    if picks.empty:
        return {"n": 0}
    picks = picks.copy()
    picks["cluster"] = picks["league_name"].apply(
        lambda l: "top5_ucl" if _league_cluster_fn(l) == "top5_ucl" else "second_tier"
    )
    cluster_stats = picks.groupby("cluster").agg(
        n=("won", "size"),
        wr=("won", "mean"),
        avg_odds=("side_odds", "mean"),
        avg_gem_score=("gem_score", "mean"),
    ).reset_index()
    league_stats = picks.groupby("league_name").agg(
        n=("won", "size"),
        wr=("won", "mean"),
        avg_odds=("side_odds", "mean"),
    ).reset_index().sort_values("n", ascending=False)
    return {
        "by_cluster": cluster_stats.to_dict(orient="records"),
        "by_league":  league_stats.to_dict(orient="records"),
    }


# ── 4. feature signatures (WIN vs LOSS) ──────────────────────────────────────

def feature_signatures(
    picks: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    info_full: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    For each feature, compute mean value among WIN picks vs LOSS picks
    plus standardised effect size (Cohen's d). Top-k by |d|.
    """
    if picks.empty:
        return pd.DataFrame()

    # Map picks back to original X via match_id
    info_idx = pd.Series(info_full.index, index=info_full["match_id"])
    sel_idx = info_idx.loc[picks["match_id"]].values
    Xp = X[sel_idx]
    won = picks["won"].values

    n_win = won.sum()
    n_loss = len(won) - n_win
    if n_win < 5 or n_loss < 5:
        return pd.DataFrame()

    rows = []
    for j, fname in enumerate(feature_names):
        col = Xp[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        win_vals  = col[valid & (won == 1)]
        loss_vals = col[valid & (won == 0)]
        if len(win_vals) < 3 or len(loss_vals) < 3:
            continue
        mean_w, mean_l = win_vals.mean(), loss_vals.mean()
        var_w, var_l = win_vals.var(ddof=1), loss_vals.var(ddof=1)
        pooled_sd = np.sqrt(((len(win_vals) - 1) * var_w + (len(loss_vals) - 1) * var_l) /
                            (len(win_vals) + len(loss_vals) - 2))
        d = (mean_w - mean_l) / pooled_sd if pooled_sd > 1e-9 else 0.0
        rows.append({
            "feature": fname,
            "mean_win":  round(float(mean_w), 4),
            "mean_loss": round(float(mean_l), 4),
            "delta":     round(float(mean_w - mean_l), 4),
            "cohens_d":  round(float(d), 3),
        })
    return (
        pd.DataFrame(rows)
        .assign(abs_d=lambda x: x["cohens_d"].abs())
        .sort_values("abs_d", ascending=False)
        .head(top_k)
        .drop(columns=["abs_d"])
        .reset_index(drop=True)
    )


# ── 5. market vs model calibration comparison ────────────────────────────────

def market_vs_model(proba: np.ndarray, info: pd.DataFrame, y: np.ndarray, covered: np.ndarray) -> dict:
    """
    On covered rows: compute log-loss for our model vs market-implied probs.
    If model log-loss > market log-loss, market is better calibrated overall.
    """
    df = info.reset_index(drop=True).copy()
    df["covered"] = covered
    df["y"] = y
    df = df[df["covered"]].reset_index(drop=True)

    mp_rows = [
        market_probs_from_odds(h, d, a)
        for h, d, a in zip(df["home_odds"], df["draw_odds"], df["away_odds"])
    ]
    market = pd.DataFrame(mp_rows)
    has_market = market["home"].notna()
    df = df[has_market].reset_index(drop=True)
    market = market[has_market].reset_index(drop=True)

    if df.empty:
        return {"error": "No market odds available on covered rows"}

    # Match proba to filtered df
    proba_cov = proba[covered][has_market]

    # Log-loss per row
    eps = 1e-9
    y_arr = df["y"].values
    market_arr = np.stack([market["home"], market["draw"], market["away"]], axis=1)
    model_ll  = -np.log(np.clip(proba_cov[np.arange(len(y_arr)), y_arr], eps, 1))
    market_ll = -np.log(np.clip(market_arr[np.arange(len(y_arr)), y_arr], eps, 1))

    return {
        "n":              int(len(df)),
        "model_log_loss":  float(model_ll.mean()),
        "market_log_loss": float(market_ll.mean()),
        "model_minus_market": float(model_ll.mean() - market_ll.mean()),
        "model_better_pct": float((model_ll < market_ll).mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(use_raw: bool = False) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading OOF + info from artifacts …")
    d = load_oof()

    proba = d["oof_raw"] if use_raw else d["oof_cal"]
    logger.info(f"Using {'RAW' if use_raw else 'CALIBRATED'} OOF probabilities")
    info = d["info"]
    y = d["y"]
    covered = d["covered"]

    # Need full X for feature signatures — rebuild
    logger.info("Rebuilding feature matrix for signatures (cheap) …")
    from model.gem.data import load_historical
    from model.gem.feature_matrix import build_feature_matrix, inject_league_priors
    from model.gem.preprocessing import LeagueTargetEncoder
    from model.gem.team_state import build_h2h, build_team_state
    raw = load_historical()
    ts = build_team_state(raw["matches"], raw["stats"])
    h2h_dict = build_h2h(raw["matches"])
    X, _, _ = build_feature_matrix(
        raw["matches"], raw["stats"], raw["odds"], raw["injuries"], ts, h2h_dict,
    )
    enc = LeagueTargetEncoder.load(ARTIFACTS_DIR / "league_encoder.pkl")
    X = inject_league_priors(X, info, enc, d["feature_names"])

    # Build the canonical pick set with current default filter
    picks = build_picks(proba, info, y, covered)
    logger.info(f"Default filter → {len(picks)} picks")

    # Run analyses
    report: dict = {}
    report["default_filter"] = {
        "n_picks": int(len(picks)),
        "wr":      float(picks["won"].mean()) if len(picks) else None,
        "avg_odds": float(picks["side_odds"].mean()) if len(picks) else None,
        "roi":     float(((picks["side_odds"] - 1) * picks["won"] - (1 - picks["won"])).mean()) if len(picks) else None,
    }

    logger.info("[1/5] gem_score analysis …")
    report["gem_score"] = gem_score_analysis(picks)

    logger.info("[2/5] filter sweep …")
    sweep = sweep_filter(proba, info, y, covered)
    report["sweep_top20"] = sweep.head(20).to_dict(orient="records")
    sweep.to_csv(REPORTS_DIR / "filter_sweep.csv", index=False)

    logger.info("[3/5] per-cluster + per-league …")
    report["per_cluster"] = per_cluster(picks)

    logger.info("[4/5] feature signatures (WIN vs LOSS) …")
    sig = feature_signatures(picks, X, d["feature_names"], info)
    report["feature_signatures"] = sig.to_dict(orient="records")

    logger.info("[5/5] market vs model log-loss …")
    report["market_vs_model"] = market_vs_model(proba, info, y, covered)

    suffix = "_raw" if use_raw else "_cal"
    out_path = REPORTS_DIR / f"analysis{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved → {out_path}")
    return report


if __name__ == "__main__":
    import sys
    use_raw = "--raw" in sys.argv
    run(use_raw=use_raw)
