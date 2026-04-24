"""
model/gem/evaluate.py

Evaluation suite for Gem model:
  * ML metrics: log-loss, Brier per class, AUC-ROC OvR, AUC-PR per class,
                overfit_gap (train - val log-loss)
  * Baselines: DummyClassifier (class-prior), GlickoOnly (LR on 3 glicko probs)
  * Gem backtest simulation: applies P(draw)<MAX_DRAW, P(bet)>MIN_BET, odds bound
    filter; simulates chronological flat 4% stake on bankroll; reports WR, ROI,
    drawdown, Sharpe

Caveat on odds: 94% of historical odds rows were recorded post-match
(closing/final snapshot from SStats). ROI simulated with these is OPTIMISTIC
relative to opening odds a real bettor would face — expect 2-3pp downward
bias adjustment in production.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from model.gem.niches import MAX_DRAW_PROB, MIN_BET_PROB, MIN_ODDS, MAX_ODDS, FLAT_STAKE_FRAC


# ── ML metrics ───────────────────────────────────────────────────────────────

def ml_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Per-class Brier, AUC-PR, AUC-ROC; overall log-loss."""
    n_classes = proba.shape[1]
    out = {
        "log_loss": float(log_loss(y_true, proba)),
        "brier": {},
        "auc_roc": {},
        "auc_pr": {},
    }
    class_names = ["H", "D", "A"]
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        p_bin = proba[:, c]
        out["brier"][class_names[c]]   = float(brier_score_loss(y_bin, p_bin))
        out["auc_roc"][class_names[c]] = float(roc_auc_score(y_bin, p_bin))
        out["auc_pr"][class_names[c]]  = float(average_precision_score(y_bin, p_bin))
    out["auc_roc_macro"] = float(np.mean(list(out["auc_roc"].values())))
    out["auc_pr_macro"]  = float(np.mean(list(out["auc_pr"].values())))
    return out


def report(metrics: dict, name: str = "") -> str:
    """Pretty-print metrics dict."""
    lines = [f"── {name} ──" if name else ""]
    lines.append(f"  log-loss        : {metrics['log_loss']:.4f}")
    lines.append(f"  AUC-ROC macro   : {metrics['auc_roc_macro']:.4f}")
    lines.append(f"  AUC-PR  macro   : {metrics['auc_pr_macro']:.4f}")
    for cls in ("H", "D", "A"):
        lines.append(
            f"    [{cls}] brier={metrics['brier'][cls]:.4f}  "
            f"auc_roc={metrics['auc_roc'][cls]:.4f}  "
            f"auc_pr={metrics['auc_pr'][cls]:.4f}"
        )
    return "\n".join(lines)


# ── Baselines ────────────────────────────────────────────────────────────────

def baseline_dummy(y: np.ndarray, covered_mask: np.ndarray) -> np.ndarray:
    """Class-prior baseline (returns constant class probabilities)."""
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y), 1)), y)
    return dummy.predict_proba(np.zeros((len(y), 1)))


def baseline_glicko_only(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splits,
) -> np.ndarray:
    """
    Simple 3-class logistic regression on the 3 Glicko features only.
    Walk-forward OOF so it's a fair comparison with the ensemble.
    """
    idx = [
        feature_names.index("glicko_home_prob"),
        feature_names.index("glicko_away_prob"),
        feature_names.index("glicko_draw_prob"),
    ]
    X_g = X[:, idx]
    # Impute NaN with column mean from training set (per fold) and clip [0,1]
    oof = np.full((len(X), 3), 1 / 3.0)
    for s in splits:
        X_tr = X_g[s.train_idx]
        X_val = X_g[s.val_idx]
        col_mean = np.nanmean(X_tr, axis=0)
        X_tr = np.where(np.isnan(X_tr), col_mean, X_tr)
        X_val = np.where(np.isnan(X_val), col_mean, X_val)
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_tr, y[s.train_idx])
        oof[s.val_idx] = lr.predict_proba(X_val)
    return oof


# ── Betting simulation ───────────────────────────────────────────────────────

def simulate_gem_bets(
    proba: np.ndarray,
    info: pd.DataFrame,
    starting_bank: float = 1000.0,
    stake_frac: float = FLAT_STAKE_FRAC,
    max_draw_prob: float = MAX_DRAW_PROB,
    min_bet_prob: float = MIN_BET_PROB,
    min_odds: float = MIN_ODDS,
    max_odds: float = MAX_ODDS,
) -> dict:
    """
    Chronological flat-stake backtest of the gem filter.

    For each match: pick side = argmax(P(H), P(A)).
    Apply gem filter: P(D)<max_draw_prob, P(side)>min_bet_prob,
                      min_odds ≤ side_odds ≤ max_odds.
    Stake = stake_frac × current bankroll. Settle with actual result.

    Returns dict with summary stats and per-bet + per-week DataFrames.
    """
    df = info.reset_index(drop=True).copy()
    df["pH"] = proba[:, 0]
    df["pD"] = proba[:, 1]
    df["pA"] = proba[:, 2]
    df = df.sort_values("date").reset_index(drop=True)

    bank = starting_bank
    bets: list[dict] = []

    for _, m in df.iterrows():
        if m["pD"] >= max_draw_prob:
            continue
        side = "H" if m["pH"] >= m["pA"] else "A"
        p_side = m["pH"] if side == "H" else m["pA"]
        if p_side < min_bet_prob:
            continue
        odds = m["home_odds"] if side == "H" else m["away_odds"]
        if pd.isna(odds) or odds < min_odds or odds > max_odds:
            continue

        stake = stake_frac * bank
        won = (side == "H" and m["result"] == "H") or (side == "A" and m["result"] == "A")
        pnl = stake * (odds - 1) if won else -stake
        bank_after = bank + pnl

        bets.append({
            "date": m["date"], "league": m["league_name"], "side": side,
            "odds": odds, "p_side": p_side, "p_draw": m["pD"],
            "won": int(won), "stake": stake, "pnl": pnl, "bank_after": bank_after,
        })
        bank = bank_after

    if not bets:
        return {
            "n_bets": 0, "wr": None, "roi": None, "final_bank": starting_bank,
            "max_drawdown": 0.0, "sharpe_weekly": None,
            "bets_df": pd.DataFrame(), "weekly_df": pd.DataFrame(),
        }

    bets_df = pd.DataFrame(bets)
    total_staked = bets_df["stake"].sum()
    total_pnl = bets_df["pnl"].sum()

    # Max drawdown on bankroll curve
    peak = bets_df["bank_after"].cummax()
    drawdown = (bets_df["bank_after"] - peak) / peak
    max_dd = float(drawdown.min())

    # Weekly Sharpe (log returns)
    bets_df["week"] = bets_df["date"].dt.to_period("W").astype(str)
    weekly = bets_df.groupby("week").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        pnl=("pnl", "sum"),
        bank_end=("bank_after", "last"),
    ).reset_index()
    weekly["ret"] = weekly["pnl"] / (weekly["bank_end"] - weekly["pnl"]).clip(1e-9)
    sharpe = (
        float(weekly["ret"].mean() / weekly["ret"].std() * np.sqrt(52))
        if weekly["ret"].std() > 0 else None
    )

    summary = {
        "n_bets":        int(len(bets_df)),
        "wr":            float(bets_df["won"].mean()),
        "roi":           float(total_pnl / total_staked),
        "total_pnl":     float(total_pnl),
        "final_bank":    float(bets_df["bank_after"].iloc[-1]),
        "max_drawdown":  max_dd,
        "sharpe_weekly": sharpe,
        "avg_odds":      float(bets_df["odds"].mean()),
        "yield_per_wk":  float(len(bets_df) / max(weekly["week"].nunique(), 1)),
        "bets_df":       bets_df,
        "weekly_df":     weekly,
    }
    return summary


def report_simulation(sim: dict, starting_bank: float = 1000.0) -> str:
    if sim["n_bets"] == 0:
        return "⚠️  No bets passed the gem filter."
    roi_pct = sim["roi"] * 100
    wr_pct  = sim["wr"] * 100
    dd_pct  = sim["max_drawdown"] * 100
    sharpe  = f"{sim['sharpe_weekly']:.2f}" if sim["sharpe_weekly"] else "n/a"
    return (
        f"  N bets          : {sim['n_bets']:,}\n"
        f"  Avg odds        : {sim['avg_odds']:.2f}\n"
        f"  Bets / week     : {sim['yield_per_wk']:.1f}\n"
        f"  Win rate        : {wr_pct:.1f}%\n"
        f"  ROI             : {roi_pct:+.2f}%\n"
        f"  Final bank      : ${sim['final_bank']:,.0f} (from ${starting_bank:,.0f})\n"
        f"  Max drawdown    : {dd_pct:.1f}%\n"
        f"  Sharpe (weekly) : {sharpe}\n"
    )


# ── SHAP ─────────────────────────────────────────────────────────────────────

def shap_summary(xgb_model, X: np.ndarray, feature_names: list[str], top_k: int = 20) -> pd.DataFrame:
    """Returns a DataFrame of top-k features by mean |SHAP| over all classes."""
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # Older shap API: list per class
        arr = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # (n, k, classes) or (n, k)
        abs_vals = np.abs(shap_values)
        if abs_vals.ndim == 3:
            arr = abs_vals.mean(axis=(0, 2))
        else:
            arr = abs_vals.mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": arr})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)
