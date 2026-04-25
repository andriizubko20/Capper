"""
model/gem/shap_audit.py

Full SHAP ranking of every feature in saved Gem ensemble (v2 trained).
Identifies "dead weight" candidates — features with negligible mean |SHAP|.

Output:
  reports/shap_full_ranking.csv  — all 72 features ranked
  console: top 20 + bottom 20 + per-group totals
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from loguru import logger

from model.gem.data import load_historical
from model.gem.ensemble import GemEnsemble
from model.gem.feature_matrix import build_feature_matrix, inject_league_priors
from model.gem.team_state import build_h2h, build_team_state

ARTIFACTS = Path(__file__).parent / "artifacts"
REPORTS = Path(__file__).parent / "reports"

# Map feature → category for grouped reporting
FEATURE_GROUPS = {
    # Strength / Glicko
    "STRENGTH": [
        "home_glicko", "away_glicko", "glicko_gap",
        "glicko_home_prob", "glicko_away_prob", "glicko_draw_prob",
    ],
    # Raw xG
    "XG_RAW": [
        "home_xg_for", "home_xg_against", "home_xg_diff",
        "away_xg_for", "away_xg_against", "away_xg_diff",
        "xg_diff_gap",
    ],
    # xG context (home/away splits)
    "XG_SPLITS": [
        "home_xg_diff_at_home", "away_xg_diff_on_road", "context_xg_gap",
    ],
    # PPG / form
    "FORM": [
        "home_ppg", "away_ppg", "ppg_gap",
        "home_ppg_at_home", "away_ppg_on_road", "context_ppg_gap",
        "home_form_5", "away_form_5", "form_gap",
    ],
    # Momentum
    "MOMENTUM": [
        "home_xg_trend", "away_xg_trend",
        "home_win_streak", "home_lose_streak",
        "away_win_streak", "away_lose_streak",
        "home_glicko_momentum", "away_glicko_momentum", "glicko_momentum_gap",
    ],
    # Style
    "STYLE": [
        "home_possession", "away_possession",
        "home_sot", "away_sot",
        "home_pass_acc", "away_pass_acc",
    ],
    # H2H
    "H2H": [
        "h2h_home_wr", "h2h_avg_goals", "h2h_home_last_result",
    ],
    # Physical
    "PHYSICAL": [
        "home_rest_days", "away_rest_days", "rest_gap",
        "home_has_any_injuries", "away_has_any_injuries",
    ],
    # League
    "LEAGUE_CLUSTER": [
        "league_cluster_top5", "league_cluster_second",
    ],
    "LEAGUE_PRIORS": [
        "league_prior_home_wr", "league_prior_draw_rate", "league_prior_away_wr",
    ],
    # Composite v2
    "COMPOSITE_v2": [
        "dominance_score", "xg_quality_gap", "momentum_alignment_score",
        "home_advantage_factor", "away_hotness_signal",
    ],
}


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logger.info("Loading saved Gem ensemble …")
    ens = GemEnsemble.load(ARTIFACTS)

    logger.info("Building feature matrix …")
    data = load_historical()
    ts = build_team_state(data["matches"], data["stats"])
    h2h_dict = build_h2h(data["matches"])
    X, _, info = build_feature_matrix(
        data["matches"], data["stats"], data["odds"], data["injuries"], ts, h2h_dict,
    )
    X_inj = inject_league_priors(X, info, ens.final_encoder, ens.feature_names)

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_inj), min(2500, len(X_inj)), replace=False)
    X_sample = X_inj[sample_idx]
    logger.info(f"Sampled {len(X_sample)} rows for SHAP")

    logger.info("Computing SHAP on XGBoost (3-class) …")
    xgb_model = ens.base_models["xgb"]
    booster = xgb_model._b if hasattr(xgb_model, "_b") else xgb_model
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_sample)
    abs_vals = np.abs(sv)
    if abs_vals.ndim == 3:
        mean_abs = abs_vals.mean(axis=(0, 2))
    elif isinstance(sv, list):
        mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        mean_abs = abs_vals.mean(axis=0)

    df = pd.DataFrame({
        "feature":       ens.feature_names,
        "mean_abs_shap": mean_abs,
    })

    # Add category & one-hot indicator
    feature_to_cat = {}
    for cat, feats in FEATURE_GROUPS.items():
        for f in feats:
            feature_to_cat[f] = cat

    def categorize(f: str) -> str:
        if f in feature_to_cat:
            return feature_to_cat[f]
        if f.startswith("is_"):
            return "LEAGUE_ONEHOT"
        return "OTHER"

    df["category"] = df["feature"].apply(categorize)
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["mean_abs_shap"] = df["mean_abs_shap"].round(5)

    df.to_csv(REPORTS / "shap_full_ranking.csv", index=False)

    # ── Output ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"SHAP FULL RANKING — {len(df)} features (sample n={len(X_sample)})")
    print("=" * 90)

    print(f"\n  {'rank':>4s}  {'feature':>32s}  {'mean_abs_shap':>13s}  {'category':>14s}")
    print("  " + "-" * 70)
    print("\n  TOP 20 (driving the model):")
    for r in df.head(20).itertuples():
        print(f"  {r.rank:>4d}  {r.feature:>32s}  {r.mean_abs_shap:>13.5f}  {r.category:>14s}")

    print("\n  BOTTOM 20 (likely dead weight):")
    for r in df.tail(20).iloc[::-1].itertuples():
        print(f"  {r.rank:>4d}  {r.feature:>32s}  {r.mean_abs_shap:>13.5f}  {r.category:>14s}")

    print("\n" + "=" * 90)
    print("  PER-CATEGORY TOTAL SHAP IMPORTANCE")
    print("=" * 90)
    cat_stats = df.groupby("category").agg(
        n_features=("feature", "size"),
        total_shap=("mean_abs_shap", "sum"),
        avg_shap=("mean_abs_shap", "mean"),
    ).reset_index().sort_values("total_shap", ascending=False)
    print(f"  {'category':>16s}  {'n_features':>10s}  {'total_shap':>11s}  {'avg_shap':>11s}")
    for r in cat_stats.itertuples():
        print(f"  {r.category:>16s}  {r.n_features:>10d}  {r.total_shap:>11.4f}  {r.avg_shap:>11.5f}")

    # Dead weight: features with mean_abs_shap < 0.005
    dead = df[df["mean_abs_shap"] < 0.005]
    print("\n" + "=" * 90)
    print(f"  DEAD WEIGHT CANDIDATES (mean_abs_shap < 0.005): {len(dead)} features")
    print("=" * 90)
    if not dead.empty:
        for r in dead.itertuples():
            print(f"  {r.feature:>34s}  {r.mean_abs_shap:>10.5f}  {r.category}")

    logger.info(f"Saved → {REPORTS / 'shap_full_ranking.csv'}")


if __name__ == "__main__":
    run()
