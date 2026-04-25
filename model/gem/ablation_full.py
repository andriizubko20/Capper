"""
model/gem/ablation_full.py

Full ablation study — drop each feature group separately and measure impact.

Pipeline (per group):
  1. Build full feature matrix.
  2. Drop the target group's features.
  3. Train ensemble using saved params (no Optuna).
  4. Build OOF, calibrate, simulate gem bets.
  5. Compare to v2 baseline.

Output: model/gem/reports/ablation_full.csv + console table.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.gem.calibration import GemCalibrator
from model.gem.cross_val import walk_forward_splits
from model.gem.data import load_historical
from model.gem.ensemble import GemEnsemble, MODEL_NAMES
from model.gem.evaluate import ml_metrics, simulate_gem_bets
from model.gem.feature_matrix import build_feature_matrix
from model.gem.features import expected_feature_names
from model.gem.team_state import build_h2h, build_team_state

ARTIFACTS = Path(__file__).parent / "artifacts"
REPORTS   = Path(__file__).parent / "reports"

# Feature groups for ablation
GROUPS = {
    "STRENGTH": [
        "home_glicko", "away_glicko", "glicko_gap",
        "glicko_home_prob", "glicko_away_prob", "glicko_draw_prob",
    ],
    "XG_RAW": [
        "home_xg_for", "home_xg_against", "home_xg_diff",
        "away_xg_for", "away_xg_against", "away_xg_diff", "xg_diff_gap",
    ],
    "XG_SPLITS": [
        "home_xg_diff_at_home", "away_xg_diff_on_road", "context_xg_gap",
    ],
    "FORM": [
        "home_ppg", "away_ppg", "ppg_gap",
        "home_ppg_at_home", "away_ppg_on_road", "context_ppg_gap",
        "home_form_5", "away_form_5", "form_gap",
    ],
    "MOMENTUM": [
        "home_xg_trend", "away_xg_trend",
        "home_win_streak", "home_lose_streak",
        "away_win_streak", "away_lose_streak",
        "home_glicko_momentum", "away_glicko_momentum", "glicko_momentum_gap",
    ],
    "STYLE": [
        "home_possession", "away_possession",
        "home_sot", "away_sot",
        "home_pass_acc", "away_pass_acc",
    ],
    "H2H": [
        "h2h_home_wr", "h2h_avg_goals", "h2h_home_last_result",
    ],
    "PHYSICAL": [
        "home_rest_days", "away_rest_days", "rest_gap",
        "home_has_any_injuries", "away_has_any_injuries",
    ],
    "INJURIES_ONLY": [
        "home_has_any_injuries", "away_has_any_injuries",
    ],
    "REST_ONLY": [
        "home_rest_days", "away_rest_days", "rest_gap",
    ],
    "STREAKS_ONLY": [
        "home_win_streak", "home_lose_streak",
        "away_win_streak", "away_lose_streak",
    ],
    "LEAGUE_PRIORS": [
        "league_prior_home_wr", "league_prior_draw_rate", "league_prior_away_wr",
    ],
    "LEAGUE_ONEHOT_AND_CLUSTER": [
        "league_cluster_top5", "league_cluster_second",
        "is_2_bundesliga", "is_allsvenskan", "is_bundesliga", "is_champions_league",
        "is_championship", "is_eliteserien", "is_eredivisie", "is_jupiler_pro_league",
        "is_la_liga", "is_ligue_1", "is_premier_league", "is_primeira_liga",
        "is_serie_a", "is_serie_b", "is_super_lig",
    ],
    "COMPOSITE_v2": [
        "dominance_score", "xg_quality_gap", "momentum_alignment_score",
        "home_advantage_factor", "away_hotness_signal",
    ],
}


def evaluate_setup(
    X: np.ndarray, y: np.ndarray, info: pd.DataFrame, feature_names: list[str],
    saved_params: dict, n_cv_folds: int = 12, calib_tail_frac: float = 0.15,
) -> dict:
    """Train + OOF + calibrate + simulate. Return all metrics."""
    ens = GemEnsemble()
    oof_arrays = ens.train(
        X, y, info, feature_names,
        n_optuna_trials=0, n_cv_folds=n_cv_folds,
        params_override=saved_params,
    )
    splits = walk_forward_splits(info["date"], n_folds=n_cv_folds)
    covered = np.zeros(len(X), dtype=bool)
    for s in splits:
        covered[s.val_idx] = True
    oof_stack = np.hstack([oof_arrays[n] for n in MODEL_NAMES])
    raw = np.full((len(X), 3), 1 / 3.0)
    raw[covered] = ens.meta_model.predict_proba(oof_stack[covered])
    cal = raw.copy()
    calibrator = GemCalibrator(tail_frac=calib_tail_frac)
    cal[covered] = calibrator.fit_transform(raw[covered], y[covered], info["date"][covered])

    m_raw = ml_metrics(y[covered], raw[covered])
    m_cal = ml_metrics(y[covered], cal[covered])
    sim = simulate_gem_bets(cal[covered], info[covered].reset_index(drop=True))

    return {
        "log_loss_raw":  m_raw["log_loss"],
        "log_loss_cal":  m_cal["log_loss"],
        "auc_roc":       m_cal["auc_roc_macro"],
        "auc_pr":        m_cal["auc_pr_macro"],
        "n_bets":        sim["n_bets"],
        "wr":            sim["wr"],
        "roi":           sim["roi"],
        "max_dd":        sim["max_drawdown"],
        "sharpe":        sim["sharpe_weekly"],
    }


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = datetime.utcnow()
    logger.info("=" * 72)
    logger.info("FULL ABLATION STUDY")
    logger.info("=" * 72)

    with open(ARTIFACTS / "params.json") as f:
        saved_params = json.load(f)

    logger.info("Loading data + building feature matrix …")
    data = load_historical()
    ts = build_team_state(data["matches"], data["stats"])
    h2h_dict = build_h2h(data["matches"])
    full_names = expected_feature_names()
    X_full, y, info = build_feature_matrix(
        data["matches"], data["stats"], data["odds"], data["injuries"],
        ts, h2h_dict,
    )
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    # ── Baseline (no ablation) ─────────────────────────────────────────
    logger.info("Baseline run (all 72 features) …")
    baseline = evaluate_setup(X_full, y, info, full_names, saved_params)
    logger.info(
        f"  Baseline: WR={baseline['wr']:.3%}, ROI={baseline['roi']:+.3%}, "
        f"log-loss={baseline['log_loss_cal']:.4f}, sharpe={baseline['sharpe']:.2f}"
    )

    # ── Per-group ablation ─────────────────────────────────────────────
    rows = [{"group": "BASELINE", "n_dropped": 0, **baseline}]

    for group_name, group_features in GROUPS.items():
        drop_idx = [name_to_idx[f] for f in group_features if f in name_to_idx]
        if not drop_idx:
            logger.warning(f"  Skip {group_name}: no features matched")
            continue
        keep_idx = [i for i in range(len(full_names)) if i not in set(drop_idx)]
        keep_names = [full_names[i] for i in keep_idx]
        X_dropped = X_full[:, keep_idx]

        logger.info(f"  Drop {group_name} ({len(drop_idx)} features) …")
        try:
            res = evaluate_setup(X_dropped, y, info, keep_names, saved_params)
        except Exception as e:
            logger.error(f"  {group_name} failed: {e}")
            continue
        rows.append({"group": group_name, "n_dropped": len(drop_idx), **res})

        delta_ll = res["log_loss_cal"] - baseline["log_loss_cal"]
        delta_roi = res["roi"] - baseline["roi"]
        verdict = "🟢 helps drop" if delta_ll < -0.001 else (
            "🔴 hurts drop" if delta_ll > 0.005 else "🟡 neutral"
        )
        logger.info(
            f"    {group_name}: ll={res['log_loss_cal']:.4f} (Δ{delta_ll:+.4f}) "
            f"ROI={res['roi']:+.3%} (Δ{delta_roi:+.3%}) {verdict}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "ablation_full.csv", index=False)

    # ── Pretty console table ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FULL ABLATION RESULTS — drop each group, retrain with saved params")
    print("=" * 100)
    print(f"  {'group':>27s}  {'dropped':>7s}  {'log_loss':>9s}  {'WR':>6s}  {'ROI':>7s}  "
          f"{'sharpe':>7s}  {'maxDD':>7s}  {'n_bets':>7s}  verdict")
    print("  " + "-" * 100)

    base = df[df["group"] == "BASELINE"].iloc[0]
    for r in df.itertuples():
        if r.group == "BASELINE":
            verdict = "📌 ref"
        else:
            d_ll  = r.log_loss_cal - base["log_loss_cal"]
            d_roi = r.roi - base["roi"]
            d_shp = (r.sharpe or 0) - (base["sharpe"] or 0)
            verdict = (
                "🟢 helps to DROP" if d_ll < -0.001 and d_roi >= -0.005 else
                "🔴 HURTS to drop" if d_ll > 0.005 or d_roi < -0.015 else
                "🟡 neutral"
            )
        ll  = f"{r.log_loss_cal:.4f}"
        wr  = f"{r.wr:.1%}"  if pd.notna(r.wr)  else "n/a"
        roi = f"{r.roi:+.1%}" if pd.notna(r.roi) else "n/a"
        shp = f"{r.sharpe:.2f}" if pd.notna(r.sharpe) else "n/a"
        dd  = f"{r.max_dd:.1%}"  if pd.notna(r.max_dd)  else "n/a"
        nb  = f"{int(r.n_bets)}"
        print(
            f"  {r.group:>27s}  {r.n_dropped:>7d}  {ll:>9s}  {wr:>6s}  {roi:>7s}  "
            f"{shp:>7s}  {dd:>7s}  {nb:>7s}  {verdict}"
        )

    duration = (datetime.utcnow() - t0).total_seconds() / 60
    print(f"\nTotal runtime: {duration:.1f} min")
    logger.info(f"Saved → {REPORTS / 'ablation_full.csv'}")


if __name__ == "__main__":
    run()
