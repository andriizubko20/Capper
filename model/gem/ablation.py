"""
model/gem/ablation.py

Drop 18 dead-weight features (per SHAP audit) and retrain with saved params.
Compare to v2 baseline.

Dead-weight: 14 league one-hots + 2 league_cluster + 2 has_any_injuries.
We KEEP league_priors (they do all the league-context work).

Output: console comparison v2 baseline vs pruned (ROI, WR, log-loss).
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
from model.gem.evaluate import (
    baseline_dummy, baseline_glicko_only, ml_metrics, report,
    report_simulation, simulate_gem_bets,
)
from model.gem.feature_matrix import build_feature_matrix
from model.gem.features import expected_feature_names
from model.gem.team_state import build_h2h, build_team_state

ARTIFACTS = Path(__file__).parent / "artifacts"
REPORTS   = Path(__file__).parent / "reports"

# Dead-weight features per SHAP audit (mean_abs_shap < 0.005, except keeping
# `home_advantage_factor` for now — it's in COMPOSITE_v2 group and may help indirectly).
LEAGUE_ONEHOT_TO_DROP = [
    "is_2_bundesliga", "is_allsvenskan", "is_bundesliga", "is_champions_league",
    "is_championship", "is_eliteserien", "is_eredivisie", "is_jupiler_pro_league",
    "is_la_liga", "is_ligue_1", "is_premier_league", "is_primeira_liga",
    "is_serie_a", "is_serie_b", "is_super_lig",
]

DEAD_WEIGHT = LEAGUE_ONEHOT_TO_DROP + [
    "league_cluster_top5", "league_cluster_second",
    "home_has_any_injuries", "away_has_any_injuries",
    "home_advantage_factor",
]


def run(n_cv_folds: int = 12, calib_tail_frac: float = 0.15) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = datetime.utcnow()
    logger.info("=" * 72)
    logger.info("ABLATION: drop dead-weight features and retrain")
    logger.info("=" * 72)

    # 1. Load saved params
    with open(ARTIFACTS / "params.json") as f:
        saved_params = json.load(f)
    logger.info(f"Loaded saved params for {list(saved_params)}")

    # 2. Build full feature matrix
    data = load_historical()
    ts = build_team_state(data["matches"], data["stats"])
    h2h_dict = build_h2h(data["matches"])

    full_names = expected_feature_names()
    X_full, y, info = build_feature_matrix(
        data["matches"], data["stats"], data["odds"], data["injuries"],
        ts, h2h_dict,
    )

    # Map names → column index
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    drop_idx = [name_to_idx[f] for f in DEAD_WEIGHT if f in name_to_idx]
    keep_idx = [i for i in range(len(full_names)) if i not in set(drop_idx)]
    keep_names = [full_names[i] for i in keep_idx]

    logger.info(f"Original features: {len(full_names)}")
    logger.info(f"Dropping {len(drop_idx)} dead-weight features")
    logger.info(f"Remaining: {len(keep_names)}")

    X_pruned = X_full[:, keep_idx]

    # 3. Train pruned ensemble using saved params (skip Optuna)
    logger.info("Training pruned ensemble with saved params (no Optuna) …")
    ens = GemEnsemble()
    oof_arrays = ens.train(
        X_pruned, y, info, keep_names,
        n_optuna_trials=0,  # not used
        n_cv_folds=n_cv_folds,
        params_override=saved_params,
    )

    # 4. Stacked OOF
    splits = walk_forward_splits(info["date"], n_folds=n_cv_folds)
    covered = np.zeros(len(X_pruned), dtype=bool)
    for s in splits:
        covered[s.val_idx] = True
    oof_stack = np.hstack([oof_arrays[n] for n in MODEL_NAMES])
    raw_ens_oof = np.full((len(X_pruned), 3), 1 / 3.0)
    raw_ens_oof[covered] = ens.meta_model.predict_proba(oof_stack[covered])

    # 5. Calibrate
    calibrator = GemCalibrator(tail_frac=calib_tail_frac)
    calibrator.fit(raw_ens_oof[covered], y[covered], info["date"][covered])
    cal_ens_oof = raw_ens_oof.copy()
    cal_ens_oof[covered] = calibrator.transform(raw_ens_oof[covered])

    # 6. Metrics
    print("\n" + "=" * 72)
    print("PRUNED MODEL — Metrics")
    print("=" * 72)
    m_raw = ml_metrics(y[covered], raw_ens_oof[covered])
    m_cal = ml_metrics(y[covered], cal_ens_oof[covered])
    print(report(m_raw, "Pruned ensemble (raw)"))
    print(report(m_cal, "Pruned ensemble (calibrated)"))

    print("\n" + "=" * 72)
    print("GEM BET SIMULATION (pruned model, calibrated)")
    print("=" * 72)
    sim = simulate_gem_bets(cal_ens_oof[covered], info[covered].reset_index(drop=True))
    print(report_simulation(sim))

    # 7. Compare to baseline (read from latest experiment JSON)
    baseline = None
    exps = sorted((REPORTS.parent / "experiments").glob("exp_*.json"))
    for exp_path in reversed(exps):
        try:
            with open(exp_path) as f:
                exp = json.load(f)
            if exp.get("config", {}).get("feature_count") == 72:
                baseline = exp
                logger.info(f"Comparing to baseline: {exp_path.name}")
                break
        except Exception:
            continue

    print("\n" + "=" * 72)
    print("v2 BASELINE vs PRUNED")
    print("=" * 72)
    if baseline:
        b_metrics = baseline.get("metrics", {})
        b_cal = b_metrics.get("ensemble_calibrated", {})
        b_raw = b_metrics.get("ensemble_raw", {})
        b_sim = b_metrics.get("simulation", {})
        rows = [
            ("log-loss raw",     b_raw.get("log_loss"), m_raw["log_loss"]),
            ("log-loss cal",     b_cal.get("log_loss"), m_cal["log_loss"]),
            ("AUC-ROC macro",    b_cal.get("auc_roc_macro"), m_cal["auc_roc_macro"]),
            ("AUC-PR macro",     b_cal.get("auc_pr_macro"),  m_cal["auc_pr_macro"]),
            ("N bets",           b_sim.get("n_bets"),    sim["n_bets"]),
            ("WR",               b_sim.get("wr"),        sim["wr"]),
            ("ROI",              b_sim.get("roi"),       sim["roi"]),
            ("Max drawdown",     b_sim.get("max_drawdown"), sim["max_drawdown"]),
            ("Sharpe weekly",    b_sim.get("sharpe_weekly"), sim["sharpe_weekly"]),
        ]
        print(f"  {'metric':>20s}  {'baseline (72f)':>14s}  {'pruned (54f)':>14s}  {'delta':>8s}")
        print("  " + "-" * 64)
        for name, b, p in rows:
            if b is None or p is None:
                print(f"  {name:>20s}  {'n/a':>14s}  {'n/a':>14s}")
                continue
            try:
                delta = p - b
                if isinstance(b, float) and abs(b) < 5:
                    print(f"  {name:>20s}  {b:>14.4f}  {p:>14.4f}  {delta:>+8.4f}")
                else:
                    print(f"  {name:>20s}  {b:>14}  {p:>14}  {delta:>+8}")
            except TypeError:
                print(f"  {name:>20s}  {b:>14}  {p:>14}")
    else:
        print("  ⚠️  No baseline experiment JSON found.")

    duration = (datetime.utcnow() - t0).total_seconds() / 60
    print(f"\nAblation runtime: {duration:.1f} min")


if __name__ == "__main__":
    run()
