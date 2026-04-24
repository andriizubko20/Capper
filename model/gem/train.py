"""
model/gem/train.py

Main Gem model training pipeline.

Flow:
  1. Load historical data (matches, stats, odds, injuries)
  2. Build team state + H2H (chronological, leakage-free)
  3. Assemble feature matrix
  4. Train stacking ensemble (XGB + LGB + CatBoost + L2 meta)
  5. Calibrate (isotonic, last 15% of data)
  6. Evaluate: ML metrics + baselines + gem bet simulation + SHAP
  7. Save artifacts + experiment JSON log

Run: python -m model.gem.train [--trials N] [--folds N]
"""
import argparse
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
    baseline_dummy,
    baseline_glicko_only,
    ml_metrics,
    report,
    report_simulation,
    shap_summary,
    simulate_gem_bets,
)
from model.gem.feature_matrix import build_feature_matrix
from model.gem.features import expected_feature_names
from model.gem.team_state import build_h2h, build_team_state

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def run(
    n_optuna_trials: int = 200,
    n_cv_folds: int = 12,
    calib_tail_frac: float = 0.15,
) -> dict:
    t0 = datetime.utcnow()
    logger.info("=" * 72)
    logger.info(f"Gem training — {t0.isoformat()}Z")
    logger.info("=" * 72)

    # 1. Load data
    data = load_historical()
    matches = data["matches"]
    stats = data["stats"]
    odds = data["odds"]
    injuries = data["injuries"]

    # 2. Team state + H2H
    logger.info("Building team state and H2H …")
    ts = build_team_state(matches, stats)
    h2h = build_h2h(matches)

    # 3. Feature matrix (league priors left as NaN — filled per fold inside trainer)
    logger.info("Building feature matrix …")
    feature_names = expected_feature_names()
    X, y, info = build_feature_matrix(matches, stats, odds, injuries, ts, h2h)
    logger.info(f"Feature matrix: X={X.shape}, feature_names={len(feature_names)}")

    # 4. Train ensemble
    logger.info("Training stacking ensemble …")
    ensemble = GemEnsemble()
    oof_arrays = ensemble.train(
        X, y, info, feature_names,
        n_optuna_trials=n_optuna_trials,
        n_cv_folds=n_cv_folds,
    )

    # Stacked OOF probs (pre-calibration)
    splits = walk_forward_splits(info["date"], n_folds=n_cv_folds)
    covered = np.zeros(len(X), dtype=bool)
    for s in splits:
        covered[s.val_idx] = True

    oof_stack = np.hstack([oof_arrays[n] for n in MODEL_NAMES])
    raw_ens_oof = np.full((len(X), 3), 1 / 3.0)
    raw_ens_oof[covered] = ensemble.meta_model.predict_proba(oof_stack[covered])

    # 5. Calibrate on the tail of OOF predictions
    logger.info("Fitting calibrator on OOF tail …")
    calibrator = GemCalibrator(tail_frac=calib_tail_frac)
    calibrator.fit(raw_ens_oof[covered], y[covered], info["date"][covered])
    cal_ens_oof = raw_ens_oof.copy()
    cal_ens_oof[covered] = calibrator.transform(raw_ens_oof[covered])

    # 6. Evaluate
    logger.info("=" * 72)
    logger.info("EVALUATION")
    logger.info("=" * 72)

    metrics_all: dict = {}

    # Per-model OOF
    for name in MODEL_NAMES:
        m = ml_metrics(y[covered], oof_arrays[name][covered])
        metrics_all[name] = m
        logger.info("\n" + report(m, f"Base {name}"))

    # Raw ensemble
    m_raw = ml_metrics(y[covered], raw_ens_oof[covered])
    metrics_all["ensemble_raw"] = m_raw
    logger.info("\n" + report(m_raw, "Ensemble (raw)"))

    # Calibrated ensemble
    m_cal = ml_metrics(y[covered], cal_ens_oof[covered])
    metrics_all["ensemble_calibrated"] = m_cal
    logger.info("\n" + report(m_cal, "Ensemble (calibrated)"))

    # Baselines
    logger.info("Running baselines …")
    dummy_proba = baseline_dummy(y, covered)
    m_dummy = ml_metrics(y[covered], dummy_proba[covered])
    metrics_all["baseline_dummy"] = m_dummy
    logger.info("\n" + report(m_dummy, "Baseline: class-prior"))

    glicko_oof = baseline_glicko_only(X, y, feature_names, splits)
    m_glicko = ml_metrics(y[covered], glicko_oof[covered])
    metrics_all["baseline_glicko_only"] = m_glicko
    logger.info("\n" + report(m_glicko, "Baseline: Glicko-only LR"))

    # Overfit gap on train (using ensemble's final in-sample fit)
    train_proba = ensemble.predict_proba_from_info(X, info)
    m_train = ml_metrics(y, train_proba)
    overfit_gap = m_train["log_loss"] - m_cal["log_loss"]
    metrics_all["overfit_gap_logloss"] = float(overfit_gap)
    logger.info(
        f"\nOverfit gap (log-loss) = {m_train['log_loss']:.4f} (train) - "
        f"{m_cal['log_loss']:.4f} (val) = {overfit_gap:+.4f}"
    )

    # Gem bet simulation (calibrated OOF)
    logger.info("=" * 72)
    logger.info("GEM BET SIMULATION (calibrated OOF)")
    logger.info("=" * 72)
    sim = simulate_gem_bets(cal_ens_oof[covered], info[covered].reset_index(drop=True))
    logger.info("\n" + report_simulation(sim))
    metrics_all["simulation"] = {
        k: v for k, v in sim.items() if k not in ("bets_df", "weekly_df")
    }

    # SHAP on XGBoost (sample for speed)
    logger.info("Computing SHAP summary on XGBoost …")
    try:
        from model.gem.feature_matrix import inject_league_priors
        X_shap = inject_league_priors(X, info, ensemble.final_encoder, feature_names)
        sample_idx = np.random.RandomState(42).choice(
            len(X_shap), min(2000, len(X_shap)), replace=False,
        )
        shap_df = shap_summary(ensemble.base_models["xgb"], X_shap[sample_idx], feature_names)
        metrics_all["shap_top20"] = shap_df.to_dict(orient="records")
        logger.info("\nTop-20 features by mean |SHAP|:\n" + shap_df.to_string(index=False))
    except Exception as e:
        logger.warning(f"SHAP computation failed (non-fatal): {e}")
        metrics_all["shap_top20"] = None

    # 7. Save
    logger.info("=" * 72)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 72)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ensemble.save(ARTIFACTS_DIR)
    calibrator.save(ARTIFACTS_DIR)

    # Experiment log
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_tag = t0.strftime("%Y%m%d_%H%M%S")
    exp_path = EXPERIMENTS_DIR / f"exp_{ts_tag}.json"
    dt_sec = (datetime.utcnow() - t0).total_seconds()
    exp_log = {
        "timestamp_utc":    t0.isoformat() + "Z",
        "duration_minutes": round(dt_sec / 60, 1),
        "config": {
            "n_optuna_trials": n_optuna_trials,
            "n_cv_folds":      n_cv_folds,
            "calib_tail_frac": calib_tail_frac,
            "feature_count":   len(feature_names),
            "n_samples":       int(len(X)),
            "n_covered":       int(covered.sum()),
        },
        "best_params":   ensemble.params,
        "metrics":       metrics_all,
    }
    with open(exp_path, "w") as f:
        json.dump(exp_log, f, indent=2, default=str)
    logger.info(f"Experiment log saved → {exp_path}")

    return exp_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="Optuna trials per base model")
    ap.add_argument("--folds",  type=int, default=12,  help="Walk-forward CV folds")
    ap.add_argument("--tail",   type=float, default=0.15, help="Calibration tail fraction")
    args = ap.parse_args()
    run(n_optuna_trials=args.trials, n_cv_folds=args.folds, calib_tail_frac=args.tail)
