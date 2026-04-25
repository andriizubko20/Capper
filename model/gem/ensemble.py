"""
model/gem/ensemble.py

3-model stacking ensemble: XGBoost + LightGBM + CatBoost → L2 logistic meta.

Training flow:
  1. tune_hyperparams() — Optuna on last 3 walk-forward folds per model
  2. build_oof()        — Full walk-forward OOF predictions (tuned params,
                          per-fold league target encoding to prevent leakage)
  3. fit_meta()         — L2 logistic regression on OOF stack (9 features)
  4. fit_final()        — Refit base models on ALL data with tuned params

League priors (3 features per row) are RECOMPUTED per fold from training
data only — this is done inside build_oof() and tune_hyperparams().
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from model.gem.cross_val import WalkForwardSplit, walk_forward_splits
from model.gem.feature_matrix import inject_league_priors
from model.gem.preprocessing import LeagueTargetEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_NAMES = ("xgb", "lgb", "cat")


# ── Model factories ──────────────────────────────────────────────────────────

def _make_xgb(params: dict):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
        **params,
    )


def _make_lgb(params: dict):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        verbosity=-1,
        random_state=42,
        **params,
    )


def _make_cat(params: dict):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        bootstrap_type="Bernoulli",  # required when using `subsample`
        silent=True,
        random_seed=42,
        **params,
    )


_MAKERS = {"xgb": _make_xgb, "lgb": _make_lgb, "cat": _make_cat}


# ── Hyperparameter search spaces ─────────────────────────────────────────────

def _xgb_space(trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0),
    }


def _lgb_space(trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 127),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def _cat_space(trial) -> dict:
    # Note: using Bernoulli bootstrap (set in _make_cat) to enable `subsample`.
    # `bagging_temperature` is Bayesian-only, so excluded.
    return {
        "iterations":    trial.suggest_int("iterations", 200, 800),
        "depth":         trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
    }


_SPACES = {"xgb": _xgb_space, "lgb": _lgb_space, "cat": _cat_space}


def _fit(name: str, model, X_tr, y_tr, X_val=None, y_val=None):
    """Early-stopping-aware fit. When X_val is None, disable early stopping."""
    if name == "xgb":
        if X_val is not None:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # Recreate without early stopping; sklearn metadata needs clean init
            params = model.get_params()
            params.pop("early_stopping_rounds", None)
            import xgboost as xgb
            fresh = xgb.XGBClassifier(**params)
            fresh.fit(X_tr, y_tr)
            # Mutate passed-in model so caller keeps the reference
            model.__dict__.update(fresh.__dict__)
    elif name == "lgb":
        if X_val is not None:
            import lightgbm as lgb_mod
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb_mod.early_stopping(30, verbose=False)],
            )
        else:
            model.fit(X_tr, y_tr)
    else:  # catboost
        model.fit(X_tr, y_tr)


# ── Per-fold league encoding ──────────────────────────────────────────────────

def _fold_encode(
    X: np.ndarray,
    y: np.ndarray,
    info: pd.DataFrame,
    feature_names: list[str],
    split: WalkForwardSplit,
) -> tuple[np.ndarray, np.ndarray, LeagueTargetEncoder]:
    """
    Fits LeagueTargetEncoder on the fold's train portion, injects priors into
    BOTH train and val rows, returns (X_with_priors, y, encoder).
    """
    enc = LeagueTargetEncoder().fit(
        info["league_name"].iloc[split.train_idx],
        y[split.train_idx],
    )
    X_inj = inject_league_priors(X, info, enc, feature_names)
    return X_inj, y, enc


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_hyperparams(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    info: pd.DataFrame,
    feature_names: list[str],
    splits: list[WalkForwardSplit],
    n_trials: int = 200,
    n_tuning_folds: int = 3,
) -> dict:
    """Tune hyperparams for one base model on the last n_tuning_folds."""
    tuning_splits = splits[-n_tuning_folds:]

    # Precompute per-fold X (with priors) once — speeds up all trials
    precomputed = []
    for s in tuning_splits:
        X_f, y_f, _ = _fold_encode(X, y, info, feature_names, s)
        precomputed.append((
            X_f[s.train_idx], y_f[s.train_idx],
            X_f[s.val_idx],   y_f[s.val_idx],
        ))

    def objective(trial):
        params = _SPACES[model_name](trial)
        losses = []
        for X_tr, y_tr, X_val, y_val in precomputed:
            model = _MAKERS[model_name](params)
            _fit(model_name, model, X_tr, y_tr, X_val, y_val)
            losses.append(log_loss(y_val, model.predict_proba(X_val)))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=2, show_progress_bar=False)
    logger.info(f"[{model_name}] best log-loss={study.best_value:.4f} over {n_trials} trials")
    return study.best_params


# ── OOF build (full walk-forward) ─────────────────────────────────────────────

def build_oof(
    model_name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    info: pd.DataFrame,
    feature_names: list[str],
    splits: list[WalkForwardSplit],
) -> np.ndarray:
    """
    Full walk-forward OOF with per-fold priors. Rows not covered by any fold
    (early history) remain at uniform [1/3, 1/3, 1/3].
    """
    oof = np.full((len(X), 3), 1 / 3.0)
    for split in splits:
        X_f, y_f, _ = _fold_encode(X, y, info, feature_names, split)
        X_tr, y_tr = X_f[split.train_idx], y_f[split.train_idx]
        X_val, y_val = X_f[split.val_idx], y_f[split.val_idx]

        model = _MAKERS[model_name](params)
        _fit(model_name, model, X_tr, y_tr, X_val, y_val)
        oof[split.val_idx] = model.predict_proba(X_val)
        loss = log_loss(y_val, oof[split.val_idx])
        logger.debug(f"[{model_name}] fold {split.fold}: log-loss={loss:.4f} (n_val={len(split.val_idx)})")
    return oof


# ── Main ensemble class ───────────────────────────────────────────────────────

class GemEnsemble:
    """
    Stacking: XGBoost + LightGBM + CatBoost → L2 logistic meta-learner.
    """

    def __init__(self):
        self.params: dict[str, dict] = {}
        self.base_models: dict[str, object] = {}
        self.meta_model: LogisticRegression | None = None
        self.feature_names: list[str] = []
        self.final_encoder: LeagueTargetEncoder | None = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        info: pd.DataFrame,
        feature_names: list[str],
        n_optuna_trials: int = 200,
        n_cv_folds: int = 12,
        params_override: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Full training flow. Returns dict of OOF arrays per model.

        params_override: dict of {model_name: params}. If provided, skip Optuna
            tuning and use these params directly. Used for fast OOF rebuilds.
        """
        self.feature_names = list(feature_names)
        splits = walk_forward_splits(info["date"], n_folds=n_cv_folds)
        if not splits:
            raise RuntimeError("Walk-forward CV returned no folds — check data range")

        oof_arrays: dict[str, np.ndarray] = {}

        for name in MODEL_NAMES:
            if params_override and name in params_override:
                logger.info(f"─ Using saved params for {name}, skipping Optuna ─")
                self.params[name] = params_override[name]
            else:
                logger.info(f"─ Tuning {name} ({n_optuna_trials} Optuna trials) ─")
                self.params[name] = tune_hyperparams(
                    name, X, y, info, feature_names, splits, n_optuna_trials,
                )
            logger.info(f"─ OOF for {name} ({len(splits)} folds) ─")
            oof_arrays[name] = build_oof(
                name, self.params[name], X, y, info, feature_names, splits,
            )

        oof_stack = np.hstack([oof_arrays[n] for n in MODEL_NAMES])

        covered = np.zeros(len(X), dtype=bool)
        for s in splits:
            covered[s.val_idx] = True

        logger.info(f"Fitting L2 meta-learner on {int(covered.sum())} OOF rows …")
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_model.fit(oof_stack[covered], y[covered])

        # Fit final encoder on ALL training data, inject into X, refit base models
        logger.info("Fitting final LeagueTargetEncoder on full dataset …")
        self.final_encoder = LeagueTargetEncoder().fit(info["league_name"], y)
        X_final = inject_league_priors(X, info, self.final_encoder, feature_names)

        for name in MODEL_NAMES:
            logger.info(f"Refitting {name} on full dataset …")
            model = _MAKERS[name](self.params[name])
            _fit(name, model, X_final, y)
            self.base_models[name] = model

        return oof_arrays

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Stacked (n, 3) probability matrix. X must already contain league priors
        injected via the final encoder (use predict_proba_from_info for convenience).
        """
        stack = np.hstack([self.base_models[n].predict_proba(X) for n in MODEL_NAMES])
        return self.meta_model.predict_proba(stack)

    def predict_proba_from_info(
        self,
        X_raw: np.ndarray,
        info: pd.DataFrame,
    ) -> np.ndarray:
        """Injects league priors using the final encoder, then predicts."""
        X = inject_league_priors(X_raw, info, self.final_encoder, self.feature_names)
        return self.predict_proba(X)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: Path | None = None) -> None:
        import joblib
        d = Path(directory or ARTIFACTS_DIR)
        d.mkdir(parents=True, exist_ok=True)

        self.base_models["xgb"].get_booster().save_model(str(d / "xgb_model.json"))
        self.base_models["lgb"].booster_.save_model(str(d / "lgb_model.txt"))
        self.base_models["cat"].save_model(str(d / "cat_model.cbm"))
        joblib.dump(self.meta_model, d / "meta_model.pkl")
        joblib.dump(self.final_encoder, d / "league_encoder.pkl")

        with open(d / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        with open(d / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Ensemble saved to {d}/")

    @classmethod
    def load(cls, directory: Path | None = None) -> "GemEnsemble":
        import joblib
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        d = Path(directory or ARTIFACTS_DIR)
        obj = cls()

        xgb_booster = xgb.Booster()
        xgb_booster.load_model(str(d / "xgb_model.json"))
        obj.base_models["xgb"] = _XGBBoosterShim(xgb_booster)

        obj.base_models["lgb"] = _LGBBoosterShim(lgb.Booster(model_file=str(d / "lgb_model.txt")))

        cat_m = CatBoostClassifier()
        cat_m.load_model(str(d / "cat_model.cbm"))
        obj.base_models["cat"] = cat_m

        obj.meta_model = joblib.load(d / "meta_model.pkl")
        obj.final_encoder = joblib.load(d / "league_encoder.pkl")

        with open(d / "params.json") as f:
            obj.params = json.load(f)
        with open(d / "feature_names.json") as f:
            obj.feature_names = json.load(f)

        logger.info(f"Ensemble loaded from {d}/")
        return obj


class _LGBBoosterShim:
    """Wraps a raw lgb.Booster to expose predict_proba()."""

    def __init__(self, booster):
        self._b = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._b.predict(X)


class _XGBBoosterShim:
    """Wraps a raw xgb.Booster to expose predict_proba()."""

    def __init__(self, booster):
        self._b = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        return self._b.predict(xgb.DMatrix(X))
