"""
BETA/models/m1_lgbm.py

M1 — LightGBM multiclass classifier + Platt calibration.
Predicts P(H), P(D), P(A) for each match.

Calibration via CalibratedClassifierCV (cv='prefit') applied
after initial training to improve probability sharpness.
"""
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


class M1LightGBM:
    """
    LightGBM via native API + Platt calibration (LogisticRegression on leaf outputs).
    Avoids sklearn compatibility issues with lgb sklearn wrapper.
    Interface: fit(X, y) / predict_proba(X) — same as sklearn.
    """

    def __init__(
        self,
        n_estimators:   int   = 500,
        learning_rate:  float = 0.03,
        num_leaves:     int   = 31,
        min_child_samples: int = 20,
        subsample:      float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha:      float = 0.1,
        reg_lambda:     float = 0.1,
        random_state:   int   = 42,
        calibrate:      bool  = True,
        calibration_frac: float = 0.20,
    ):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.num_leaves        = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.reg_alpha         = reg_alpha
        self.reg_lambda        = reg_lambda
        self.random_state      = random_state
        self.calibrate         = calibrate
        self.calibration_frac  = calibration_frac

        self._booster   = None
        self._calibrator = None   # LogisticRegression on cal set
        self.feature_importances_ = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        cal_size = max(50, int(n * self.calibration_frac))
        X_train, X_cal = X[:-cal_size], X[-cal_size:]
        y_train, y_cal = y[:-cal_size], y[-cal_size:]

        params = {
            'objective':        'multiclass',
            'num_class':        3,
            'num_leaves':       self.num_leaves,
            'learning_rate':    self.learning_rate,
            'min_child_samples': self.min_child_samples,
            'subsample':        self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha':        self.reg_alpha,
            'reg_lambda':       self.reg_lambda,
            'random_state':     self.random_state,
            'n_jobs':           -1,
            'verbose':          -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        self._booster = lgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        self.feature_importances_ = self._booster.feature_importance(importance_type='gain')

        # Platt calibration via LogisticRegression on held-out set
        if self.calibrate and len(np.unique(y_cal)) == 3:
            raw_cal = self._booster.predict(X_cal)   # (n_cal, 3)
            self._calibrator = LogisticRegression(max_iter=200, C=1.0)
            self._calibrator.fit(raw_cal, y_cal)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._booster.predict(X)   # (n, 3)
        if self._calibrator is not None:
            return self._calibrator.predict_proba(raw)
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
