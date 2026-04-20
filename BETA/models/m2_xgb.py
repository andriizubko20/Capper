"""
BETA/models/m2_xgb.py

M2 — XGBoost multiclass + Isotonic calibration.
More regularized than M1, different calibration method.
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class M2XGBoost:
    """
    XGBoost via native API + isotonic calibration per class.
    """

    def __init__(
        self,
        n_estimators:     int   = 400,
        learning_rate:    float = 0.05,
        max_depth:        int   = 4,
        min_child_weight: int   = 10,
        subsample:        float = 0.75,
        colsample_bytree: float = 0.75,
        gamma:            float = 0.1,
        reg_alpha:        float = 0.5,
        reg_lambda:       float = 1.0,
        random_state:     int   = 42,
        calibrate:        bool  = True,
        calibration_frac: float = 0.20,
    ):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_child_weight  = min_child_weight
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.gamma             = gamma
        self.reg_alpha         = reg_alpha
        self.reg_lambda        = reg_lambda
        self.random_state      = random_state
        self.calibrate         = calibrate
        self.calibration_frac  = calibration_frac

        self._booster    = None
        self._iso        = None   # isotonic per class (list of 3)
        self.feature_importances_ = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        cal_size = max(50, int(n * self.calibration_frac))
        X_train, X_cal = X[:-cal_size], X[-cal_size:]
        y_train, y_cal = y[:-cal_size], y[-cal_size:]

        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            'objective':        'multi:softprob',
            'num_class':        3,
            'max_depth':        self.max_depth,
            'learning_rate':    self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample':        self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma':            self.gamma,
            'alpha':            self.reg_alpha,
            'lambda':           self.reg_lambda,
            'seed':             self.random_state,
            'nthread':          -1,
            'eval_metric':      'mlogloss',
            'verbosity':        0,
        }
        self._booster = xgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )
        self.feature_importances_ = np.array([
            self._booster.get_score(importance_type='gain').get(f'f{i}', 0.0)
            for i in range(X.shape[1])
        ])

        # Isotonic calibration per class
        if self.calibrate and len(np.unique(y_cal)) == 3:
            dcal   = xgb.DMatrix(X_cal)
            raw    = self._booster.predict(dcal).reshape(-1, 3)
            self._iso = []
            for c in range(3):
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(raw[:, c], (y_cal == c).astype(int))
                self._iso.append(ir)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        raw  = self._booster.predict(dmat).reshape(-1, 3)
        if self._iso is not None:
            cal = np.column_stack([self._iso[c].predict(raw[:, c]) for c in range(3)])
            # re-normalize
            row_sum = cal.sum(axis=1, keepdims=True)
            return cal / np.where(row_sum == 0, 1, row_sum)
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
