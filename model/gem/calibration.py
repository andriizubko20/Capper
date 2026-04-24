"""
model/gem/calibration.py

Time-aware multi-class isotonic calibration.

Fits one IsotonicRegression per class (OvR) on the last chronological
slice of the data (tail_frac, default 15%), then normalises outputs to
sum to 1.

Why time-aware: calibrate against the most recent distribution, which
better matches what the model will face in production than the full
historical distribution (older, noisier matches).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class GemCalibrator:
    """Per-class OvR isotonic regression with renormalisation."""

    def __init__(self, tail_frac: float = 0.15):
        self.tail_frac = tail_frac
        self._regs: list[IsotonicRegression] = [
            IsotonicRegression(out_of_bounds="clip") for _ in range(3)
        ]
        self._fitted = False

    def fit(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        dates: pd.Series | None = None,
    ) -> "GemCalibrator":
        """
        proba: (n, 3) raw probabilities from ensemble.
        y: integer labels 0=H, 1=D, 2=A.
        dates: if provided, fit on last tail_frac of data by date; otherwise
               on last tail_frac of rows (assumes chronological order).
        """
        n = len(proba)
        if dates is not None:
            dates = pd.to_datetime(dates).reset_index(drop=True)
            cutoff = dates.quantile(1.0 - self.tail_frac)
            mask = (dates >= cutoff).to_numpy()
        else:
            cutoff_idx = int(n * (1.0 - self.tail_frac))
            mask = np.zeros(n, dtype=bool)
            mask[cutoff_idx:] = True

        p_tail = proba[mask]
        y_tail = y[mask]
        logger.info(f"Calibrating on {int(mask.sum()):,} / {n:,} tail samples")

        for cls in range(3):
            self._regs[cls].fit(p_tail[:, cls], (y_tail == cls).astype(float))

        self._fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() first"
        cal = np.stack(
            [self._regs[cls].predict(proba[:, cls]) for cls in range(3)],
            axis=1,
        ).clip(0.0, 1.0)
        row_sums = cal.sum(axis=1, keepdims=True).clip(1e-9)
        return cal / row_sums

    def fit_transform(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        dates: pd.Series | None = None,
    ) -> np.ndarray:
        return self.fit(proba, y, dates).transform(proba)

    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or ARTIFACTS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "calibrator.pkl")
        logger.info(f"Calibrator saved to {d}/calibrator.pkl")

    @classmethod
    def load(cls, directory: Path | None = None) -> "GemCalibrator":
        d = Path(directory or ARTIFACTS_DIR)
        return joblib.load(d / "calibrator.pkl")
