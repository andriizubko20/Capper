"""
model/gem/preprocessing.py

League target encoder. Computes per-league historical outcome rates
(home_wr, draw_rate, away_wr) on the training fold only, then maps
any input row to its league's priors. Includes Bayesian smoothing
toward global rates so small leagues don't swing to 100%/0%.

Used to densify league signal beyond the 14 one-hot columns.
"""
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class LeagueTargetEncoder:
    """
    league_name → {home_wr, draw_rate, away_wr} from training data only.

    fit(league_series, y):
        Computes smoothed rates per league, weighted toward global rates
        based on sample size.

    transform_row(league_name):
        Returns dict for one league (unknown league → global rates).
    """

    def __init__(self, smoothing: int = 50):
        """smoothing: prior pseudocount. Higher = more shrinkage toward global."""
        self.smoothing = smoothing
        self.global_rates: dict[str, float] = {}
        self.league_rates: dict[str, dict[str, float]] = {}
        self._fitted = False

    def fit(self, league_names: pd.Series, y: np.ndarray) -> "LeagueTargetEncoder":
        """
        league_names: pd.Series[str] of length n
        y: array of 0=H, 1=D, 2=A labels
        """
        if len(league_names) != len(y):
            raise ValueError("league_names and y must align")

        league_names = pd.Series(league_names).reset_index(drop=True)
        y = np.asarray(y)

        self.global_rates = {
            "home_wr":   float((y == 0).mean()),
            "draw_rate": float((y == 1).mean()),
            "away_wr":   float((y == 2).mean()),
        }

        for lg in league_names.unique():
            mask = (league_names == lg).to_numpy()
            n = int(mask.sum())
            if n == 0:
                self.league_rates[lg] = dict(self.global_rates)
                continue
            raw = {
                "home_wr":   float((y[mask] == 0).mean()),
                "draw_rate": float((y[mask] == 1).mean()),
                "away_wr":   float((y[mask] == 2).mean()),
            }
            w = n / (n + self.smoothing)
            self.league_rates[lg] = {
                k: w * raw[k] + (1 - w) * self.global_rates[k]
                for k in ("home_wr", "draw_rate", "away_wr")
            }

        self._fitted = True
        logger.debug(
            f"LeagueTargetEncoder fit on {len(y)} samples → "
            f"{len(self.league_rates)} leagues "
            f"(global home_wr={self.global_rates['home_wr']:.3f})"
        )
        return self

    def transform_row(self, league_name: str) -> dict[str, float]:
        assert self._fitted, "call fit() first"
        return self.league_rates.get(league_name, self.global_rates)

    def transform(self, league_names: pd.Series) -> pd.DataFrame:
        """Returns DataFrame with columns home_wr, draw_rate, away_wr."""
        assert self._fitted, "call fit() first"
        rows = [self.transform_row(lg) for lg in league_names]
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LeagueTargetEncoder":
        return joblib.load(path)
