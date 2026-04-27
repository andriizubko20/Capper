"""
model/gem/calibration.py

Time-aware multi-class isotonic calibration with per-league heads.

Architecture:
  - One GLOBAL calibrator (3 isotonic regressors per outcome) — used as
    fallback when a league lacks calibration data.
  - One PER-LEAGUE calibrator for each league with >= MIN_PER_LEAGUE
    tail samples — captures league-specific probability drift (e.g. La
    Liga's home_wr differs from global avg).

Why per-league: a single global calibrator forces all 18 leagues to
share the same probability mapping. But each league has different
home-advantage / draw rate / model bias. Per-league heads close that
gap without adding training cost (just more tiny isotonic fits).

Why time-aware (tail_frac): calibrate against the most recent
distribution which better matches what the model will face in
production than the full historical distribution.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _new_regs() -> list[IsotonicRegression]:
    return [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]


def _fit_regs(regs, p_tail: np.ndarray, y_tail: np.ndarray) -> None:
    for cls in range(3):
        regs[cls].fit(p_tail[:, cls], (y_tail == cls).astype(float))


def _apply_regs(regs, p: np.ndarray) -> np.ndarray:
    return np.stack(
        [regs[cls].predict(p[:, cls]) for cls in range(3)], axis=1
    ).clip(0.0, 1.0)


class GemCalibrator:
    """Per-class OvR isotonic regression with optional per-league heads."""

    def __init__(self, tail_frac: float = 0.15, min_per_league: int = 30):
        self.tail_frac = tail_frac
        self.min_per_league = min_per_league
        self._global: list[IsotonicRegression] = _new_regs()
        self._by_league: dict[str, list[IsotonicRegression]] = {}
        self._league_n: dict[str, int] = {}
        self._fitted = False

    # ── fit ────────────────────────────────────────────────────────────

    def fit(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        dates: pd.Series | None = None,
        leagues: pd.Series | np.ndarray | None = None,
    ) -> "GemCalibrator":
        """
        proba: (n, 3) raw probabilities from ensemble.
        y: integer labels 0=H, 1=D, 2=A.
        dates: optional — chronological filter for tail.
        leagues: optional canonical league names per row. If provided,
                 fits a per-league head for any league with >=
                 min_per_league tail samples. Without leagues, only
                 global fit is performed (legacy behaviour).
        """
        n = len(proba)
        if dates is not None:
            dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
            cutoff = dates.quantile(1.0 - self.tail_frac)
            mask = (dates >= cutoff).to_numpy()
        else:
            cutoff_idx = int(n * (1.0 - self.tail_frac))
            mask = np.zeros(n, dtype=bool)
            mask[cutoff_idx:] = True

        p_tail = proba[mask]
        y_tail = y[mask]
        logger.info(f"Calibrating GLOBAL on {int(mask.sum()):,} / {n:,} tail samples")
        _fit_regs(self._global, p_tail, y_tail)

        if leagues is not None:
            leagues = np.asarray(leagues)
            leagues_tail = leagues[mask]
            self._by_league.clear()
            self._league_n.clear()
            for lg in pd.unique(leagues_tail):
                if pd.isna(lg) or lg in ("", "nan"):
                    continue
                lg_mask = leagues_tail == lg
                n_lg = int(lg_mask.sum())
                if n_lg < self.min_per_league:
                    logger.info(
                        f"  per-league SKIP {lg}: {n_lg} tail samples (< {self.min_per_league})"
                    )
                    continue
                p_lg = p_tail[lg_mask]
                y_lg = y_tail[lg_mask]
                regs = _new_regs()
                _fit_regs(regs, p_lg, y_lg)
                self._by_league[lg] = regs
                self._league_n[lg] = n_lg
                logger.info(f"  per-league HEAD  {lg}: {n_lg} samples")

        self._fitted = True
        return self

    # ── transform ──────────────────────────────────────────────────────

    def transform(
        self,
        proba: np.ndarray,
        leagues: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply per-league head where available; else global. Renormalise rows to 1."""
        assert self._fitted, "call fit() first"

        if leagues is None or not self._by_league:
            cal = _apply_regs(self._global, proba)
        else:
            leagues = np.asarray(leagues)
            cal = np.zeros_like(proba)
            uniq = pd.unique(leagues)
            for lg in uniq:
                lm = leagues == lg
                regs = self._by_league.get(lg, self._global)
                cal[lm] = _apply_regs(regs, proba[lm])

        row_sums = cal.sum(axis=1, keepdims=True).clip(1e-9)
        return cal / row_sums

    # ── helpers ────────────────────────────────────────────────────────

    def fit_transform(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        dates: pd.Series | None = None,
        leagues: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(proba, y, dates, leagues).transform(proba, leagues)

    def league_summary(self) -> dict[str, int]:
        """Returns {league_canonical: n_tail_samples} for diagnostics."""
        return dict(self._league_n)

    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or ARTIFACTS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "calibrator.pkl")
        n_heads = len(self._by_league)
        logger.info(
            f"Calibrator saved to {d}/calibrator.pkl "
            f"(global + {n_heads} per-league heads)"
        )

    @classmethod
    def load(cls, directory: Path | None = None) -> "GemCalibrator":
        d = Path(directory or ARTIFACTS_DIR)
        return joblib.load(d / "calibrator.pkl")
