"""
model/gem/calibration.py

Time-aware multi-class isotonic calibration with hierarchical heads.

Three-tier architecture (winner of grid search 2026-04-27, +7.5 u/yr):
  1. PER-LEAGUE head — when the league has >= MIN_PER_LEAGUE tail samples.
     Most accurate but easiest to overfit.
  2. CLUSTER head — when no per-league head, but league belongs to a
     cluster with enough pooled data. Cluster strategy:
       "kmeans-K": data-driven on (pH, pD, pA) tail profile per league.
                   Best per grid search (K=5 with mpl=50, tail=0.20).
       "tier":     semantic TOP5_UCL / SECOND_TIER from niches.py.
  3. GLOBAL head — final fallback.

At inference, each row dispatches to its most-specific available head.

Why hierarchical: small-sample leagues (Allsvenskan 18 tail, Eliteserien
25) get noisy isotonic curves on their own. Falling back to a cluster
of similar leagues — by playing-style profile, not just continent —
gives more stable calibration than the broad global fit.

Why time-aware (tail_frac): calibrate against the most recent
distribution which better matches what the model will face in
production than the full historical distribution.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

from model.gem.niches import league_cluster

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
    """Per-class OvR isotonic regression with hierarchical league > cluster > global heads.

    Default config (`tail_frac=0.15, min_per_league=30, cluster_strategy="none"`)
    matches the current gem_v1 production calibrator (+49.16 u/yr).

    Grid-search winner (Grid #2): `tail_frac=0.10, min_per_league=20,
    cluster_strategy="kmeans-3"` → +52.49 u/yr in local reproduction
    (+6.8%). Used by gem_v2_kmeans3 A/B variant.
    """

    def __init__(
        self,
        tail_frac: float = 0.15,
        min_per_league: int = 30,
        cluster_strategy: str = "none",
        min_per_cluster: int = 50,
    ):
        self.tail_frac = tail_frac
        self.min_per_league = min_per_league
        self.cluster_strategy = cluster_strategy
        self.min_per_cluster = min_per_cluster
        self._global: list[IsotonicRegression] = _new_regs()
        self._by_league: dict[str, list[IsotonicRegression]] = {}
        self._by_cluster: dict[str, list[IsotonicRegression]] = {}
        self._league_to_cluster: dict[str, str] = {}
        self._league_n: dict[str, int] = {}
        self._cluster_n: dict[str, int] = {}
        self._fitted = False

    # ── cluster labelling ──────────────────────────────────────────────

    def _label_clusters(
        self,
        leagues_tail: np.ndarray,
        p_tail: np.ndarray,
    ) -> None:
        """Build self._league_to_cluster on tail data."""
        if self.cluster_strategy == "none":
            return
        if self.cluster_strategy == "tier":
            for lg in pd.unique(leagues_tail):
                if pd.isna(lg) or lg in ("", "nan"):
                    continue
                self._league_to_cluster[lg] = league_cluster(lg)
            return
        if self.cluster_strategy.startswith("kmeans-"):
            k = int(self.cluster_strategy.split("-")[1])
            df = pd.DataFrame({
                "lg": leagues_tail,
                "pH": p_tail[:, 0], "pD": p_tail[:, 1], "pA": p_tail[:, 2],
            })
            df = df[df["lg"].notna() & (df["lg"] != "") & (df["lg"] != "nan")]
            prof = df.groupby("lg")[["pH", "pD", "pA"]].mean()
            if len(prof) < k:
                logger.warning(
                    f"K-means {k} clusters but only {len(prof)} leagues — skipping cluster heads"
                )
                return
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(prof.values)
            for lg, lab in zip(prof.index, km.labels_):
                self._league_to_cluster[lg] = f"k{lab}"
            return
        raise ValueError(f"Unknown cluster_strategy: {self.cluster_strategy}")

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
            self._by_cluster.clear()
            self._league_to_cluster.clear()
            self._league_n.clear()
            self._cluster_n.clear()

            # Cluster labels first (used as fallback for small-sample leagues).
            self._label_clusters(leagues_tail, p_tail)

            for lg in pd.unique(leagues_tail):
                if pd.isna(lg) or lg in ("", "nan"):
                    continue
                lg_mask = leagues_tail == lg
                n_lg = int(lg_mask.sum())
                if n_lg < self.min_per_league:
                    logger.info(
                        f"  per-league SKIP {lg}: {n_lg} tail samples "
                        f"(< {self.min_per_league}) → cluster fallback"
                    )
                    continue
                regs = _new_regs()
                _fit_regs(regs, p_tail[lg_mask], y_tail[lg_mask])
                self._by_league[lg] = regs
                self._league_n[lg] = n_lg
                logger.info(f"  per-league HEAD  {lg}: {n_lg} samples")

            # Cluster heads (using tail samples in each cluster).
            if self.cluster_strategy != "none" and self._league_to_cluster:
                for cl in set(self._league_to_cluster.values()):
                    cl_mask = np.array([
                        self._league_to_cluster.get(l) == cl for l in leagues_tail
                    ])
                    n_cl = int(cl_mask.sum())
                    if n_cl < self.min_per_cluster:
                        logger.info(
                            f"  cluster SKIP {cl}: {n_cl} tail samples (< {self.min_per_cluster})"
                        )
                        continue
                    regs = _new_regs()
                    _fit_regs(regs, p_tail[cl_mask], y_tail[cl_mask])
                    self._by_cluster[cl] = regs
                    self._cluster_n[cl] = n_cl
                    logger.info(f"  cluster HEAD     {cl}: {n_cl} samples")

        self._fitted = True
        return self

    # ── transform ──────────────────────────────────────────────────────

    def transform(
        self,
        proba: np.ndarray,
        leagues: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply most-specific head per row: league > cluster > global.
        Rows are renormalised to sum to 1."""
        assert self._fitted, "call fit() first"

        if leagues is None or (not self._by_league and not self._by_cluster):
            cal = _apply_regs(self._global, proba)
        else:
            leagues = np.asarray(leagues)
            cal = np.zeros_like(proba)
            for lg in pd.unique(leagues):
                lm = leagues == lg
                if lg in self._by_league:
                    regs = self._by_league[lg]
                else:
                    cl = self._league_to_cluster.get(lg)
                    regs = self._by_cluster.get(cl, self._global)
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

    def cluster_summary(self) -> dict[str, dict]:
        """Returns {cluster_id: {n_samples: int, leagues: [list]}} for diagnostics."""
        out: dict[str, dict] = {}
        for cl, n in self._cluster_n.items():
            members = [lg for lg, c in self._league_to_cluster.items() if c == cl]
            out[cl] = {"n_samples": n, "leagues": members}
        return out

    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or ARTIFACTS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "calibrator.pkl")
        logger.info(
            f"Calibrator saved to {d}/calibrator.pkl "
            f"(global + {len(self._by_league)} per-league heads "
            f"+ {len(self._by_cluster)} cluster heads, strategy={self.cluster_strategy})"
        )

    @classmethod
    def load(cls, directory: Path | None = None) -> "GemCalibrator":
        d = Path(directory or ARTIFACTS_DIR)
        return joblib.load(d / "calibrator.pkl")
