"""
model/gem/cross_val.py

Walk-forward cross-validation for time-series football match data.

Expanding training window; non-overlapping 2-month validation slices;
10-day gap between train end and val start to prevent same-round leakage
(EPL midweek rounds have 3-4 days between matches, so 10d > 2 rounds).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WalkForwardSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold:2d}: "
            f"train<{self.train_end.date()} | gap | "
            f"val [{self.val_start.date()}, {self.val_end.date()}) "
            f"— {len(self.train_idx):,} train / {len(self.val_idx):,} val"
        )


def walk_forward_splits(
    dates: pd.Series,
    n_folds: int = 12,
    val_months: int = 2,
    gap_days: int = 10,
    min_train_months: int = 12,
) -> list[WalkForwardSplit]:
    """
    Returns walk-forward (train, val) index splits in chronological order.

    Args:
        dates: pd.Series of match dates (datetime64).
        n_folds: max number of folds (fewer if data is short).
        val_months: width of each validation window in months.
        gap_days: days excluded between train end and val start.
        min_train_months: minimum months of history before first fold.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    min_date = dates.min()
    max_date = dates.max()

    splits: list[WalkForwardSplit] = []
    fold_end = max_date + pd.Timedelta(days=1)

    for _ in range(n_folds):
        fold_start = fold_end - pd.DateOffset(months=val_months)
        train_cutoff = fold_start - pd.Timedelta(days=gap_days)
        earliest_allowed = min_date + pd.DateOffset(months=min_train_months)

        if train_cutoff <= earliest_allowed:
            break

        val_mask = (dates >= fold_start) & (dates < fold_end)
        train_mask = dates < train_cutoff

        val_idx = np.where(val_mask)[0]
        train_idx = np.where(train_mask)[0]

        if len(val_idx) >= 30 and len(train_idx) >= 300:
            splits.append(WalkForwardSplit(
                fold=0,
                train_idx=train_idx,
                val_idx=val_idx,
                train_end=train_cutoff,
                val_start=fold_start,
                val_end=fold_end,
            ))

        fold_end = fold_start

    splits = list(reversed(splits))
    for i, s in enumerate(splits):
        s.fold = i + 1

    logger.info(f"Walk-forward CV: {len(splits)} folds over {min_date.date()} → {max_date.date()}")
    for s in splits:
        logger.debug(repr(s))

    return splits
