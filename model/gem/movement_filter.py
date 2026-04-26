"""
model/gem/movement_filter.py

Sharp-money pick filter for the Gem model.

Reads odds-snapshot history written by `scheduler/tasks/track_odds_movement.py`
and decides whether the market has moved against our pick strongly enough to
skip it.

Public API:
    movement_signals(db, match_id) -> dict[str, float | int | None]
        Aggregated movement features for a match. None values mean "not enough
        data to compute" — caller must treat them as graceful no-ops.

    should_skip_pick(signals, pick_side) -> tuple[bool, str | None]
        Decision helper. Returns (skip, reason). reason is None when not skipped.

Design doc: model/gem/movement_filter.md
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean, median, pstdev

from sqlalchemy import text

from model.gem.niches import (
    ENABLE_MOVEMENT_FILTER,
    MOVEMENT_DISPERSION_THRESHOLD,
    MOVEMENT_DRIFT_THRESHOLD,
    MOVEMENT_MIN_BOOKMAKERS,
    MOVEMENT_MIN_SNAPSHOTS,
    MOVEMENT_VELOCITY_THRESHOLD,
)

SIDES = ("home", "draw", "away")


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _per_book_series(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (bookmaker, outcome), sorted by recorded_at ascending."""
    out: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["bookmaker"], r["outcome"])
        out.setdefault(key, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["recorded_at"])
    return out


def _compute_per_side(
    series: dict[tuple[str, str], list[dict]],
    side: str,
    now: datetime,
) -> dict[str, float | int | None]:
    """Compute drift / velocity / dispersion for one outcome side."""
    drifts: list[float] = []
    velocities: list[float] = []
    current_values: list[float] = []
    snap_counts: list[int] = []

    for (book, outcome), bucket in series.items():
        if outcome != side or not bucket:
            continue
        opening = bucket[0].get("opening_value") or bucket[0]["value"]
        if opening is None or opening <= 0:
            continue
        latest = bucket[-1]["value"]
        if latest is None:
            continue
        current_values.append(latest)
        snap_counts.append(len(bucket))
        drifts.append((latest - opening) / opening)

        # Velocity: last value vs. value ≥ 30 min ago. Pick the latest row
        # whose recorded_at is at least 30 min before the latest row.
        cutoff = bucket[-1]["recorded_at"] - timedelta(minutes=30)
        prior = [r for r in bucket[:-1] if r["recorded_at"] <= cutoff]
        if prior:
            v_prior = prior[-1]["value"]
            if v_prior and v_prior > 0:
                velocities.append((latest - v_prior) / opening)

    drift = median(drifts) if drifts else None
    velocity = median(velocities) if velocities else None
    if len(current_values) >= 2:
        m = mean(current_values)
        dispersion = (pstdev(current_values) / m) if m > 0 else None
    else:
        dispersion = None
    n_books = len(current_values)
    n_snaps = max(snap_counts) if snap_counts else 0

    return {
        f"drift_{side}":      drift,
        f"velocity_{side}":   velocity,
        f"dispersion_{side}": dispersion,
        f"n_books_{side}":    n_books,
        f"n_snaps_{side}":    n_snaps,
    }


def movement_signals(db, match_id: int) -> dict[str, float | int | None]:
    """Read odds snapshots for a match and return aggregated movement features.

    Always returns a dict with the same keys; missing data → None.

    Keys:
        drift_<side>        median_b ((current − opening) / opening)
        velocity_<side>     median_b ((current − value_30m_ago) / opening)
        dispersion_<side>   std_b(current) / mean_b(current)
        n_books_<side>      number of bookmakers contributing
        n_snaps_<side>      max snapshot count across bookmakers
        n_snapshots         max(n_snaps_<side>) — overall data quality
        n_bookmakers        max(n_books_<side>) — market breadth
    """
    rows = db.execute(text(
        """
        SELECT bookmaker, outcome, value, opening_value, recorded_at
        FROM odds
        WHERE match_id = :mid AND market = '1x2'
        ORDER BY recorded_at ASC
        """
    ), {"mid": match_id}).fetchall()

    base: dict[str, float | int | None] = {
        f"drift_{s}":      None for s in SIDES
    }
    for s in SIDES:
        base[f"velocity_{s}"]   = None
        base[f"dispersion_{s}"] = None
        base[f"n_books_{s}"]    = 0
        base[f"n_snaps_{s}"]    = 0
    base["n_snapshots"]  = 0
    base["n_bookmakers"] = 0

    if not rows:
        return base

    parsed = [
        {
            "bookmaker":     r.bookmaker,
            "outcome":       r.outcome,
            "value":         r.value,
            "opening_value": r.opening_value,
            "recorded_at":   r.recorded_at,
        }
        for r in rows
    ]
    series = _per_book_series(parsed)
    now = max(r["recorded_at"] for r in parsed)

    out = dict(base)
    for s in SIDES:
        out.update(_compute_per_side(series, s, now))

    out["n_snapshots"]  = max(out[f"n_snaps_{s}"]  for s in SIDES)
    out["n_bookmakers"] = max(out[f"n_books_{s}"]  for s in SIDES)
    return out


def should_skip_pick(
    signals: dict,
    pick_side: str,
) -> tuple[bool, str | None]:
    """Decide whether to skip a Gem pick given movement signals.

    Returns (skip: bool, reason: str | None). Reason describes the offending
    metric for logging. Conservative: insufficient data → never skip.
    """
    if not ENABLE_MOVEMENT_FILTER:
        return False, None
    if pick_side not in ("home", "away"):
        return False, None

    n_snaps = signals.get("n_snapshots") or 0
    if n_snaps < MOVEMENT_MIN_SNAPSHOTS:
        return False, None

    drift = signals.get(f"drift_{pick_side}")
    if drift is not None and drift > MOVEMENT_DRIFT_THRESHOLD:
        return True, f"drift {drift:+.1%} > {MOVEMENT_DRIFT_THRESHOLD:.0%}"

    velocity = signals.get(f"velocity_{pick_side}")
    if velocity is not None and velocity > MOVEMENT_VELOCITY_THRESHOLD:
        return True, f"velocity {velocity:+.1%} > {MOVEMENT_VELOCITY_THRESHOLD:.0%}"

    dispersion = signals.get(f"dispersion_{pick_side}")
    n_books = signals.get(f"n_books_{pick_side}") or 0
    if (
        dispersion is not None
        and n_books >= MOVEMENT_MIN_BOOKMAKERS
        and dispersion > MOVEMENT_DISPERSION_THRESHOLD
    ):
        return True, f"dispersion {dispersion:.1%} > {MOVEMENT_DISPERSION_THRESHOLD:.0%}"

    return False, None
