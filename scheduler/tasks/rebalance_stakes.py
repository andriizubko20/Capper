"""
scheduler/tasks/rebalance_stakes.py

Re-stake unsettled future picks based on current bankroll.

Problem this solves:
  When the daily picks task fires, it generates 5-15 picks for tomorrow's matches
  with stake = Kelly × current_bankroll. If 3 of those bets settle as losses
  during the day, the remaining picks (still pending) keep stale over-priced
  stakes — they should be sized against the SHRUNK bankroll.

Logic (per model version):
  1. Pull all predictions for that model.
  2. Replay settled (chronologically) → compute current bankroll.
  3. For every UNSETTLED prediction:
        new_stake = round(min(kelly_fraction × bankroll, KELLY_CAP × bankroll), 2)
        Update DB if changed.

Runs every 2 hours via scheduler.
"""
from loguru import logger

from db.models import Match, Prediction
from db.session import SessionLocal

KELLY_CAP   = 0.10
STARTING_BK = 1000.0
FINISHED = {"Finished", "FT", "finished", "ft", "Match Finished"}

# Models that use Kelly+cap sizing — Pure and Gem share the convention
KELLY_VERSIONS = [
    "ws_gap_kelly_v1",
    "monster_v1_kelly",
    "aquamarine_v1_kelly",
    "pure_v1",
    "gem_v1",
]


def _rebalance_one(model_ver: str, db) -> tuple[int, float]:
    """Returns (n_updated, current_bankroll)."""
    preds = (
        db.query(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Prediction.model_version == model_ver)
        .filter(Prediction.match_id.isnot(None))
        .order_by(Match.date.asc(), Prediction.id.asc())
        .all()
    )
    if not preds:
        return 0, STARTING_BK

    # Replay settled to get current bankroll
    bankroll = STARTING_BK
    for p in preds:
        if p.result == "win":
            bankroll += round((p.stake or 0) * (p.odds_used - 1), 2)
        elif p.result == "loss":
            bankroll -= (p.stake or 0)

    # Re-stake unsettled
    updated = 0
    for p in preds:
        if p.result is not None:
            continue
        # Only restake if match is still ahead (not played yet)
        if p.match and p.match.status in FINISHED:
            continue
        kf = p.kelly_fraction or 0.0
        new_stake = round(min(kf * bankroll, KELLY_CAP * bankroll), 2)
        old_stake = p.stake or 0.0
        if abs(new_stake - old_stake) >= 0.01:
            logger.debug(
                f"[{model_ver}] match {p.match_id}: stake ${old_stake:.2f} → ${new_stake:.2f} "
                f"(bankroll=${bankroll:.0f}, kf={kf:.4f})"
            )
            p.stake = new_stake
            updated += 1
    db.commit()
    return updated, bankroll


def run_rebalance_stakes() -> None:
    db = SessionLocal()
    try:
        for mv in KELLY_VERSIONS:
            try:
                n, bk = _rebalance_one(mv, db)
                logger.info(f"[Rebalance] {mv}: bankroll=${bk:.0f}, restaked {n} pending picks")
            except Exception as e:
                logger.error(f"[Rebalance] {mv} failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    run_rebalance_stakes()
