"""
scheduler/tasks/track_odds_movement.py

High-frequency odds tracking: snapshots upcoming matches' 1x2 + double-chance
odds every 30 min. Each snapshot is a NEW row (not update), so we get a full
line-drift timeline per match.

After ~3 months of accumulation:
  - opening_value = first snapshot per (match, outcome)
  - closing_value = last snapshot before kickoff
  - drift = closing - opening
  - drift_velocity = derivative near kickoff (sharp money)

These features unlock the quantum leap for Gem v3.

Difference from existing collect_data.py:
  collect_data: every 3h, UPDATES `value`, runs across 7-day window
  track_odds_movement: every 30min, INSERTS new rows, 36-hour window only
"""
import time
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import text

from data.api_client import SStatsClient
from data.collectors.odds import fetch_odds
from db.models import Match, Odds
from db.session import SessionLocal

# Lookahead window: cover today + tomorrow morning kickoffs
LOOKAHEAD_HOURS = 36
# Skip team if a fresh snapshot (< this minutes old) exists — avoids near-dup
MIN_SNAPSHOT_GAP_MINUTES = 25
# Sleep between API calls (rate limit safety)
DELAY_SEC = 1.0
# Markets to track
TRACK_MARKETS = ("1x2",)


def _has_recent_snapshot(db, match_id: int, market: str, threshold_minutes: int) -> bool:
    """True if any odds row for this match+market within `threshold_minutes`."""
    cutoff = datetime.utcnow() - timedelta(minutes=threshold_minutes)
    return db.execute(text(
        """
        SELECT 1 FROM odds
        WHERE match_id = :mid AND market = :mkt AND recorded_at > :cutoff
        LIMIT 1
        """
    ), {"mid": match_id, "mkt": market, "cutoff": cutoff}).fetchone() is not None


def _save_snapshot(db, match_id: int, odds_list: list[dict]) -> int:
    """Insert new odds rows (one per outcome). Returns count saved."""
    saved = 0
    now = datetime.utcnow()
    for o in odds_list:
        if o["market"] not in TRACK_MARKETS:
            continue
        # Find the existing first opening_value for this (match, outcome) — preserve it
        existing_first = db.execute(text(
            """
            SELECT MIN(opening_value) FROM odds
            WHERE match_id = :mid AND market = :mkt AND outcome = :out
              AND opening_value IS NOT NULL
            """
        ), {"mid": match_id, "mkt": o["market"], "out": o["outcome"]}).scalar()
        opening = existing_first if existing_first is not None else o.get("opening_odds") or o["odds"]

        try:
            db.add(Odds(
                match_id=match_id,
                market=o["market"],
                bookmaker=o["bookmaker"],
                outcome=o["outcome"],
                value=o["odds"],
                opening_value=opening,
                is_closing=False,
                recorded_at=now,
            ))
            saved += 1
        except Exception:
            db.rollback()
            continue
    db.commit()
    return saved


def run_track_odds_movement() -> None:
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=LOOKAHEAD_HOURS)

    db = SessionLocal()
    try:
        upcoming = db.query(Match).filter(
            Match.date >= now.replace(tzinfo=None),
            Match.date < window_end.replace(tzinfo=None),
            Match.status == "Not Started",
        ).all()
        logger.info(f"[OddsTracker] {len(upcoming)} matches in next {LOOKAHEAD_HOURS}h")

        if not upcoming:
            return

        n_skipped = n_fetched = n_saved = 0
        with SStatsClient() as client:
            for match in upcoming:
                # Skip if recent snapshot already exists for this match
                if _has_recent_snapshot(db, match.id, "1x2", MIN_SNAPSHOT_GAP_MINUTES):
                    n_skipped += 1
                    continue
                try:
                    odds_list = fetch_odds(match.api_id, client)
                    n_fetched += 1
                    if odds_list:
                        n_saved += _save_snapshot(db, match.id, odds_list)
                except Exception as e:
                    logger.warning(f"[OddsTracker] {match.api_id} fail: {e}")
                time.sleep(DELAY_SEC)

        logger.info(
            f"[OddsTracker] Done: fetched {n_fetched}, saved {n_saved} rows, "
            f"skipped {n_skipped} (fresh)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    run_track_odds_movement()
