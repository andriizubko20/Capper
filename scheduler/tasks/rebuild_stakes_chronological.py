"""
scheduler/tasks/rebuild_stakes_chronological.py

Rebuild stakes + pnls for all models with chronological Kelly compounding.

Bug being fixed: production tasks compute bankroll ONCE per cron run, so all
picks generated in the same run share the same bankroll snapshot — stakes
don't compound with prior wins/losses within that batch.

Correct behavior: bankroll_T = INITIAL + Σ pnl(prior settled picks).
Each pick's stake is sized against the bankroll AT THE TIME of that match.

For each model_version, walks ALL predictions in chronological order
(by match_date, then id), recomputes:
  stake_T = min(bankroll_T × kelly_fraction, bankroll_T × KELLY_CAP)
  pnl_T  = stake_T × (odds_used - 1)  if result == 'win'
        = -stake_T                    if result == 'loss'
        = 0                           if result == 'push' or NULL
  bankroll_{T+1} = bankroll_T + pnl_T  (only if settled)

Usage:
  python -m scheduler.tasks.rebuild_stakes_chronological             # all models
  python -m scheduler.tasks.rebuild_stakes_chronological --model gem_v1
  python -m scheduler.tasks.rebuild_stakes_chronological --dry-run
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from loguru import logger
from sqlalchemy import asc

from config.settings import settings
from db.models import Match, Prediction
from db.session import SessionLocal

KELLY_CAP = 0.10  # all models use the same cap


def rebuild(model_version: str | None = None, dry_run: bool = False) -> None:
    initial = float(settings.bankroll)
    db = SessionLocal()
    try:
        q = db.query(Prediction).join(Match, Prediction.match_id == Match.id)
        if model_version:
            q = q.filter(Prediction.model_version == model_version)
        picks_all = q.order_by(asc(Match.date), asc(Prediction.id)).all()

        if not picks_all:
            logger.warning("No predictions found")
            return

        # Group by model_version, preserving chronological order
        by_model: dict[str, list[Prediction]] = defaultdict(list)
        for p in picks_all:
            by_model[p.model_version].append(p)

        total_updated = 0
        for mv, picks in by_model.items():
            bankroll = initial
            updated = 0
            settled = 0
            wins = 0
            for p in picks:
                kf = float(p.kelly_fraction or 0)
                if kf <= 0 or not p.odds_used:
                    continue
                new_stake = round(min(bankroll * kf, bankroll * KELLY_CAP), 2)

                # Only mutate if the value actually changes (avoid no-op writes)
                changed = False
                if p.stake is None or abs((p.stake or 0) - new_stake) > 0.005:
                    p.stake = new_stake
                    changed = True

                # Recompute pnl if settled
                if p.result is not None:
                    settled += 1
                    if p.result == "win":
                        new_pnl = round(new_stake * (float(p.odds_used) - 1), 2)
                        wins += 1
                    elif p.result == "loss":
                        new_pnl = round(-new_stake, 2)
                    else:  # push, void, etc.
                        new_pnl = 0.0
                    if p.pnl is None or abs((p.pnl or 0) - new_pnl) > 0.005:
                        p.pnl = new_pnl
                        changed = True
                    bankroll = round(bankroll + new_pnl, 2)

                if changed:
                    updated += 1

            wr = (wins / settled * 100) if settled else 0
            roi = ((bankroll - initial) / initial * 100)
            logger.info(
                f"[{mv}] picks={len(picks)} settled={settled} wins={wins} "
                f"WR={wr:.1f}% bankroll=${bankroll:.2f} ROI={roi:+.1f}% "
                f"updated={updated}"
            )
            total_updated += updated

        if dry_run:
            logger.info(f"DRY RUN — would update {total_updated} predictions; rolling back")
            db.rollback()
        else:
            db.commit()
            logger.info(f"Committed {total_updated} prediction updates")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="restrict to one model_version")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rebuild(model_version=args.model, dry_run=args.dry_run)
