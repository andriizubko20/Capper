"""
data/recalculate_kelly_cap.py

Перераховує stake і pnl для всіх _kelly predictions з новим cap=10%.

Логіка:
- Сортуємо predictions кожної моделі за датою матчу (asc)
- Стартуємо з bankroll = 1000
- Для кожного settled (result IS NOT NULL): new_stake = min(kf * bankroll, 0.10 * bankroll)
- Оновлюємо stake і pnl у БД
- Bankroll коригується тільки для settled bets

Запуск:
  docker compose exec scheduler python -m data.recalculate_kelly_cap
"""
from __future__ import annotations

import sys
from loguru import logger

from db.session import SessionLocal
from db.models import Match, Prediction

KELLY_CAP    = 0.10
STARTING_BK  = 1000.0

KELLY_VERSIONS = [
    "ws_gap_kelly_v1",
    "monster_v1_kelly",
    "aquamarine_v1_kelly",
]

FINISHED = {'Finished', 'FT', 'finished', 'ft', 'Match Finished'}


def recalculate(model_ver: str, db) -> None:
    preds = (
        db.query(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Prediction.model_version == model_ver)
        .filter(Prediction.match_id.isnot(None))   # тільки production
        .order_by(Match.date.asc(), Prediction.id.asc())
        .all()
    )

    if not preds:
        logger.info(f"[{model_ver}] No predictions found, skipping")
        return

    bankroll = STARTING_BK
    updated = 0

    for pred in preds:
        kf = pred.kelly_fraction or 0.0
        new_stake = round(min(kf * bankroll, KELLY_CAP * bankroll), 2)

        # Оновлюємо stake
        old_stake = pred.stake
        pred.stake = new_stake

        # Оновлюємо pnl якщо є result
        if pred.result == 'win':
            new_pnl = round(new_stake * (pred.odds_used - 1), 2)
            bankroll += new_pnl
        elif pred.result == 'loss':
            new_pnl = round(-new_stake, 2)
            bankroll -= new_stake
        else:
            # pending / null — stake оновлюємо, pnl не міняємо
            new_pnl = pred.pnl

        if pred.result in ('win', 'loss'):
            pred.pnl = new_pnl
            updated += 1

        logger.debug(
            f"  [{pred.match.date.date() if pred.match else '?'}] "
            f"result={pred.result} kf={kf:.4f} "
            f"stake ${old_stake or 0:.2f}→${new_stake:.2f} "
            f"pnl {new_pnl or 0:.2f} bankroll=${bankroll:.0f}"
        )

    db.commit()
    logger.info(
        f"[{model_ver}] {len(preds)} predictions processed, "
        f"{updated} settled updated. Final bankroll: ${bankroll:.2f}"
    )


def main() -> None:
    db = SessionLocal()
    try:
        for mv in KELLY_VERSIONS:
            logger.info(f"--- Recalculating {mv} ---")
            recalculate(mv, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
