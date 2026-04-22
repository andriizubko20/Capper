"""
Щодня о 10:00 Київ (07:00 UTC) — підтвердження/деактивація pending picks.

Логіка:
- Всі active pending predictions на наступні 7 днів
- Перечитуємо поточні коефіцієнти з БД (останні odds)
- Перераховуємо EV = probability × current_odds − 1
- Якщо EV < 0 → is_active=False (ставка деактивована)
- Якщо матч сьогодні → timing='final', інакше timing='early'

TODO: додати model-specific threshold check після ресьорчу моделей.
"""
from datetime import date, datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import desc

from db.models import Match, Prediction, Odds
from db.session import SessionLocal

DEACTIVATE_EV_THRESHOLD = 0.0   # EV нижче 0 → деактивуємо
LOOKAHEAD_DAYS = 7


def run_confirm_picks() -> None:
    """Перевіряє та оновлює статус ранніх пікс."""
    logger.info("Running daily confirm_picks (10:00 Kyiv)")
    db = SessionLocal()
    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=LOOKAHEAD_DAYS)

    try:
        # Всі active pending predictions на найближчі 7 днів
        pending = (
            db.query(Prediction)
            .join(Match, Prediction.match_id == Match.id)
            .filter(
                Prediction.result.is_(None),
                Prediction.is_active.is_(True),
                Match.date >= datetime.combine(today, datetime.min.time()),
                Match.date <= datetime.combine(cutoff, datetime.max.time()),
            )
            .all()
        )

        if not pending:
            logger.info("No pending picks to confirm")
            return

        logger.info(f"Confirming {len(pending)} pending picks")

        deactivated = 0
        finalized = 0
        marked_early = 0

        for pred in pending:
            match: Match = pred.match
            match_date = match.date.date() if hasattr(match.date, 'date') else match.date

            # Отримуємо найсвіжіші non-closing odds для цього матчу і ринку
            latest_odds = (
                db.query(Odds)
                .filter(
                    Odds.match_id == pred.match_id,
                    Odds.market == pred.market,
                    Odds.outcome == pred.outcome,
                    Odds.is_closing.is_(False),
                )
                .order_by(desc(Odds.recorded_at))
                .first()
            )

            if latest_odds:
                current_odds = latest_odds.value
                ev = pred.probability * current_odds - 1
            else:
                # Якщо нових odds немає — використовуємо збережені
                ev = pred.ev

            # Деактивуємо якщо EV впав нижче порогу
            if ev < DEACTIVATE_EV_THRESHOLD:
                pred.is_active = False
                deactivated += 1
                logger.info(
                    f"  DEACTIVATED pred {pred.id}: "
                    f"{match.home_team.name if match.home_team else '?'} vs "
                    f"{match.away_team.name if match.away_team else '?'} "
                    f"({pred.market}/{pred.outcome}) EV={ev:.3f}"
                )
                continue

            # Визначаємо timing
            if match_date == today:
                pred.timing = 'final'
                finalized += 1
            else:
                pred.timing = 'early'
                marked_early += 1

        db.commit()
        logger.info(
            f"confirm_picks done: {finalized} final, {marked_early} early, "
            f"{deactivated} deactivated"
        )

    finally:
        db.close()
