from datetime import date, timedelta

from aiogram import Bot
from loguru import logger

from db.models import Match, Prediction, User
from db.session import SessionLocal


async def broadcast_picks(bot: Bot) -> None:
    """Щоденна розсилка picks всім активним користувачам."""
    db = SessionLocal()
    try:
        tomorrow = date.today() + timedelta(days=1)
        matches = db.query(Match).filter(
            Match.date >= str(tomorrow),
            Match.date < str(tomorrow + timedelta(days=2)),
        ).all()

        picks_text = []
        for match in matches:
            preds = db.query(Prediction).filter_by(match_id=match.id).all()
            for pred in preds:
                home = match.home_team.name if match.home_team else "?"
                away = match.away_team.name if match.away_team else "?"
                outcome_label = {"home": f"П1 ({home})", "away": f"П2 ({away})"}.get(pred.outcome, pred.outcome)
                time_str = match.date.strftime("%H:%M")
                picks_text.append(
                    f"⚽ {home} — {away} ({time_str})\n"
                    f"  {outcome_label} | {pred.odds_used:.2f} | EV {pred.ev * 100:.1f}%"
                )

        if not picks_text:
            logger.info("No picks to broadcast")
            return

        message = (
            f"📊 Picks на {tomorrow.strftime('%d.%m')} — {len(picks_text)} ставок:\n\n"
            + "\n\n".join(picks_text)
        )

        users = db.query(User).filter_by(is_active=True).all()
        sent = 0
        for user in users:
            try:
                await bot.send_message(user.telegram_id, message)
                sent += 1
            except Exception as e:
                logger.warning(f"Failed to send to {user.telegram_id}: {e}")

        logger.info(f"Broadcast sent to {sent}/{len(users)} users")
    finally:
        db.close()
