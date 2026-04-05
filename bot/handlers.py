from datetime import date, timedelta

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from loguru import logger

from db.models import Match, Prediction, Team, League
from db.session import SessionLocal

router = Router()


def _format_pick(match: Match, pred: Prediction) -> str:
    home = match.home_team.name if match.home_team else "?"
    away = match.away_team.name if match.away_team else "?"
    outcome_label = {"home": f"П1 ({home})", "away": f"П2 ({away})"}.get(pred.outcome, pred.outcome)
    time_str = match.date.strftime("%H:%M")

    return (
        f"⚽ {home} — {away} ({time_str})\n"
        f"  Ставка: {outcome_label}\n"
        f"  Кеф: {pred.odds_used:.2f}  |  EV: {pred.ev * 100:.1f}%\n"
        f"  Наша ймовірність: {pred.probability * 100:.1f}%"
    )


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привіт! Я Capper — ML бот для ставок на футбол.\n\n"
        "Я аналізую матчі за xG, формою, Elo і ринковими коефіцієнтами, "
        "і відправляю тільки ставки з EV ≥ 17% і підтвердженим сигналом.\n\n"
        "Команди:\n"
        "/picks — picks на сьогодні\n"
        "/tomorrow — picks на завтра\n"
        "/stats — статистика моделі"
    )


@router.message(Command("picks"))
async def cmd_picks(message: Message):
    db = SessionLocal()
    try:
        today = date.today()
        matches = db.query(Match).filter(
            Match.date >= str(today),
            Match.date < str(today + timedelta(days=1)),
        ).all()

        picks = []
        for match in matches:
            preds = db.query(Prediction).filter_by(match_id=match.id).all()
            for pred in preds:
                picks.append(_format_pick(match, pred))

        if not picks:
            await message.answer("Сьогодні picks немає.")
        else:
            header = f"📊 Picks на сьогодні ({today.strftime('%d.%m')}) — {len(picks)} ставок:\n\n"
            await message.answer(header + "\n\n".join(picks))
    finally:
        db.close()


@router.message(Command("tomorrow"))
async def cmd_tomorrow(message: Message):
    db = SessionLocal()
    try:
        tomorrow = date.today() + timedelta(days=1)
        matches = db.query(Match).filter(
            Match.date >= str(tomorrow),
            Match.date < str(tomorrow + timedelta(days=2)),
        ).all()

        picks = []
        for match in matches:
            preds = db.query(Prediction).filter_by(match_id=match.id).all()
            for pred in preds:
                picks.append(_format_pick(match, pred))

        if not picks:
            await message.answer("На завтра picks ще немає або не сформовані.")
        else:
            header = f"📊 Picks на завтра ({tomorrow.strftime('%d.%m')}) — {len(picks)} ставок:\n\n"
            await message.answer(header + "\n\n".join(picks))
    finally:
        db.close()


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    db = SessionLocal()
    try:
        from sqlalchemy import text
        r = db.execute(text("""
            SELECT
                COUNT(*) as total,
                AVG(ev) as avg_ev,
                MIN(created_at) as since
            FROM predictions
        """)).fetchone()

        if not r or r[0] == 0:
            await message.answer("Статистика ще недоступна — немає predictions.")
            return

        await message.answer(
            f"📈 Статистика моделі:\n\n"
            f"Всього picks: {r[0]}\n"
            f"Середній EV: {r[1] * 100:.1f}%\n"
            f"Активна з: {r[2].strftime('%d.%m.%Y') if r[2] else '—'}\n\n"
            f"Версія: v1-test\n"
            f"Модель: Logistic Regression\n"
            f"EV поріг: 17% | Cap: 4% | Kelly: 25%"
        )
    finally:
        db.close()
