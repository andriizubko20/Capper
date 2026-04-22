"""
bot/handlers.py — Capper bot entry point.

Single /start command: welcome message + Mini App button.
"""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from loguru import logger

from config.settings import settings

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="Open Capper",
            web_app=WebAppInfo(url=settings.miniapp_url),
        )
    ]])

    try:
        await message.answer(
            "👋 <b>Welcome to Capper</b>\n\n"
            "AI-powered football betting picks — backed by ML models with real edge.\n\n"
            "📊 <b>3 models</b> · WS Gap · Monster · Aquamarine\n"
            "✅ Value bets only (EV+ filtered)\n"
            "💰 Kelly Criterion stake sizing\n\n"
            "Tap the button below to open the app.",
            reply_markup=keyboard,
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning(f"cmd_start: failed to reply to {message.from_user.id}: {e}")
