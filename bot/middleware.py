"""
bot/middleware.py — Whitelist access control middleware.

Якщо ALLOWED_TELEGRAM_IDS в .env порожній — бот відкритий для всіх.
Якщо заповнений (comma-separated IDs) — тільки ці юзери мають доступ.
"""
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, CallbackQuery, TelegramObject

from config.settings import settings


class WhitelistMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        allowed = settings.allowed_ids_set
        if not allowed:
            return await handler(event, data)

        user_id = None
        if isinstance(event, Message):
            user_id = event.from_user.id if event.from_user else None
        elif isinstance(event, CallbackQuery):
            user_id = event.from_user.id if event.from_user else None

        if user_id is not None and user_id not in allowed:
            if isinstance(event, Message):
                await event.answer("⛔️ Нет доступа.")
            elif isinstance(event, CallbackQuery):
                await event.answer("⛔️ Нет доступа.", show_alert=True)
            return

        return await handler(event, data)
