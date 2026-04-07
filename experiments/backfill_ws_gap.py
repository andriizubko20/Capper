"""
experiments/backfill_ws_gap.py

Одноразовий запуск: генерує WS Gap predictions для матчів 04.04–06.04.2026
(тобто матчів що вже відбулись, щоб зафіксувати що б модель поставила).

Записує в таблицю predictions з model_version='ws_gap_v1'.
НЕ надсилає в Telegram.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from loguru import logger

from scheduler.tasks.generate_picks_ws_gap import run_generate_picks_ws_gap

START = datetime(2026, 4, 4, 0, 0, tzinfo=timezone.utc)
END   = datetime(2026, 4, 7, 0, 0, tzinfo=timezone.utc)  # не включаємо 07.04

if __name__ == "__main__":
    logger.info(f"Backfill WS Gap: {START.date()} → {END.date()}")

    # Патчимо _send_picks_to_telegram щоб не слати в Telegram під час бекфілу
    import scheduler.tasks.generate_picks_ws_gap as module
    module._send_picks_to_telegram = lambda picks: logger.info(
        f"[Backfill] Skipping Telegram send for {len(picks)} picks"
    )

    run_generate_picks_ws_gap(
        match_date_from=START,
        match_date_to=END,
        include_finished=True,
    )
    logger.info("Backfill done")
