from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from scheduler.tasks.collect_data import run_daily_collection
from scheduler.tasks.generate_picks import run_generate_picks
from scheduler.tasks.generate_picks_ws_gap import run_generate_picks_ws_gap
from scheduler.tasks.update_clv import run_clv_update
from scheduler.tasks.update_results import run_update_results
from scheduler.tasks.load_historical import run_load_historical
from scheduler.tasks.retrain import run_retrain


def start() -> None:
    scheduler = BlockingScheduler(timezone="UTC")

    # 06:00 UTC = 09:00 Київ — збір матчів і коефіцієнтів
    scheduler.add_job(
        run_daily_collection,
        CronTrigger(hour=6, minute=0),
        id="collect_data",
        name="Daily data collection",
        misfire_grace_time=300,
    )

    # ML v1 — вимкнено, активна тільки WS Gap модель
    # scheduler.add_job(run_generate_picks, ...)

    # Щогодини — WS Gap модель
    scheduler.add_job(
        run_generate_picks_ws_gap,
        CronTrigger(minute=5),  # зміщено на :05 щоб не конфліктувати з основним
        id="generate_picks_ws_gap",
        name="Generate picks WS Gap (hourly)",
        misfire_grace_time=300,
    )

    # Щогодини в :30 — оновлення результатів завершених матчів
    scheduler.add_job(
        run_update_results,
        CronTrigger(minute=30),
        id="update_results",
        name="Update match results (hourly)",
        misfire_grace_time=300,
    )

    # 23:00 UTC = 02:00 Київ — оновлення CLV
    scheduler.add_job(
        run_clv_update,
        CronTrigger(hour=23, minute=0),
        id="update_clv",
        name="Update CLV for finished matches",
        misfire_grace_time=300,
    )

    # 02:00 UTC = 05:00 Київ — нічне завантаження нових ліг
    scheduler.add_job(
        run_load_historical,
        CronTrigger(hour=2, minute=0),
        id="load_historical",
        name="Nightly historical data load",
        misfire_grace_time=3600,
    )

    # Понеділок 04:00 UTC = 07:00 Київ — ретрейнінг
    scheduler.add_job(
        run_retrain,
        CronTrigger(day_of_week="mon", hour=4, minute=0),
        id="retrain",
        name="Weekly model retrain",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name} → {job.trigger}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    start()
