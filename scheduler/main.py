from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from scheduler.tasks.collect_data import run_daily_collection
from scheduler.tasks.generate_picks import run_generate_picks
from scheduler.tasks.update_clv import run_clv_update
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

    # 09:00 UTC = 12:00 Київ — генерація пиків
    scheduler.add_job(
        run_generate_picks,
        CronTrigger(hour=9, minute=0),
        id="generate_picks",
        name="Generate daily picks",
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
