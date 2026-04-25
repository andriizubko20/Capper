from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from scheduler.tasks.collect_data import run_daily_collection
from scheduler.tasks.generate_picks import run_generate_picks
from scheduler.tasks.generate_picks_ws_gap import run_generate_picks_ws_gap
from scheduler.tasks.generate_picks_monster import run_generate_picks_monster
from scheduler.tasks.generate_picks_aquamarine import run_generate_picks_aquamarine
from scheduler.tasks.generate_picks_pure import run_generate_picks_pure
from scheduler.tasks.update_monster_p_is import run_update_monster_p_is
from scheduler.tasks.update_clv import run_clv_update
from scheduler.tasks.update_results import run_update_results
from scheduler.tasks.live_tracker import run_live_tracker
from scheduler.tasks.confirm_picks import run_confirm_picks
from scheduler.tasks.load_historical import run_load_historical
from scheduler.tasks.retrain import run_retrain


def start() -> None:
    scheduler = BlockingScheduler(timezone="UTC")

    # Кожні 3 години (06, 09, 12, 15, 18, 21 UTC) — збір і оновлення коефіцієнтів
    scheduler.add_job(
        run_daily_collection,
        CronTrigger(hour="6,9,12,15,18,21", minute=0),
        id="collect_data",
        name="Odds collection (every 3h)",
        misfire_grace_time=300,
    )

    # ML v1 — вимкнено, активна тільки WS Gap модель
    # scheduler.add_job(run_generate_picks, ...)

    # Щогодини — WS Gap picks (~5h до старту)
    scheduler.add_job(
        run_generate_picks_ws_gap,
        CronTrigger(minute=5),
        id="generate_picks_ws_gap",
        name="Generate picks WS Gap final (hourly)",
        misfire_grace_time=300,
    )

    # Щогодини в :15 — Monster picks (~5h до старту)
    scheduler.add_job(
        run_generate_picks_monster,
        CronTrigger(minute=15),
        id="generate_picks_monster",
        name="Generate picks Monster (hourly)",
        misfire_grace_time=300,
    )

    # Щогодини в :20 — Aquamarine picks (~5h до старту)
    scheduler.add_job(
        run_generate_picks_aquamarine,
        CronTrigger(minute=20),
        id="generate_picks_aquamarine",
        name="Generate picks Aquamarine (hourly)",
        misfire_grace_time=300,
    )

    # Щогодини в :25 — Pure picks (~5h до старту)
    scheduler.add_job(
        run_generate_picks_pure,
        CronTrigger(minute=25),
        id="generate_picks_pure",
        name="Generate picks Pure (hourly)",
        misfire_grace_time=300,
    )

    # Кожні 3 хвилини — live tracking (score + result для завершених матчів)
    scheduler.add_job(
        run_live_tracker,
        CronTrigger(minute="*/3"),
        id="live_tracker",
        name="Live match tracker (every 3 min)",
        misfire_grace_time=60,
    )

    # Щогодини в :30 — backup оновлення результатів
    scheduler.add_job(
        run_update_results,
        CronTrigger(minute=30),
        id="update_results",
        name="Update match results (hourly backup)",
        misfire_grace_time=300,
    )

    # Кожні 2 години в :45 — підтвердження / деактивація пікс + timing update
    scheduler.add_job(
        run_confirm_picks,
        CronTrigger(hour="*/2", minute=45),
        id="confirm_picks",
        name="Confirm picks (every 2h at :45)",
        misfire_grace_time=600,
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

    # Вівторок 03:00 UTC — оновлення p_is для Monster ніш
    scheduler.add_job(
        run_update_monster_p_is,
        CronTrigger(day_of_week="tue", hour=3, minute=0),
        id="update_monster_p_is",
        name="Weekly Monster p_is update (Tue 03:00 UTC)",
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
