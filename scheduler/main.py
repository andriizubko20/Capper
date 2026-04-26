from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from scheduler.tasks.collect_data import run_daily_collection
from scheduler.tasks.generate_picks import run_generate_picks
from scheduler.tasks.generate_picks_ws_gap import run_generate_picks_ws_gap
from scheduler.tasks.generate_picks_monster import run_generate_picks_monster
from scheduler.tasks.generate_picks_aquamarine import run_generate_picks_aquamarine
from scheduler.tasks.generate_picks_pure import run_generate_picks_pure
from scheduler.tasks.rebalance_stakes import run_rebalance_stakes
from scheduler.tasks.rebuild_stakes_chronological import rebuild as run_rebuild_stakes_chronological
from scheduler.tasks.update_team_ratings import run_update_team_ratings
from scheduler.tasks.track_odds_movement import run_track_odds_movement
from scheduler.tasks.collect_injuries import run_collect_injuries
from scheduler.tasks.collect_player_stats import run_collect_player_stats
from scheduler.tasks.collect_lineups import run_collect_lineups
from scheduler.tasks.generate_picks_gem import run_generate_picks_gem
from datetime import datetime, timedelta, timezone


def _run_pure_late():
    """Pure late-pass — re-run pick generation 1.5h before kickoff using
    confirmed lineups data."""
    from scheduler.tasks.generate_picks_pure import run_generate_picks_pure
    now = datetime.now(timezone.utc)
    run_generate_picks_pure(
        match_date_from=now + timedelta(hours=1.0),
        match_date_to=now + timedelta(hours=2.0),
    )
from scheduler.tasks.update_monster_p_is import run_update_monster_p_is
from scheduler.tasks.update_clv import run_clv_update
from scheduler.tasks.clv_monitor import run_clv_monitor
from scheduler.tasks.monitor_sstats_proxy import run_monitor_sstats_proxy
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

    # Кожні 30 хв — track 1x2 odds movement for upcoming 36h matches.
    # Each call INSERTS new row (vs collect_data which UPDATES). Builds line-drift
    # timeline for 3-month accumulation → unlocks market features in Gem v3.
    scheduler.add_job(
        run_track_odds_movement,
        CronTrigger(minute="*/30"),
        id="track_odds_movement",
        name="Track odds movement (every 30 min)",
        misfire_grace_time=300,
    )

    # Щогодини в :30 — Gem ML picks (3-model ensemble + calibration + gem filter)
    scheduler.add_job(
        run_generate_picks_gem,
        CronTrigger(minute=33),
        id="generate_picks_gem",
        name="Generate picks Gem (hourly)",
        misfire_grace_time=300,
    )

    # Щогодини в :35 — Pure LATE pass — re-pick using confirmed lineups
    scheduler.add_job(
        _run_pure_late,
        CronTrigger(minute=35),
        id="generate_picks_pure_late",
        name="Generate picks Pure LATE (~1.5h, hourly)",
        misfire_grace_time=300,
    )

    # Кожні 15 хв — confirmed lineups з API-Football (1.5-2h до старту)
    scheduler.add_job(
        run_collect_lineups,
        CronTrigger(minute="*/15"),
        id="collect_lineups",
        name="Collect confirmed lineups (every 15 min)",
        misfire_grace_time=300,
    )

    # 03:30 UTC щодня — оновлення injuries з API-Football
    scheduler.add_job(
        run_collect_injuries,
        CronTrigger(hour=3, minute=30),
        id="collect_injuries",
        name="Daily injuries collection (API-Football)",
        misfire_grace_time=600,
    )

    # Wed 04:00 UTC щотижня — top scorers + xG share per team
    scheduler.add_job(
        run_collect_player_stats,
        CronTrigger(day_of_week="wed", hour=4, minute=0),
        id="collect_player_stats",
        name="Weekly player stats collection (API-Football)",
        misfire_grace_time=3600,
    )

    # Щогодини в :40 — пересчитати власний Glicko-2 з результатів finished matches.
    # Зменшує залежність від SStats /Games/glicko/{fixture_id} endpoint.
    scheduler.add_job(
        run_update_team_ratings,
        CronTrigger(minute=40),
        id="update_team_ratings",
        name="Update self-Glicko team ratings (hourly)",
        misfire_grace_time=300,
    )

    # Кожні 2 години в :50 — restake unsettled picks based on current bankroll.
    # Solves the over-bet problem where multiple picks settle as losses but
    # the remaining pending picks keep stale (over-sized) stakes.
    scheduler.add_job(
        run_rebalance_stakes,
        CronTrigger(hour="*/2", minute=50),
        id="rebalance_stakes",
        name="Rebalance stakes for unsettled picks (every 2h at :50)",
        misfire_grace_time=300,
    )

    # Кожні 30 хв — health-check Mac SSH tunnel + socat proxy chain.
    # SStats геоблочить datacenter IP-адреси → VPS отримує truncated 14762 байт.
    # Tunnel через Mac (residential IP) дає повну відповідь. Якщо tunnel впав
    # — alert у Telegram.
    scheduler.add_job(
        run_monitor_sstats_proxy,
        CronTrigger(minute="*/30"),
        id="monitor_sstats_proxy",
        name="Monitor SStats proxy tunnel (every 30 min)",
        misfire_grace_time=300,
    )

    # Раз на тиждень (неділя 04:00 UTC) — повний рекомпьют історичних stakes/pnls
    # з compound Kelly. Idempotent — нічого не змінить якщо все вже консистентно.
    # Захищає від drift після backfill або ручних правок.
    scheduler.add_job(
        run_rebuild_stakes_chronological,
        CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="rebuild_stakes_chronological",
        name="Rebuild all stakes chronologically (weekly Sun 04:00)",
        misfire_grace_time=3600,
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

    # 08:00 UTC щодня — моніторинг CLV (rolling 30d avg per model).
    # Шле Telegram alert якщо модель пробила -2% поріг (або відновилася).
    # Запускається після нічного update_clv (23:00) — свіжі closing odds вже є.
    scheduler.add_job(
        run_clv_monitor,
        CronTrigger(hour=8, minute=0),
        id="clv_monitor",
        name="CLV monitoring (daily 08:00 UTC)",
        misfire_grace_time=3600,
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
