"""
Щогодини (:30) — backup-оновлення результатів завершених матчів.
Для матчів, що вже закінчились, розраховує result/pnl для всіх predictions
і оновлює bankroll користувачів.
"""
import time
from datetime import date, datetime

from loguru import logger

from data.api_client import SStatsClient
from data.collectors.apifootball_fallback import fetch_fixture_status, is_finished
from db.models import Match, Prediction, User, BankrollSnapshot
from db.session import SessionLocal
from scheduler.tasks._result_utils import calculate_result, calculate_pnl
from config.settings import CANONICAL_VERSIONS

FINISHED_STATUSES = {"Finished", "FT"}
LIVE_STATUS_EXCLUDE = {0, 1, 8}  # Not started, not started alt, Finished


def run_update_results() -> None:
    """
    1. Знаходить матчі зі ставками без результату.
    2. Запитує SStats — якщо status=8 → оновлює score.
    3. Розраховує result/pnl для predictions цього матчу.
    4. Оновлює bankroll + зберігає snapshot для кожного user.
    """
    logger.info("Starting match results update")
    db = SessionLocal()
    try:
        # Матчі зі ставками без результату (де predictions ще не закриті)
        open_predictions = (
            db.query(Prediction)
            .filter(Prediction.result.is_(None), Prediction.is_active.is_(True))
            .all()
        )
        if not open_predictions:
            logger.info("No open predictions to update")
            return

        match_ids = {p.match_id for p in open_predictions}

        today = date.today()
        all_matches = db.query(Match).filter(Match.id.in_(match_ids)).all()

        # Path A: already Finished in DB with scores → no API call needed
        finished_map: dict[int, dict] = {}
        for m in all_matches:
            if m.status in FINISHED_STATUSES and m.home_score is not None and m.away_score is not None:
                finished_map[m.id] = {
                    "home_score": m.home_score,
                    "away_score": m.away_score,
                }

        # Path B: not yet Finished in DB — poll SStats for past matches
        past_unfinished = [
            m for m in all_matches
            if m.status not in FINISHED_STATUSES
            and (m.date.date() if hasattr(m.date, 'date') else m.date) <= today
        ]

        if not past_unfinished and not finished_map:
            logger.info("No past unfinished matches found")
            return

        if past_unfinished:
            logger.info(f"Checking {len(past_unfinished)} matches via SStats")
            sstats_stale: list[Match] = []  # status != 8 — try API-Football fallback
            with SStatsClient() as client:
                for m in past_unfinished:
                    try:
                        data = client.get(f"/Games/glicko/{m.api_id}")
                        fixture = (data.get("data") or {}).get("fixture") or {}
                        if fixture.get("status") == 8:
                            finished_map[m.id] = {
                                "home_score": fixture.get("homeFTResult"),
                                "away_score": fixture.get("awayFTResult"),
                            }
                        else:
                            # SStats returns 0/1/2/etc — match supposedly hasn't
                            # finished. But m.date is in the past. SStats can be
                            # stale for postponed/rescheduled fixtures → API-Football
                            # fallback.
                            sstats_stale.append(m)
                        time.sleep(0.3)
                    except Exception as e:
                        logger.warning(f"Failed to fetch match {m.api_id}: {e}")
                        sstats_stale.append(m)

            if sstats_stale:
                logger.info(
                    f"SStats reports {len(sstats_stale)} matches as not-yet-finished "
                    "despite past kickoff — checking API-Football fallback"
                )
                for m in sstats_stale:
                    status = fetch_fixture_status(m.api_id)
                    if is_finished(status):
                        finished_map[m.id] = {
                            "home_score": status["home_score"],
                            "away_score": status["away_score"],
                        }
                        logger.info(
                            f"  API-Football: {m.api_id} = "
                            f"{status['home_score']}:{status['away_score']} "
                            f"({status['status_short']})"
                        )
                    time.sleep(7)  # 10 req/min free tier — be polite

        if not finished_map:
            logger.info("No newly finished matches")
            return

        # Оновлюємо Match.status та score (тільки для тих що ще не Finished в БД)
        match_by_id = {m.id: m for m in all_matches}
        for match_id, scores in finished_map.items():
            m = match_by_id[match_id]
            m.status = "Finished"
            m.home_score = scores["home_score"]
            m.away_score = scores["away_score"]
            logger.info(
                f"  {m.home_team.name if m.home_team else '?'} vs "
                f"{m.away_team.name if m.away_team else '?'}: "
                f"{scores['home_score']}:{scores['away_score']}"
            )

        # Розраховуємо result/pnl для predictions
        preds_by_match: dict[int, list[Prediction]] = {}
        for p in open_predictions:
            preds_by_match.setdefault(p.match_id, []).append(p)

        newly_settled = []
        for match_id, scores in finished_map.items():
            preds = preds_by_match.get(match_id, [])
            hs, as_ = scores["home_score"], scores["away_score"]
            if hs is None or as_ is None:
                continue
            for pred in preds:
                if pred.stake is None:
                    continue
                # Refresh from DB — live_tracker may have already settled this prediction
                db.refresh(pred)
                if pred.result is not None:
                    continue  # already settled, skip to avoid double bankroll
                result = calculate_result(pred.market, pred.outcome, hs, as_)
                if result:
                    pred.result = result
                    pred.pnl = calculate_pnl(result, pred.stake, pred.odds_used)
                    newly_settled.append(pred)

        db.commit()

        _update_bankrolls(db, newly_settled)

        logger.info(f"Results updated: {len(finished_map)} matches, {len(newly_settled)} predictions")
    except Exception as e:
        db.rollback()
        logger.error(f"update_results failed [{type(e).__name__}]: {e}")
    finally:
        db.close()


def _update_bankrolls(db, newly_settled: list) -> None:
    """Оновлює bankroll та створює snapshot тільки для щойно settled predictions."""
    if not newly_settled:
        return

    users = db.query(User).filter(User.is_active.is_(True)).all()
    if not users:
        return

    canonical = [p for p in newly_settled if p.model_version in CANONICAL_VERSIONS]
    total_pnl = sum(p.pnl for p in canonical if p.pnl is not None)
    if total_pnl == 0:
        return

    for user in users:
        user.bankroll = round(user.bankroll + total_pnl, 2)
        db.add(BankrollSnapshot(
            user_id=user.id,
            balance=user.bankroll,
            created_at=datetime.utcnow(),
        ))
        logger.info(f"  User {user.telegram_id}: bankroll → {user.bankroll} (pnl={total_pnl:+.2f})")

    db.commit()
