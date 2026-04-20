"""
Кожні 3 хвилини — live tracking матчів, що зараз грають.
Поллить /Games/glicko/{id} тільки для матчів на сьогодні зі ставками.
Оновлює поточний рахунок; якщо матч завершився → result/pnl + bankroll snapshot.
"""
import time
from datetime import date, datetime

from loguru import logger

from data.api_client import SStatsClient
from db.models import Match, Prediction, User, BankrollSnapshot
from db.session import SessionLocal
from scheduler.tasks._result_utils import calculate_result, calculate_pnl

# status codes від SStats
STATUS_NOT_STARTED = {0, 1}
STATUS_FINISHED = 8


def run_live_tracker() -> None:
    """
    1. Знаходить матчі на сьогодні зі відкритими predictions.
    2. Запитує /Games/glicko/{id} для кожного.
    3. Оновлює поточний рахунок (home_score / away_score = homeResult / awayResult).
    4. Якщо status=8 → фіксує фінальний рахунок, result/pnl, bankroll snapshot.
    """
    logger.debug("Live tracker tick")
    db = SessionLocal()
    try:
        today = date.today()

        # Матчі на сьогодні зі відкритими predictions
        open_preds = (
            db.query(Prediction)
            .filter(Prediction.result.is_(None), Prediction.is_active.is_(True))
            .all()
        )
        if not open_preds:
            return

        match_ids = {p.match_id for p in open_preds}
        matches = (
            db.query(Match)
            .filter(Match.id.in_(match_ids))
            .all()
        )

        # Тільки матчі на сьогодні (не майбутні, не вже Finished у БД)
        target_matches = [
            m for m in matches
            if (m.date.date() if hasattr(m.date, 'date') else m.date) == today
            and m.status not in {"Finished", "FT"}
        ]

        if not target_matches:
            return

        logger.info(f"Live tracker: polling {len(target_matches)} matches")

        newly_finished: dict[int, dict] = {}  # match.id → {home_ft, away_ft}

        with SStatsClient() as client:
            for m in target_matches:
                try:
                    data = client.get(f"/Games/glicko/{m.api_id}")
                    fixture = (data.get("data") or {}).get("fixture") or {}
                    status_code = fixture.get("status")

                    if status_code in STATUS_NOT_STARTED:
                        pass  # матч ще не почався

                    elif status_code == STATUS_FINISHED:
                        # Фінальний рахунок
                        home_ft = fixture.get("homeFTResult")
                        away_ft = fixture.get("awayFTResult")
                        if home_ft is not None and away_ft is not None:
                            m.status = "Finished"
                            m.home_score = home_ft
                            m.away_score = away_ft
                            newly_finished[m.id] = {
                                "home_score": home_ft,
                                "away_score": away_ft,
                            }
                            logger.info(
                                f"  FINISHED: {m.home_team.name if m.home_team else '?'} "
                                f"{home_ft}:{away_ft} "
                                f"{m.away_team.name if m.away_team else '?'}"
                            )

                    else:
                        # Live — оновлюємо поточний рахунок (homeResult / awayResult)
                        home_cur = fixture.get("homeResult")
                        away_cur = fixture.get("awayResult")
                        elapsed = fixture.get("elapsed")
                        status_name = fixture.get("statusName", "")
                        if home_cur is not None:
                            m.home_score = home_cur
                            m.away_score = away_cur
                            m.status = status_name or str(status_code)
                            logger.debug(
                                f"  LIVE [{elapsed}']: "
                                f"{m.home_team.name if m.home_team else '?'} "
                                f"{home_cur}:{away_cur} "
                                f"{m.away_team.name if m.away_team else '?'}"
                            )

                    time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Live tracker error for match {m.api_id}: {e}")

        db.commit()

        if not newly_finished:
            return

        # Розраховуємо result/pnl для щойно завершених матчів
        preds_by_match: dict[int, list[Prediction]] = {}
        for p in open_preds:
            preds_by_match.setdefault(p.match_id, []).append(p)

        updated = 0
        for match_id, scores in newly_finished.items():
            hs, as_ = scores["home_score"], scores["away_score"]
            for pred in preds_by_match.get(match_id, []):
                if pred.stake is None:
                    continue
                result = calculate_result(pred.market, pred.outcome, hs, as_)
                if result:
                    pred.result = result
                    pred.pnl = calculate_pnl(result, pred.stake, pred.odds_used)
                    updated += 1

        db.commit()

        if updated:
            _update_bankrolls(db, newly_finished, preds_by_match)
            logger.info(f"Live tracker: settled {updated} predictions across {len(newly_finished)} matches")

    finally:
        db.close()


def _update_bankrolls(db, finished_map: dict, preds_by_match: dict) -> None:
    """Підсумовує PnL нових ставок і створює bankroll snapshot."""
    users = db.query(User).filter(User.is_active.is_(True)).all()
    if not users:
        return

    total_pnl = sum(
        pred.pnl
        for match_id in finished_map
        for pred in preds_by_match.get(match_id, [])
        if pred.pnl is not None
    )

    if total_pnl == 0:
        return

    for user in users:
        user.bankroll = round(user.bankroll + total_pnl, 2)
        db.add(BankrollSnapshot(
            user_id=user.id,
            balance=user.bankroll,
            created_at=datetime.utcnow(),
        ))

    db.commit()
    logger.info(f"Bankroll updated: pnl={total_pnl:+.2f}")
