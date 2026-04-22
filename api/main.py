"""
FastAPI app — 5 endpoints для Telegram miniapp.
Запускається окремим процесом через api/run.py (uvicorn, port 8000).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func

from db.models import BankrollSnapshot, Match, Odds, Prediction, User
from db.session import SessionLocal
from api.utils import (
    LEAGUE_FLAGS,
    MODEL_VERSIONS,
    format_side,
    match_status_label,
)

app = FastAPI(title="Capper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────────

def get_user_or_404(db, telegram_id: int) -> User:
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def period_to_days(period: str) -> int:
    mapping = {"7d": 7, "30d": 30, "90d": 90, "all": 3650}
    return mapping.get(period, 30)


def prediction_to_pick(pred: Prediction, match: Match) -> dict[str, Any]:
    """Серіалізує Prediction + Match у dict для фронту."""
    league_name = match.league.name if match.league else ""
    league_country = match.league.country if match.league else ""
    flag = LEAGUE_FLAGS.get(league_name, "🏟")
    # Resolve "Premier League" collision: Ukraine vs England
    if league_name == "Premier League" and league_country == "Ukraine":
        flag = "🇺🇦"
    home = match.home_team.name if match.home_team else ""
    away = match.away_team.name if match.away_team else ""
    home_id = match.home_team.api_id if match.home_team else 0
    away_id = match.away_team.api_id if match.away_team else 0

    status_raw = match_status_label(match.status, match.date)
    # якщо prediction закрита — використовуємо result
    if pred.result:
        status = pred.result  # 'win' | 'loss'
    else:
        status = status_raw   # 'live' | 'pending'

    score = None
    if match.home_score is not None and match.away_score is not None:
        score = f"{match.home_score}-{match.away_score}"

    # match.date зберігається як UTC — віддаємо ISO рядок, фронт конвертує в локальний TZ
    match_dt_utc = match.date.replace(tzinfo=timezone.utc)
    datetime_utc = match_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    # fallback time_str (UTC) — для pending/finished; live замінюємо на хвилину матчу
    time_str = match_dt_utc.strftime("%H:%M")

    # Для live матчів — показуємо хвилину матчу замість часу початку
    if status_raw == "live":
        s = (match.status or "").strip().lower()

        if "half time" in s or "halftime" in s or "break" in s:
            time_str = "HT"
        elif "extra time" in s or "extra" in s:
            time_str = "ET"
        elif "penalty" in s or "penalties" in s:
            time_str = "PKS"
        elif match.elapsed is not None:
            # Точна хвилина від SStats (оновлюється live_tracker кожні 3хв)
            time_str = f"{match.elapsed}'"
        else:
            # Fallback: математика від часу початку
            now_utc = datetime.now(timezone.utc)
            elapsed_total = (now_utc - match_dt_utc).total_seconds() / 60
            if "second half" in s:
                second_half_min = int(elapsed_total - 60) + 45
                second_half_min = max(45, min(second_half_min, 99))
                time_str = f"{second_half_min}'"
            else:
                first_half_min = int(min(elapsed_total, 50))
                time_str = f"{max(1, first_half_min)}'"

    profit = round(pred.stake * (pred.odds_used - 1), 0) if pred.stake else 0

    # Derive timing from current date rather than DB field — confirm_picks only
    # runs once daily at 07:00 UTC, so DB timing lags for early-morning matches
    match_today = match.date.date() == date.today()
    timing = "final" if match_today else (pred.timing or "early")

    return {
        "id": str(pred.id),
        "league": league_name,
        "leagueFlag": flag,
        "homeTeam": home,
        "awayTeam": away,
        "homeTeamId": home_id,
        "awayTeamId": away_id,
        "time": time_str,
        "datetime_utc": datetime_utc,
        "side": format_side(pred.market, pred.outcome),
        "odds": round(pred.odds_used, 2),
        "ev": round(pred.ev * 100, 1),         # у відсотках
        "stake": pred.stake,
        "status": status,
        "timing": timing,
        "score": score,
        "pnl": pred.pnl,
        "date": match.date.date().isoformat(),
    }


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/picks")
def get_picks(
    model: str = Query("Monster"),
    date_: str = Query(None, alias="date"),
    telegram_id: int = Query(None),  # optional, reserved for future per-user filtering
):
    """
    Повертає picks для заданого дня (default: сьогодні).
    Тільки active picks для вибраної моделі.
    """
    versions = MODEL_VERSIONS.get(model)
    if not versions:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    target_date = date.fromisoformat(date_) if date_ else date.today()
    day_start = datetime.combine(target_date, datetime.min.time())
    day_end = datetime.combine(target_date, datetime.max.time())

    db = SessionLocal()
    try:
        preds = (
            db.query(Prediction)
            .join(Match, Prediction.match_id == Match.id)
            .filter(
                Prediction.model_version.in_(versions),
                Prediction.is_active.is_(True),
                Prediction.stake.isnot(None),
                Match.date >= day_start,
                Match.date <= day_end,
            )
            .order_by(Match.date)
            .all()
        )
        return {"picks": [prediction_to_pick(p, p.match) for p in preds]}
    finally:
        db.close()


@app.get("/api/stats")
def get_stats(
    model: str = Query("Monster"),
    period: str = Query("30d"),
    telegram_id: int = Query(None),
):
    """ROI, winrate, profit, streak, by_league за вибраний period."""
    versions = MODEL_VERSIONS.get(model)
    if not versions:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    days = period_to_days(period)
    since = datetime.utcnow() - timedelta(days=days)

    db = SessionLocal()
    try:
        preds = (
            db.query(Prediction)
            .join(Match, Prediction.match_id == Match.id)
            .filter(
                Prediction.model_version.in_(versions),
                Prediction.match_id.isnot(None),
                Prediction.stake.isnot(None),
                Prediction.result.isnot(None),
                Match.date >= since,
            )
            .order_by(Match.date)
            .all()
        )

        if not preds:
            return _empty_stats()

        total = len(preds)
        wins = sum(1 for p in preds if p.result == "win")
        total_stake = sum(p.stake for p in preds if p.stake)
        total_pnl = sum(p.pnl for p in preds if p.pnl is not None)
        win_rate = round(wins / total * 100, 1) if total else 0
        roi = round(total_pnl / total_stake * 100, 1) if total_stake else 0
        avg_odds = round(
            sum(p.odds_used for p in preds) / total, 2
        ) if total else 0

        # Streak (останні closed bets)
        streak = [
            {"result": p.result, "id": str(p.id)}
            for p in preds[-20:]
        ]

        # By league
        league_map: dict[str, dict] = {}
        for p in preds:
            lg = (p.match.league.name if p.match and p.match.league else None) \
                 or p.league_name or "Other"
            if lg not in league_map:
                league_map[lg] = {"bets": 0, "wins": 0, "pnl": 0.0}
            league_map[lg]["bets"] += 1
            if p.result == "win":
                league_map[lg]["wins"] += 1
            league_map[lg]["pnl"] += p.pnl or 0

        by_league = [
            {
                "league": lg,
                "flag": LEAGUE_FLAGS.get(lg, "🏟"),
                "bets": v["bets"],
                "winRate": round(v["wins"] / v["bets"] * 100, 1),
                "pnl": round(v["pnl"], 2),
            }
            for lg, v in sorted(league_map.items(), key=lambda x: -x[1]["bets"])
        ]

        # Bankroll curve: стартує з $1000, кожна ставка додає pnl
        STARTING_BALANCE = 1000.0
        curve = []
        balance = STARTING_BALANCE
        for p in preds:
            balance += p.pnl or 0
            curve.append(round(balance, 2))

        return {
            "roi": roi,
            "winRate": win_rate,
            "profit": round(total_pnl, 2),
            "balance": round(balance, 2),          # поточний баланс моделі
            "startingBalance": STARTING_BALANCE,
            "totalBets": total,
            "avgOdds": avg_odds,
            "streak": streak,
            "byLeague": by_league,
            "curve": curve,
        }
    finally:
        db.close()


def _empty_stats() -> dict:
    return {
        "roi": 0, "winRate": 0, "profit": 0,
        "totalBets": 0, "avgOdds": 0,
        "streak": [], "byLeague": [], "curve": [],
    }


@app.get("/api/bankroll")
def get_bankroll(telegram_id: int = Query(None)):
    """
    Поточний баланс + sparkline.
    Якщо telegram_id переданий — дані юзера, інакше — глобальні mock-дані.
    """
    if telegram_id is None:
        # поки немає per-user — повертаємо placeholder
        return {"balance": 1000.0, "roi": 0.0, "sparkline": [1000.0]}

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            return {"balance": 1000.0, "roi": 0.0, "sparkline": [1000.0]}

        snapshots = (
            db.query(BankrollSnapshot)
            .filter(BankrollSnapshot.user_id == user.id)
            .order_by(BankrollSnapshot.created_at.desc())
            .limit(30)
            .all()
        )
        sparkline = [s.balance for s in reversed(snapshots)] or [user.bankroll]
        start_balance = 1000.0
        roi = round((user.bankroll - start_balance) / start_balance * 100, 1)

        return {"balance": user.bankroll, "roi": roi, "sparkline": sparkline}
    finally:
        db.close()


@app.get("/api/history")
def get_history(
    model: str = Query("Monster"),
    days: int = Query(90),
):
    """
    Завершені picks (win/loss) згруповані по даті, найновіші спочатку.
    Використовується для екрану «Історія» у miniapp.
    """
    versions = MODEL_VERSIONS.get(model)
    if not versions:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    since = datetime.utcnow() - timedelta(days=days)

    db = SessionLocal()
    try:
        preds = (
            db.query(Prediction)
            .join(Match, Prediction.match_id == Match.id)
            .filter(
                Prediction.model_version.in_(versions),
                Prediction.match_id.isnot(None),
                Prediction.stake.isnot(None),
                Prediction.result.isnot(None),
                Match.date >= since,
            )
            .order_by(Match.date.desc())
            .all()
        )

        # Group by date (local date of the match)
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for p in preds:
            day = p.match.date.date().isoformat()
            groups[day].append(prediction_to_pick(p, p.match))

        # Newest date first
        dates = sorted(groups.keys(), reverse=True)
        return {
            "dates": [
                {"date": d, "picks": groups[d]}
                for d in dates
            ]
        }
    finally:
        db.close()


@app.get("/api/compare")
def get_compare(period: str = Query("30d")):
    """Порівняння всіх трьох моделей за period."""
    days = period_to_days(period)
    since = datetime.utcnow() - timedelta(days=days)

    db = SessionLocal()
    try:
        result = []
        for model_name, versions in MODEL_VERSIONS.items():
            from api.utils import MODEL_META
            color = MODEL_META.get(versions[0], {}).get("color", "#ffffff")

            preds = (
                db.query(Prediction)
                .join(Match, Prediction.match_id == Match.id)
                .filter(
                    Prediction.model_version.in_(versions),
                    Prediction.match_id.isnot(None),
                    Prediction.stake.isnot(None),
                    Prediction.result.isnot(None),
                    Match.date >= since,
                )
                .order_by(Match.date)
                .all()
            )

            total = len(preds)
            wins = sum(1 for p in preds if p.result == "win")
            total_stake = sum(p.stake for p in preds if p.stake) or 1
            total_pnl = sum(p.pnl for p in preds if p.pnl is not None)
            roi = round(total_pnl / total_stake * 100, 1) if total else 0
            win_rate = round(wins / total * 100, 1) if total else 0
            avg_odds = round(sum(p.odds_used for p in preds) / total, 2) if total else 0

            STARTING_BALANCE = 1000.0
            curve = []
            bal = STARTING_BALANCE
            for p in preds:
                bal += p.pnl or 0
                curve.append(round(bal, 2))
            if not curve:
                curve = [STARTING_BALANCE]

            result.append({
                "name": model_name,
                "color": color,
                "roi": roi,
                "winRate": win_rate,
                "bets": total,
                "avgOdds": avg_odds,
                "profit": round(total_pnl, 2),
                "curve": curve,
            })

        return {"models": result, "period": period}
    finally:
        db.close()
