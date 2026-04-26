"""
FastAPI app — 5 endpoints для Telegram miniapp.
Запускається окремим процесом через api/run.py (uvicorn, port 8000).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl

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

def verify_telegram_init_data(init_data: str) -> int | None:
    """
    Verifies Telegram WebApp initData signature (HMAC-SHA256).
    Returns telegram_id on success, None on failure or missing bot token.
    """
    from config.settings import settings
    bot_token = settings.telegram_bot_token
    if not bot_token or not init_data:
        return None
    try:
        params = dict(parse_qsl(init_data, strict_parsing=True))
        received_hash = params.pop("hash", None)
        if not received_hash:
            return None
        # Verify auth_date is within 5 minutes to reject stale tokens
        auth_date = params.get("auth_date")
        if auth_date:
            age = int(datetime.now(timezone.utc).timestamp()) - int(auth_date)
            if age > 300:
                return None
        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(params.items()))
        secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed_hash, received_hash):
            return None
        user_info = json.loads(params.get("user", "{}"))
        return user_info.get("id")
    except Exception:
        return None


def get_user_or_404(db, telegram_id: int) -> User:
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


STARTING_BALANCE = 1000.0
VALID_PERIODS = {"7d": 7, "30d": 30, "90d": 90, "all": 3650}


def period_to_days(period: str) -> int:
    days = VALID_PERIODS.get(period.lower())
    if days is None:
        raise ValueError(f"Unknown period '{period}'. Valid: {list(VALID_PERIODS)}")
    return days


def prediction_to_pick(pred: Prediction, match: Match) -> dict[str, Any]:
    """Серіалізує Prediction + Match у dict для фронту."""
    league_name = match.league.name if match.league else ""
    league_country = match.league.country if match.league else ""
    # Flag lookup is keyed by (name, country) → fall back to name-only,
    # then to a globe. This handles "Premier League" / "Champions League"
    # name collisions across countries.
    flag = (
        LEAGUE_FLAGS.get((league_name, league_country))
        or LEAGUE_FLAGS.get(league_name)
        or "🏟"
    )
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
    # runs once daily at 07:00 UTC, so DB timing lags for early-morning matches.
    # Settled picks (result not None) are always "final" regardless of date.
    match_today = match.date.date() == datetime.now(timezone.utc).date()
    if pred.result is not None:
        timing = "final"
    elif match_today:
        timing = "final"
    else:
        timing = pred.timing or "early"

    return {
        "id": str(pred.id),
        "league": league_name,
        "leagueCountry": league_country,
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
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


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

    try:
        target_date = date.fromisoformat(date_) if date_ else date.today()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_}")
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

    try:
        days = period_to_days(period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    since = datetime.now(timezone.utc) - timedelta(days=days)

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

        # By league — keyed by (name, country) so colliding names (e.g.
        # England vs Ukraine "Premier League") aggregate separately.
        league_map: dict[tuple[str, str], dict] = {}
        for p in preds:
            if p.match and p.match.league:
                lg_key = (p.match.league.name, p.match.league.country or "")
            else:
                lg_key = (p.league_name or "Other", "")
            if lg_key not in league_map:
                league_map[lg_key] = {"bets": 0, "wins": 0, "pnl": 0.0, "stake": 0.0}
            league_map[lg_key]["bets"] += 1
            if p.result == "win":
                league_map[lg_key]["wins"] += 1
            league_map[lg_key]["pnl"]   += p.pnl or 0
            league_map[lg_key]["stake"] += p.stake or 0

        by_league = [
            {
                "league":  lg_name,
                "country": lg_country,
                "flag":    LEAGUE_FLAGS.get((lg_name, lg_country))
                           or LEAGUE_FLAGS.get(lg_name)
                           or "🏟",
                "bets":    v["bets"],
                "winRate": round(v["wins"] / v["bets"] * 100, 1),
                "pnl":     round(v["pnl"], 2),
                "roi":     round(v["pnl"] / v["stake"] * 100, 1) if v["stake"] else 0.0,
            }
            for (lg_name, lg_country), v in sorted(league_map.items(), key=lambda x: -x[1]["bets"])
        ]

        # Bankroll curve: стартує з STARTING_BALANCE, кожна ставка додає pnl
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
        "balance": STARTING_BALANCE,
        "startingBalance": STARTING_BALANCE,
        "totalBets": 0, "avgOdds": 0,
        "streak": [], "byLeague": [], "curve": [STARTING_BALANCE],
    }


@app.get("/api/bankroll")
def get_bankroll(
    init_data: str = Query(None),
    telegram_id: int = Query(None),
):
    """
    Поточний баланс + sparkline.
    init_data: Telegram WebApp initData (підписаний HMAC-SHA256) — пріоритет над telegram_id.
    Якщо жодного — повертає placeholder.
    """
    # Preferred: verify initData signature → extract telegram_id
    if init_data:
        verified_id = verify_telegram_init_data(init_data)
        if verified_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired initData")
        telegram_id = verified_id
    elif telegram_id is not None:
        # Direct telegram_id without initData: allowed only in dev mode
        from config.settings import settings
        if settings.env == "production":
            raise HTTPException(status_code=401, detail="initData required in production")

    if telegram_id is None:
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
    days: int = Query(90, ge=1, le=365),
):
    """
    Завершені picks (win/loss) згруповані по даті, найновіші спочатку.
    Використовується для екрану «Історія» у miniapp.
    """
    versions = MODEL_VERSIONS.get(model)
    if not versions:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    since = datetime.now(timezone.utc) - timedelta(days=days)

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
            .limit(2000)
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


@app.get("/api/clv")
def get_clv(
    model: str = Query(..., description="Display name: WS Gap | Monster | Aqua | Pure | Gem"),
    days: int = Query(30, ge=1, le=365, description="Rolling window in days"),
):
    """
    Returns rolling-window CLV stats + a daily-average trend series for one
    model. Mirrors the daily metric used by the CLV alert monitor.

    Response:
        {
          "model":   "Gem",
          "days":    30,
          "avg_clv": 0.012,        # weighted avg over the window
          "n_picks": 47,
          "pos_rate": 0.55,        # share of picks with CLV > 0
          "trend":   [             # daily averages (most recent last)
              {"date": "2026-04-01", "avg_clv": 0.014, "n": 3},
              ...
          ]
        }
    """
    versions = MODEL_VERSIONS.get(model)
    if not versions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Valid: {list(MODEL_VERSIONS)}",
        )
    since = datetime.now(timezone.utc) - timedelta(days=days)

    db = SessionLocal()
    try:
        rows = (
            db.query(Prediction, Match)
            .outerjoin(Match, Prediction.match_id == Match.id)
            .filter(
                Prediction.model_version.in_(versions),
                Prediction.clv.isnot(None),
            )
            .all()
        )

        # Bucket by match date (UTC) and aggregate.
        buckets: dict[date, list[float]] = {}
        for pred, match in rows:
            d: date | None = None
            if match and match.date is not None:
                md = match.date
                if md.tzinfo is None:
                    md = md.replace(tzinfo=timezone.utc)
                if md < since:
                    continue
                d = md.date()
            elif pred.match_date is not None:
                md_dt = datetime(
                    pred.match_date.year, pred.match_date.month, pred.match_date.day,
                    tzinfo=timezone.utc,
                )
                if md_dt < since:
                    continue
                d = pred.match_date
            if d is None:
                continue
            buckets.setdefault(d, []).append(float(pred.clv))

        all_clvs = [v for vals in buckets.values() for v in vals]
        n = len(all_clvs)
        avg_clv = round(sum(all_clvs) / n, 4) if n else 0.0
        pos_rate = round(sum(1 for v in all_clvs if v > 0) / n, 4) if n else 0.0

        trend = [
            {
                "date":    d.isoformat(),
                "avg_clv": round(sum(vals) / len(vals), 4),
                "n":       len(vals),
            }
            for d, vals in sorted(buckets.items())
        ]

        return {
            "model":    model,
            "days":     days,
            "avg_clv":  avg_clv,
            "n_picks":  n,
            "pos_rate": pos_rate,
            "trend":    trend,
        }
    finally:
        db.close()


@app.get("/api/compare")
def get_compare(period: str = Query("30d")):
    """Порівняння всіх трьох моделей за period."""
    try:
        days = period_to_days(period)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    since = datetime.now(timezone.utc) - timedelta(days=days)

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
