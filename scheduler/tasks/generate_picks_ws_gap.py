"""
scheduler/tasks/generate_picks_ws_gap.py

WS Gap model — picks без ML.
Логіка: ws_gap >= 80, ws_dominant >= 80, odds >= 1.7 → ставка на домінуючу команду.
Sizing: Elo-based Kelly, 25% fractional, 4% cap.
model_version = "ws_gap_v1"
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from config.settings import settings
from db.models import Match, Odds, Prediction, Team
from db.session import SessionLocal
from model.features.builder import build_match_features
from model.features.elo import compute_dynamic_elo
from model.weighted_score import compute_weighted_score

MODEL_VERSION = "ws_gap_v1"
WS_GAP_MIN    = 80
WS_DOM_MIN    = 80
ODDS_MIN      = 1.7
KELLY_CAP     = 0.04
FRACTIONAL    = 0.25


def _load_teams_elo(matches_df: pd.DataFrame) -> dict:
    """Рахує актуальний Elo з матчів (Team.elo в БД не оновлюється)."""
    elos = compute_dynamic_elo(matches_df)
    return {team_id: {"elo": elo} for team_id, elo in elos.items()}


def _load_matches_df(db) -> pd.DataFrame:
    matches = db.query(Match).filter(Match.status == "Finished").all()
    return pd.DataFrame([{
        "id": m.id,
        "date": pd.Timestamp(m.date),
        "home_team_id": m.home_team_id,
        "away_team_id": m.away_team_id,
        "home_score": m.home_score,
        "away_score": m.away_score,
        "league_id": m.league_id,
    } for m in matches])


def _load_stats_df(db) -> pd.DataFrame:
    from db.models import MatchStats
    stats = db.query(MatchStats).join(Match).all()
    return pd.DataFrame([{
        "match_id": s.match_id,
        "date": pd.Timestamp(s.match.date),
        "home_team_id": s.match.home_team_id,
        "away_team_id": s.match.away_team_id,
        "home_score": s.match.home_score,
        "away_score": s.match.away_score,
        "home_xg": s.home_xg,
        "away_xg": s.away_xg,
    } for s in stats])


def _get_odds_for_match(match_id: int, db) -> dict | None:
    odds = db.query(Odds).filter_by(match_id=match_id, market="1x2", is_closing=False).all()
    if not odds:
        return None
    return {o.outcome: o.value for o in odds}


def _ws_gap_pick(features: dict, odds: dict, bankroll: float) -> dict | None:
    """
    Основна логіка WS Gap:
    1. Рахуємо ws_home і ws_away
    2. dominant = max(ws_home, ws_away)
    3. Якщо gap >= WS_GAP_MIN і dominant >= WS_DOM_MIN і odds >= ODDS_MIN → pick
    4. Kelly sizing через Elo probability

    odds dict містить decimal odds: {"home": 1.85, "draw": 3.2, "away": 4.5}
    """
    h_odds = float(odds.get("home") or 0)
    a_odds = float(odds.get("away") or 0)
    if not h_odds or not a_odds:
        return None

    try:
        ws_h = compute_weighted_score(features, "home")
        ws_a = compute_weighted_score(features, "away")
    except Exception as e:
        logger.warning(f"WS computation failed: {e}")
        return None

    if ws_h >= ws_a:
        dominant_side = "home"
        ws_dom, ws_weak = ws_h, ws_a
        odds_val = round(h_odds, 2)
        p_elo = float(features.get("elo_home_win_prob") or 0.5)
    else:
        dominant_side = "away"
        ws_dom, ws_weak = ws_a, ws_h
        odds_val = round(a_odds, 2)
        p_elo = 1.0 - float(features.get("elo_home_win_prob") or 0.5)

    ws_gap = ws_dom - ws_weak

    if ws_gap < WS_GAP_MIN:
        logger.debug(f"WS gap {ws_gap:.0f} < {WS_GAP_MIN}, skip")
        return None
    if ws_dom < WS_DOM_MIN:
        logger.debug(f"WS dominant {ws_dom:.0f} < {WS_DOM_MIN}, skip")
        return None
    if odds_val < ODDS_MIN:
        logger.debug(f"Odds {odds_val:.2f} < {ODDS_MIN}, skip")
        return None

    # Kelly sizing через Elo
    b = odds_val - 1
    q = 1 - p_elo
    kelly = max(0.0, (p_elo * b - q) / b) * FRACTIONAL
    stake = round(min(bankroll * kelly, bankroll * KELLY_CAP), 2) if bankroll > 0 else 0.0

    # EV через Elo prob (не ринок)
    ev = round(p_elo * odds_val - 1, 4)

    logger.info(
        f"WS Gap pick: {dominant_side} | ws_dom={ws_dom:.0f} ws_gap={ws_gap:.0f} "
        f"odds={odds_val:.2f} p_elo={p_elo:.3f} EV={ev*100:.1f}% stake=${stake:.0f}"
    )

    return {
        "outcome":       dominant_side,
        "probability":   round(p_elo, 4),
        "odds":          odds_val,
        "ev":            ev,
        "kelly_fraction": round(kelly, 4),
        "stake":         stake,
        "weighted_score": int(ws_dom),
        "ws_gap":        int(ws_gap),
    }


def run_generate_picks_ws_gap(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
    include_finished: bool = False,
) -> None:
    """
    Генерує WS Gap picks.

    За замовчуванням: матчі що починаються через ~picks_hours_before годин.
    При include_finished=True: також захоплює вже завершені матчі у вказаному діапазоні
    (використовується для бекфілу).
    """
    now = datetime.now(timezone.utc)

    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(hours=settings.picks_hours_before + 0.5)

    logger.info(
        f"[WS Gap] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    db = SessionLocal()
    try:
        status_filter = ["Not Started"]
        if include_finished:
            status_filter += ["Finished", "Match Finished"]

        matches = db.query(Match).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status.in_(status_filter),
        ).all()

        if not matches:
            logger.info("[WS Gap] No matches in window, skipping")
            return

        logger.info(f"[WS Gap] Found {len(matches)} matches")

        matches_df  = _load_matches_df(db)
        stats_df    = _load_stats_df(db)
        teams       = _load_teams_elo(matches_df)

        from db.models import InjuryReport
        inj_rows = db.query(InjuryReport).all()
        injuries_df = pd.DataFrame([{
            "match_id": i.match_id,
            "team_id": i.team_id,
            "player_api_id": i.player_api_id,
        } for i in inj_rows]) if inj_rows else pd.DataFrame(
            columns=["match_id", "team_id", "player_api_id"]
        )

        new_picks = []

        for match in matches:
            existing = db.query(Prediction).filter_by(
                match_id=match.id, model_version=MODEL_VERSION
            ).first()
            if existing:
                logger.debug(f"[WS Gap] Match {match.id} already predicted, skip")
                continue

            odds = _get_odds_for_match(match.id, db)
            if not odds:
                logger.warning(f"[WS Gap] No odds for match {match.id}, skip")
                continue

            features = build_match_features(
                match={
                    "id": match.id,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "date": match.date,
                    "league_id": match.league_id,
                },
                matches_df=matches_df,
                stats_df=stats_df,
                teams=teams,
                odds=odds,
                injuries_df=injuries_df,
            )

            pick = _ws_gap_pick(features, odds, settings.bankroll)
            if pick is None:
                continue

            pred = Prediction(
                match_id=match.id,
                market="1x2",
                outcome=pick["outcome"],
                probability=float(pick["probability"]),
                odds_used=float(pick["odds"]),
                ev=float(pick["ev"]),
                kelly_fraction=float(pick["kelly_fraction"]),
                stake=float(pick["stake"]),
                weighted_score=int(pick["weighted_score"]),
                model_version=MODEL_VERSION,
            )
            db.add(pred)
            new_picks.append((match, pick))

        db.commit()

        if new_picks:
            logger.info(f"[WS Gap] Generated {len(new_picks)} picks")
            _send_picks_to_telegram(new_picks)
        else:
            logger.info("[WS Gap] No new picks")

    finally:
        db.close()


def _send_picks_to_telegram(picks: list) -> None:
    import asyncio
    from aiogram import Bot
    from db.models import User
    from db.session import SessionLocal

    async def _send():
        if not settings.telegram_bot_token:
            logger.warning("No Telegram token, skipping send")
            return

        bot = Bot(token=settings.telegram_bot_token)
        db = SessionLocal()
        try:
            users = db.query(User).filter_by(is_active=True).all()
            if not users:
                return

            lines = []
            for match, pick in picks:
                home = match.home_team.name if match.home_team else "?"
                away = match.away_team.name if match.away_team else "?"
                outcome_label = (
                    f"П1 ({home})" if pick["outcome"] == "home" else f"П2 ({away})"
                )
                time_str = match.date.strftime("%d.%m %H:%M")
                lines.append(
                    f"⚽ {home} — {away} [{time_str}]\n"
                    f"  ➤ {outcome_label}\n"
                    f"  Коеф: {pick['odds']:.2f} | EV: {pick['ev']*100:.1f}% | "
                    f"WS: {pick['weighted_score']} | Gap: {pick['ws_gap']}\n"
                    f"  Стейк: ${pick['stake']:.0f} з ${settings.bankroll:.0f}"
                )

            header = f"📐 WS Gap — {len(picks)} пік{'и' if len(picks) > 1 else 'а'}:\n\n"
            message = header + "\n\n".join(lines)

            sent = 0
            for user in users:
                try:
                    await bot.send_message(user.telegram_id, message)
                    sent += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {user.telegram_id}: {e}")

            logger.info(f"[WS Gap] Picks sent to {sent}/{len(users)} users")
        finally:
            db.close()
            await bot.session.close()

    asyncio.run(_send())
