from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from config.settings import settings
from db.models import Match, Odds, Prediction, Team
from db.session import SessionLocal
from model.features.builder import build_match_features
from model.features.elo import compute_dynamic_elo
from model.predict import predict_match

MODEL_VERSION = "v1"


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
        "home_shots_on_target":  s.home_shots_on_target,
        "away_shots_on_target":  s.away_shots_on_target,
        "home_shots_inside_box": s.home_shots_inside_box,
        "away_shots_inside_box": s.away_shots_inside_box,
        "home_possession":       s.home_possession,
        "away_possession":       s.away_possession,
        "home_corners":          s.home_corners,
        "away_corners":          s.away_corners,
        "home_gk_saves":         s.home_gk_saves,
        "away_gk_saves":         s.away_gk_saves,
        "home_passes_accurate":  s.home_passes_accurate,
        "away_passes_accurate":  s.away_passes_accurate,
    } for s in stats])


def _get_odds_for_match(match_id: int, db) -> dict | None:
    odds = db.query(Odds).filter_by(match_id=match_id, market="1x2", is_closing=False).all()
    if not odds:
        return None
    return {o.outcome: o.value for o in odds}


def run_generate_picks() -> None:
    """
    Генерує picks для матчів що починаються через ~PICKS_HOURS_BEFORE годин.
    Запускається щогодини. Уникає дублікатів через перевірку існуючих predictions.
    """
    now = datetime.now(timezone.utc)
    window_start = now + timedelta(hours=settings.picks_hours_before - 0.5)
    window_end = now + timedelta(hours=settings.picks_hours_before + 0.5)

    logger.info(
        f"Generating picks for matches {settings.picks_hours_before}h before start "
        f"(window: {window_start.strftime('%H:%M')} – {window_end.strftime('%H:%M')} UTC)"
    )

    db = SessionLocal()
    try:
        matches = db.query(Match).filter(
            Match.date >= window_start.replace(tzinfo=None),
            Match.date <= window_end.replace(tzinfo=None),
            Match.status == "Not Started",
        ).all()

        if not matches:
            logger.info("No matches in window, skipping")
            return

        logger.info(f"Found {len(matches)} matches in window")

        matches_df = _load_matches_df(db)
        stats_df = _load_stats_df(db)
        teams = _load_teams_elo(matches_df)

        from db.models import InjuryReport
        inj_rows = db.query(InjuryReport).all()
        injuries_df = pd.DataFrame([{
            "match_id": i.match_id,
            "team_id": i.team_id,
            "player_api_id": i.player_api_id,
        } for i in inj_rows]) if inj_rows else pd.DataFrame(columns=["match_id", "team_id", "player_api_id"])

        new_picks = []

        for match in matches:
            # Уникаємо дублікатів
            existing = db.query(Prediction).filter_by(
                match_id=match.id, model_version=MODEL_VERSION
            ).first()
            if existing:
                logger.debug(f"Match {match.id} already has prediction, skipping")
                continue

            odds = _get_odds_for_match(match.id, db)
            if not odds:
                logger.warning(f"No odds for match {match.id}, skipping")
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

            picks = predict_match(
                features=features,
                odds=odds,
                bankroll=settings.bankroll,
                version=MODEL_VERSION,
            )

            for pick in picks:
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
            logger.info(f"Generated {len(new_picks)} picks")
            _send_picks_to_telegram(new_picks)
        else:
            logger.info("No new picks generated")

    finally:
        db.close()


def _send_picks_to_telegram(picks: list) -> None:
    """Надсилає нові піки в Telegram одразу після генерації."""
    import asyncio
    from aiogram import Bot
    from db.models import User
    from db.session import SessionLocal

    async def _send():
        if not settings.telegram_bot_token:
            logger.warning("No Telegram token configured, skipping send")
            return

        bot = Bot(token=settings.telegram_bot_token)
        db = SessionLocal()
        try:
            users = db.query(User).filter_by(is_active=True).all()
            if not users:
                logger.info("No active users to send picks to")
                return

            lines = []
            for match, pick in picks:
                home = match.home_team.name if match.home_team else "?"
                away = match.away_team.name if match.away_team else "?"
                outcome_label = {"home": f"П1 ({home})", "away": f"П2 ({away})"}.get(pick["outcome"], pick["outcome"])
                time_str = match.date.strftime("%d.%m %H:%M")
                lines.append(
                    f"⚽ {home} — {away} [{time_str}]\n"
                    f"  ➤ {outcome_label}\n"
                    f"  Коеф: {pick['odds']:.2f} | EV: {pick['ev']*100:.1f}% | WS: {pick['weighted_score']}\n"
                    f"  Стейк: ${pick['stake']:.0f} з ${settings.bankroll:.0f}"
                )

            message = f"🎯 {len(picks)} нових пік{'и' if len(picks) > 1 else 'а'}:\n\n" + "\n\n".join(lines)

            sent = 0
            for user in users:
                try:
                    await bot.send_message(user.telegram_id, message)
                    sent += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {user.telegram_id}: {e}")

            logger.info(f"Picks sent to {sent}/{len(users)} users")
        finally:
            db.close()
            await bot.session.close()

    asyncio.run(_send())
