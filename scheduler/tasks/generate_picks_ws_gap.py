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

MODEL_VERSION             = "ws_gap_v1"         # фінальний, 4% cap
MODEL_VERSION_KELLY       = "ws_gap_kelly_v1"    # фінальний, pure Kelly 25%
MODEL_VERSION_EARLY       = "ws_gap_v1_early"    # ранній, 4% cap
MODEL_VERSION_KELLY_EARLY = "ws_gap_kelly_v1_early"  # ранній, pure Kelly 25%

LEAGUE_FLAGS = {
    "England":     "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain":       "🇪🇸",
    "Germany":     "🇩🇪",
    "Italy":       "🇮🇹",
    "France":      "🇫🇷",
    "Europe":      "🏆",
    "Netherlands": "🇳🇱",
    "Portugal":    "🇵🇹",
    "Belgium":     "🇧🇪",
}


def _league_flag(match) -> str:
    if match.league and match.league.country:
        return LEAGUE_FLAGS.get(match.league.country, "🌍")
    return "🌍"
WS_GAP_MIN    = 70   # оновлено з ablation (DOM видалено — не додає цінності)
ODDS_MIN      = 2.0  # оновлено: odds 2.0-2.5 стабільно прибуткові
KELLY_CAP     = 0.10
FRACTIONAL    = 0.25

# Whitelist ліг для генерації піків
ALLOWED_LEAGUE_API_IDS = {
    39,   # Premier League (England)
    140,  # La Liga (Spain)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    61,   # Ligue 1 (France)
    2,    # Champions League
    88,   # Eredivisie (Netherlands)
    144,  # Jupiler Pro League (Belgium)
    333,  # Premier League (Ukraine)
}


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


def _ws_gap_pick(features: dict, odds: dict, bankroll: float) -> dict | None:
    """
    Основна логіка WS Gap:
    1. Рахуємо ws_home і ws_away
    2. dominant = max(ws_home, ws_away)
    3. Якщо gap >= WS_GAP_MIN і odds в діапазоні → pick
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
    if odds_val < ODDS_MIN:
        logger.debug(f"Odds {odds_val:.2f} < {ODDS_MIN}, skip")
        return None

    # Kelly sizing через Elo
    b = odds_val - 1
    q = 1 - p_elo
    kelly = max(0.0, (p_elo * b - q) / b) * FRACTIONAL
    stake = round(bankroll * kelly, 2) if bankroll > 0 else 0.0

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


def _compute_current_bankroll(db, initial: float, model_ver: str) -> float:
    """
    Рахує поточний банкрол з урахуванням усіх завершених ставок моделі.
    """
    from db.models import Match as MatchModel
    FINISHED = {"Finished", "FT", "finished", "ft", "Match Finished"}

    preds = (
        db.query(Prediction)
        .join(MatchModel)
        .filter(
            Prediction.model_version == model_ver,
            MatchModel.status.in_(FINISHED),
            MatchModel.home_score.isnot(None),
        )
        .order_by(MatchModel.date.asc())
        .all()
    )

    bankroll = initial
    for pred in preds:
        m = pred.match
        kf = pred.kelly_fraction or 0
        stake = min(bankroll * kf, bankroll * KELLY_CAP)
        hs, as_ = m.home_score, m.away_score
        won = (pred.outcome == "home" and hs > as_) or (pred.outcome == "away" and as_ > hs)
        if won:
            bankroll += stake * (pred.odds_used - 1)
        else:
            bankroll -= stake

    return round(bankroll, 2)


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

        from db.models import League as LeagueModel

        matches = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status.in_(status_filter),
            LeagueModel.api_id.in_(ALLOWED_LEAGUE_API_IDS),
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

        # Поточний банкрол для кожної версії (compound)
        bankroll_cap   = _compute_current_bankroll(db, settings.bankroll, MODEL_VERSION)
        bankroll_kelly = _compute_current_bankroll(db, settings.bankroll, MODEL_VERSION_KELLY)
        logger.info(
            f"[WS Gap] Bankroll cap=${bankroll_cap:.0f} kelly=${bankroll_kelly:.0f} "
            f"(initial=${settings.bankroll:.0f})"
        )

        new_picks = []

        # Batch-load existing predictions and odds to avoid N+1 queries
        match_ids = [m.id for m in matches]
        existing_preds = db.query(Prediction).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version.in_([MODEL_VERSION, MODEL_VERSION_KELLY]),
        ).all()
        existing_cap   = {p.match_id for p in existing_preds if p.model_version == MODEL_VERSION}
        existing_kelly = {p.match_id for p in existing_preds if p.model_version == MODEL_VERSION_KELLY}

        all_odds_rows = db.query(Odds).filter(
            Odds.match_id.in_(match_ids),
            Odds.market == "1x2",
            Odds.is_closing.is_(False),
        ).all()
        odds_by_match: dict[int, dict] = {}
        for o in all_odds_rows:
            odds_by_match.setdefault(o.match_id, {})[o.outcome] = o.value

        for match in matches:
            # Пропускаємо якщо обидві версії вже є
            has_cap   = match.id in existing_cap
            has_kelly = match.id in existing_kelly
            if has_cap and has_kelly:
                logger.debug(f"[WS Gap] Match {match.id} already predicted, skip")
                continue

            odds = odds_by_match.get(match.id)
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

            # Генеруємо пік (stake = pure Kelly)
            pick = _ws_gap_pick(features, odds, max(bankroll_cap, bankroll_kelly))
            if pick is None:
                continue

            # Версія 1: 4% cap
            if not has_cap:
                stake_cap = round(min(bankroll_cap * pick["kelly_fraction"], bankroll_cap * KELLY_CAP), 2)
                db.add(Prediction(
                    match_id=match.id, market="1x2",
                    outcome=pick["outcome"],
                    probability=float(pick["probability"]),
                    odds_used=float(pick["odds"]),
                    ev=float(pick["ev"]),
                    kelly_fraction=float(pick["kelly_fraction"]),
                    stake=stake_cap,
                    weighted_score=int(pick["weighted_score"]),
                    model_version=MODEL_VERSION,
                ))

            # Версія 2: pure Kelly 25%
            if not has_kelly:
                stake_kelly = round(min(bankroll_kelly * pick["kelly_fraction"], bankroll_kelly * KELLY_CAP), 2)
                db.add(Prediction(
                    match_id=match.id, market="1x2",
                    outcome=pick["outcome"],
                    probability=float(pick["probability"]),
                    odds_used=float(pick["odds"]),
                    ev=float(pick["ev"]),
                    kelly_fraction=float(pick["kelly_fraction"]),
                    stake=stake_kelly,
                    weighted_score=int(pick["weighted_score"]),
                    model_version=MODEL_VERSION_KELLY,
                ))

            new_picks.append((match, pick))

        db.commit()

        if new_picks:
            logger.info(f"[WS Gap] Generated {len(new_picks)} picks (both versions)")
            _send_picks_to_telegram(new_picks, phase="final", bankroll=bankroll_cap)
        else:
            logger.info("[WS Gap] No new picks")

    finally:
        db.close()


def _send_picks_to_telegram(picks: list, phase: str = "final", bankroll: float | None = None) -> None:
    """
    phase: "final" | "early"
    bankroll: поточний compound банкрол (для відображення в повідомленні)
    """
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

            br = bankroll or settings.bankroll
            lines = []
            for match, pick in picks:
                flag = _league_flag(match)
                home = match.home_team.name if match.home_team else "?"
                away = match.away_team.name if match.away_team else "?"
                outcome_label = (
                    f"П1 ({home})" if pick["outcome"] == "home" else f"П2 ({away})"
                )
                time_str = match.date.strftime("%d.%m %H:%M")
                stake_cap = round(min(br * pick["kelly_fraction"], br * KELLY_CAP), 2)
                lines.append(
                    f"{flag} {home} — {away} [{time_str}]\n"
                    f"  ➤ {outcome_label}\n"
                    f"  Коэф: {pick['odds']:.2f} | EV: {pick['ev']*100:.1f}% | "
                    f"WS: {pick['weighted_score']} | Gap: {pick['ws_gap']}\n"
                    f"  Стейк: ${stake_cap:.0f} (банкрол: ${br:.0f})"
                )

            n = len(picks)
            picks_word = "пика" if n in (2, 3, 4) else ("пик" if n == 1 else "пиков")
            if phase == "early":
                header = f"📅 Ранние пики — {n} {picks_word} (до {settings.early_picks_days_ahead} дней):\n\n"
            else:
                header = f"🔔 Финальный пик — {n} {picks_word} (старт ~{settings.picks_hours_before}ч):\n\n"

            message = header + "\n\n".join(lines)

            sent = 0
            for user in users:
                try:
                    await bot.send_message(user.telegram_id, message)
                    sent += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {user.telegram_id}: {e}")

            logger.info(f"[WS Gap] [{phase}] Picks sent to {sent}/{len(users)} users")
        finally:
            db.close()
            await bot.session.close()

    asyncio.run(_send())


def run_early_picks_scan() -> None:
    """
    Phase 1 — щодня о 9:05 UTC.
    Генерує ранні піки для матчів від picks_hours_before+1h до early_picks_days_ahead днів вперед.
    Зберігає з model_version *_early. Не перезаписує вже наявні (early або final).
    """
    now = datetime.now(timezone.utc)
    scan_from = now + timedelta(hours=settings.picks_hours_before + 1)
    scan_to   = now + timedelta(days=settings.early_picks_days_ahead)

    logger.info(
        f"[WS Gap Early] Scanning | window: "
        f"{scan_from.strftime('%d.%m %H:%M')} – {scan_to.strftime('%d.%m %H:%M')} UTC"
    )

    db = SessionLocal()
    try:
        from db.models import League as LeagueModel

        matches = db.query(Match).join(LeagueModel).filter(
            Match.date >= scan_from.replace(tzinfo=None),
            Match.date <= scan_to.replace(tzinfo=None),
            Match.status == "Not Started",
            LeagueModel.api_id.in_(ALLOWED_LEAGUE_API_IDS),
        ).all()

        if not matches:
            logger.info("[WS Gap Early] No matches in window, skipping")
            return

        logger.info(f"[WS Gap Early] Found {len(matches)} matches")

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

        bankroll_cap   = _compute_current_bankroll(db, settings.bankroll, MODEL_VERSION)
        bankroll_kelly = _compute_current_bankroll(db, settings.bankroll, MODEL_VERSION_KELLY)

        new_picks = []

        # Batch-load existing predictions and odds to avoid N+1 queries
        early_match_ids = [m.id for m in matches]
        early_existing = db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(early_match_ids),
            Prediction.model_version.in_([
                MODEL_VERSION, MODEL_VERSION_KELLY,
                MODEL_VERSION_EARLY, MODEL_VERSION_KELLY_EARLY,
            ]),
        ).all()
        early_has_pick = {row.match_id for row in early_existing}

        early_odds_rows = db.query(Odds).filter(
            Odds.match_id.in_(early_match_ids),
            Odds.market == "1x2",
            Odds.is_closing.is_(False),
        ).all()
        early_odds_by_match: dict[int, dict] = {}
        for o in early_odds_rows:
            early_odds_by_match.setdefault(o.match_id, {})[o.outcome] = o.value

        for match in matches:
            # Пропускаємо якщо є будь-який пік (early або final) для цього матчу
            if match.id in early_has_pick:
                logger.debug(f"[WS Gap Early] Match {match.id} already has a pick, skip")
                continue

            odds = early_odds_by_match.get(match.id)
            if not odds:
                logger.debug(f"[WS Gap Early] No odds for match {match.id}, skip")
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

            pick = _ws_gap_pick(features, odds, max(bankroll_cap, bankroll_kelly))
            if pick is None:
                continue

            stake_cap   = round(min(bankroll_cap * pick["kelly_fraction"], bankroll_cap * KELLY_CAP), 2)
            stake_kelly = round(bankroll_kelly * pick["kelly_fraction"], 2)

            db.add(Prediction(
                match_id=match.id, market="1x2",
                outcome=pick["outcome"],
                probability=float(pick["probability"]),
                odds_used=float(pick["odds"]),
                ev=float(pick["ev"]),
                kelly_fraction=float(pick["kelly_fraction"]),
                stake=stake_cap,
                weighted_score=int(pick["weighted_score"]),
                model_version=MODEL_VERSION_EARLY,
            ))
            db.add(Prediction(
                match_id=match.id, market="1x2",
                outcome=pick["outcome"],
                probability=float(pick["probability"]),
                odds_used=float(pick["odds"]),
                ev=float(pick["ev"]),
                kelly_fraction=float(pick["kelly_fraction"]),
                stake=stake_kelly,
                weighted_score=int(pick["weighted_score"]),
                model_version=MODEL_VERSION_KELLY_EARLY,
            ))

            new_picks.append((match, pick))

        db.commit()

        if new_picks:
            logger.info(f"[WS Gap Early] Generated {len(new_picks)} early picks")
            _send_picks_to_telegram(new_picks, phase="early", bankroll=bankroll_cap)
        else:
            logger.info("[WS Gap Early] No new early picks")

    finally:
        db.close()
