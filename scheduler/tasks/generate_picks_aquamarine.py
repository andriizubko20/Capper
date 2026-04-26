"""
scheduler/tasks/generate_picks_aquamarine.py

Aquamarine model — niche-based picks.
Same infrastructure as Monster; different niche set.
model_version = "aquamarine_v1_kelly"
"""
from datetime import datetime, timedelta, timezone

from loguru import logger

from config.settings import settings
from data.best_odds import best_1x2_odds_dict
from db.models import Match, Prediction
from db.session import SessionLocal
from model.aquamarine.niches import MODELS, OOS_START, HIGH_RISK_ODDS, parse_niche, match_niche
from model.monster.niches import LEAGUE_API_IDS as AQUAMARINE_LEAGUE_API_IDS
from model.monster.features import (
    load_historical_data, build_team_state, build_upcoming_features, compute_p_is
)

MODEL_VERSION = "aquamarine_v1_kelly"

KELLY_FRAC  = 0.25
KELLY_CAP   = 0.10

AQUAMARINE_LEAGUE_NAMES = set(MODELS.keys())

# In-memory p_is cache (cleared on restart)
_P_IS_CACHE: dict[tuple, float] = {}


def _get_p_is(matches, stats, odds_df, league, niche_str) -> float | None:
    key = (league, niche_str)
    if key in _P_IS_CACHE:
        return _P_IS_CACHE[key]
    niche = parse_niche(niche_str)
    p = compute_p_is(matches, stats, odds_df, league, niche, OOS_START)
    if p is not None:
        _P_IS_CACHE[key] = p
    return p


def _get_odds(match_id: int, db) -> dict | None:
    """Best 1x2 odds per outcome across ALL bookmakers (bookmaker shopping)."""
    out = best_1x2_odds_dict(db, match_id, is_closing=False)
    return out or None


def _compute_bankroll(db, initial: float, model_ver: str) -> float:
    FINISHED = {'Finished', 'FT', 'finished', 'ft', 'Match Finished'}
    preds = (
        db.query(Prediction)
        .join(Match)
        .filter(
            Prediction.model_version == model_ver,
            Match.status.in_(FINISHED),
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.asc())
        .all()
    )
    bankroll = initial
    for pred in preds:
        m = pred.match
        kf = pred.kelly_fraction or 0
        stake = min(bankroll * kf, bankroll * KELLY_CAP)
        hs, as_ = m.home_score, m.away_score
        won = (pred.outcome == 'home' and hs > as_) or (pred.outcome == 'away' and as_ > hs)
        bankroll += stake * (pred.odds_used - 1) if won else -stake
    return round(bankroll, 2)


def _last10_stats(db, model_ver: str) -> dict:
    FINISHED = {'Finished', 'FT', 'finished', 'ft', 'Match Finished'}
    preds = (
        db.query(Prediction)
        .join(Match)
        .filter(
            Prediction.model_version == model_ver,
            Match.status.in_(FINISHED),
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(10)
        .all()
    )
    if not preds:
        return {'n': 0, 'wins': 0, 'roi': 0.0}
    wins = 0
    flat_pnl = 0.0
    for pred in preds:
        m = pred.match
        won = (pred.outcome == 'home' and m.home_score > m.away_score) or \
              (pred.outcome == 'away' and m.away_score > m.home_score)
        wins += int(won)
        flat_pnl += (pred.odds_used - 1) if won else -1.0
    n = len(preds)
    return {'n': n, 'wins': wins, 'roi': flat_pnl / n * 100}


def run_generate_picks_aquamarine(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(days=settings.early_picks_days_ahead)

    logger.info(
        f"[Aquamarine] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    db = SessionLocal()
    try:
        from db.models import League as LeagueModel

        matches_db = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status == 'Not Started',
            LeagueModel.name.in_(AQUAMARINE_LEAGUE_NAMES),
            LeagueModel.api_id.in_(AQUAMARINE_LEAGUE_API_IDS),
        ).all()

        if not matches_db:
            logger.info('[Aquamarine] No matches in window, skipping')
            return

        logger.info(f'[Aquamarine] Found {len(matches_db)} matches')

        logger.info('[Aquamarine] Loading historical data...')
        hist_matches, hist_stats, hist_odds = load_historical_data()
        team_state = build_team_state(hist_matches, hist_stats)

        bankroll = _compute_bankroll(db, settings.bankroll, MODEL_VERSION)
        logger.info(f'[Aquamarine] Bankroll=${bankroll:.0f}')

        new_picks = []

        for match in matches_db:
            if db.query(Prediction).filter_by(match_id=match.id, model_version=MODEL_VERSION).first():
                continue

            odds_raw = _get_odds(match.id, db)
            if not odds_raw:
                logger.debug(f'[Aquamarine] No odds for match {match.id}, skip')
                continue

            league_name = match.league.name if match.league else ''
            aqua_league = None
            for al in AQUAMARINE_LEAGUE_NAMES:
                if al == league_name:
                    aqua_league = al
                    break
                if 'Champions League' in league_name and al == 'Champions League':
                    aqua_league = al
                    break
            if not aqua_league:
                continue

            features = build_upcoming_features(
                match={
                    'home_team_id': match.home_team_id,
                    'away_team_id': match.away_team_id,
                    'league_name': aqua_league,
                },
                team_state=team_state,
                odds=odds_raw,
            )

            best = None
            best_p_is = None

            for niche_str in MODELS[aqua_league]:
                niche = parse_niche(niche_str)
                if not match_niche(features, niche, aqua_league, aqua_league):
                    continue
                p_is = _get_p_is(hist_matches, hist_stats, hist_odds, aqua_league, niche_str)
                if p_is is None:
                    continue
                if best_p_is is None or p_is > best_p_is:
                    best = (niche_str, niche, p_is)
                    best_p_is = p_is

            if best is None:
                continue

            niche_str, niche, p_is = best
            side = niche['side']
            odds_val = features['home_odds'] if side == 'home' else features['away_odds']
            if not odds_val:
                continue

            b = odds_val - 1.0
            f_star = max(0.0, (p_is * b - (1 - p_is)) / b) if b > 0 else 0.0
            if f_star <= 0:
                logger.debug(f'[Aquamarine] Negative Kelly for {match.id} niche={niche_str}, skip')
                continue

            kelly_frac = KELLY_FRAC * f_star
            ev = round(p_is * odds_val - 1, 4)

            home = match.home_team.name if match.home_team else '?'
            away = match.away_team.name if match.away_team else '?'
            logger.info(
                f'[Aquamarine] Pick: {home} vs {away} → {side} | niche={niche_str} '
                f'odds={odds_val:.2f} p_is={p_is:.3f} f*={f_star:.3f} EV={ev*100:.1f}%'
            )

            stake = round(min(bankroll * kelly_frac, bankroll * KELLY_CAP), 2)
            db.add(Prediction(
                match_id=match.id, market='1x2',
                outcome=side,
                probability=round(p_is, 4),
                odds_used=float(odds_val),
                ev=ev,
                kelly_fraction=round(kelly_frac, 4),
                stake=stake,
                model_version=MODEL_VERSION,
                league_name=match.league.name if match.league else None,
                home_name=match.home_team.name if match.home_team else None,
                away_name=match.away_team.name if match.away_team else None,
                match_date=match.date,
            ))

            new_picks.append((match, {
                'outcome': side,
                'odds': odds_val,
                'ev': ev,
                'p_is': p_is,
                'niche': niche_str,
                'stake': stake,
            }))

        db.commit()

        if new_picks:
            logger.info(f'[Aquamarine] Generated {len(new_picks)} picks')
            _send_picks_to_telegram(new_picks, db, bankroll)
        else:
            logger.info('[Aquamarine] No new picks')

    finally:
        db.close()


def _send_picks_to_telegram(picks: list, db, bankroll: float) -> None:
    import asyncio
    from aiogram import Bot
    from db.models import User
    from db.session import SessionLocal

    last10 = _last10_stats(db, MODEL_VERSION)

    async def _send():
        if not settings.telegram_bot_token:
            return
        bot = Bot(token=settings.telegram_bot_token)
        db2 = SessionLocal()
        try:
            users = db2.query(User).filter_by(is_active=True).all()
            if not users:
                return

            lines = []
            for match, pick in picks:
                home = match.home_team.name if match.home_team else '?'
                away = match.away_team.name if match.away_team else '?'
                time_str = match.date.strftime('%d.%m %H:%M')
                side = pick['outcome']
                side_label = f"П1 ({home})" if side == 'home' else f"П2 ({away})"
                high_risk = pick['odds'] >= HIGH_RISK_ODDS

                hr_prefix = "⚠️ <b>HIGH RISK</b>\n" if high_risk else ""
                line = (
                    f"{hr_prefix}"
                    f"<b>{home} — {away}</b> [{time_str}]\n"
                    f"  ➤ {side_label}\n"
                    f"  Коеф: {pick['odds']:.2f} | EV: {pick['ev']*100:.1f}% | "
                    f"p_win: {pick['p_is']:.1%}\n"
                    f"  Стейк: ${pick['stake']:.0f} (банкрол: ${bankroll:.0f})"
                )
                lines.append(line)

            n = len(picks)
            picks_word = 'пика' if n in (2, 3, 4) else ('пик' if n == 1 else 'пиков')

            l10_str = ''
            if last10['n'] > 0:
                l10_str = (
                    f"\n\n<i>Last {last10['n']}: {last10['wins']}W "
                    f"{last10['n']-last10['wins']}L | "
                    f"ROI: {last10['roi']:+.1f}%</i>"
                )

            header = f"🐬 <b>Aquamarine — {n} {picks_word}</b>{l10_str}\n\n"
            message = header + '\n\n'.join(lines)

            sent = 0
            for user in users:
                try:
                    await bot.send_message(user.telegram_id, message, parse_mode='HTML')
                    sent += 1
                except Exception as e:
                    logger.warning(f'Failed to send to {user.telegram_id}: {e}')
            logger.info(f'[Aquamarine] Picks sent to {sent}/{len(users)} users')
        finally:
            db2.close()
            await bot.session.close()

    asyncio.run(_send())
