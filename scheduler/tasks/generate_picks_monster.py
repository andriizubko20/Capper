"""
scheduler/tasks/generate_picks_monster.py

Monster model — niche-based picks.
Кожна ніша має свій IS win rate (p_is), Kelly 25%, cap 10%.
model_version = "monster_v1_kelly"
"""
import time as _time
from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger

from config.settings import settings
from db.models import Match, Odds, Prediction, MonsterPIs
from db.session import SessionLocal
from model.monster.niches import MODELS, OOS_START, HIGH_RISK_ODDS, LEAGUE_API_IDS, parse_niche, match_niche
from model.monster.features import (
    load_historical_data, build_team_state, build_upcoming_features, compute_p_is
)

MODEL_VERSION = "monster_v1_kelly"

KELLY_FRAC  = 0.25
KELLY_CAP   = 0.10

# Leagues allowed for Monster
MONSTER_LEAGUE_NAMES = set(MODELS.keys())

# In-memory cache: key → (value, cached_at_timestamp). TTL = 24h so weekly
# update_monster_p_is re-calculation is reflected the same day.
_P_IS_CACHE: dict[tuple, tuple[float, float]] = {}
_P_IS_CACHE_TTL = 86400  # 24 hours


def _get_p_is(db, matches, stats, odds_df, league, niche_str) -> float | None:
    """Read p_is from DB first; fallback to computing from scratch if missing."""
    key = (league, niche_str)
    cached = _P_IS_CACHE.get(key)
    if cached is not None:
        value, ts = cached
        if _time.time() - ts < _P_IS_CACHE_TTL:
            return value
        del _P_IS_CACHE[key]  # expired

    # Try DB
    row = db.query(MonsterPIs).filter_by(league=league, niche_str=niche_str).first()
    if row is not None:
        _P_IS_CACHE[key] = (row.p_is, _time.time())
        return row.p_is

    # Fallback: compute on the fly with fixed OOS_START cutoff
    logger.debug(f"[Monster] p_is not in DB for {league}|{niche_str}, computing...")
    niche = parse_niche(niche_str)
    p = compute_p_is(matches, stats, odds_df, league, niche, OOS_START)
    if p is not None:
        _P_IS_CACHE[key] = (p, _time.time())
    return p


def _load_odds_batch(match_ids: list[int], db) -> dict[int, dict]:
    """Best 1x2 odds per outcome per match across ALL bookmakers.

    Bookmaker shopping: for each (match, outcome) we keep the highest price
    across all bookmakers — single batch query, grouped client-side."""
    rows = db.query(Odds).filter(
        Odds.match_id.in_(match_ids),
        Odds.market == '1x2',
        Odds.is_closing == False,
    ).all()
    result: dict[int, dict] = {}
    for o in rows:
        try:
            v = float(o.value)
        except (TypeError, ValueError):
            continue
        if v <= 1.0:
            continue
        m_dict = result.setdefault(o.match_id, {})
        cur = m_dict.get(o.outcome)
        if cur is None or v > cur:
            m_dict[o.outcome] = v
    return result


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
    """Win rate + flat ROI for last 10 settled bets."""
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


def run_generate_picks_monster(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(days=settings.early_picks_days_ahead)

    logger.info(
        f"[Monster] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    db = SessionLocal()
    try:
        from db.models import League as LeagueModel

        matches_db = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status == 'Not Started',
            LeagueModel.name.in_(MONSTER_LEAGUE_NAMES),
            LeagueModel.api_id.in_(LEAGUE_API_IDS),
        ).all()

        if not matches_db:
            logger.info('[Monster] No matches in window, skipping')
            return

        logger.info(f'[Monster] Found {len(matches_db)} matches')

        # Load historical data for features + p_is
        logger.info('[Monster] Loading historical data...')
        hist_matches, hist_stats, hist_odds = load_historical_data()
        team_state = build_team_state(hist_matches, hist_stats)

        bankroll = _compute_bankroll(db, settings.bankroll, MODEL_VERSION)
        logger.info(f'[Monster] Bankroll=${bankroll:.0f}')

        match_ids = [m.id for m in matches_db]
        odds_by_match = _load_odds_batch(match_ids, db)

        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        new_picks = []

        for match in matches_db:
            if match.id in existing:
                continue

            odds_raw = odds_by_match.get(match.id)
            if not odds_raw:
                logger.debug(f'[Monster] No odds for match {match.id}, skip')
                continue

            league_name = match.league.name if match.league else ''
            # Map to Monster league name (handle England PL vs Ukraine PL)
            monster_league = None
            for ml in MONSTER_LEAGUE_NAMES:
                if ml == league_name:
                    monster_league = ml
                    break
                # Champions League / UEFA Champions League
                if 'Champions League' in league_name and ml == 'Champions League':
                    monster_league = ml
                    break
            if not monster_league:
                continue

            features = build_upcoming_features(
                match={
                    'home_team_id': match.home_team_id,
                    'away_team_id': match.away_team_id,
                    'league_name': monster_league,
                },
                team_state=team_state,
                odds=odds_raw,
            )

            # Find best matching niche (highest p_is among matches)
            best = None
            best_p_is = None

            for niche_str in MODELS[monster_league]:
                niche = parse_niche(niche_str)
                if not match_niche(features, niche, monster_league, monster_league):
                    continue
                p_is = _get_p_is(db, hist_matches, hist_stats, hist_odds, monster_league, niche_str)
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
                logger.debug(f'[Monster] Negative Kelly for {match.id} niche={niche_str}, skip')
                continue

            kelly_frac = KELLY_FRAC * f_star
            ev = round(p_is * odds_val - 1, 4)

            home = match.home_team.name if match.home_team else '?'
            away = match.away_team.name if match.away_team else '?'
            logger.info(
                f'[Monster] Pick: {home} vs {away} → {side} | niche={niche_str} '
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
            logger.info(f'[Monster] Generated {len(new_picks)} picks')
            _send_picks_to_telegram(new_picks, db, bankroll)
        else:
            logger.info('[Monster] No new picks')

    finally:
        db.close()


def _send_picks_to_telegram(picks: list, db, bankroll: float) -> None:
    import asyncio
    from aiogram import Bot
    from db.models import User
    from db.session import SessionLocal

    last10 = _last10_stats(db, MODEL_VERSION)

    # Pre-extract all ORM-dependent fields while session is open and attached
    pick_meta = [
        (
            match.home_team.name if match.home_team else '?',
            match.away_team.name if match.away_team else '?',
            match.date.strftime('%d.%m %H:%M'),
            pick,
        )
        for match, pick in picks
    ]

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
            for home, away, time_str, pick in pick_meta:
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

            header = f"⚡ <b>Monster — {n} {picks_word}</b>{l10_str}\n\n"
            message = header + '\n\n'.join(lines)

            sent = 0
            for user in users:
                try:
                    await bot.send_message(user.telegram_id, message, parse_mode='HTML')
                    sent += 1
                except Exception as e:
                    logger.warning(f'Failed to send to {user.telegram_id}: {e}')
            logger.info(f'[Monster] Picks sent to {sent}/{len(users)} users')
        finally:
            db2.close()
            await bot.session.close()

    asyncio.run(_send())
