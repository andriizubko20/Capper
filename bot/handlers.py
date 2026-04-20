"""
bot/handlers.py — Capper bot UI.

Головне меню: WS Gap | Monster
Кожна модель: Пики / Статистика / История / переключитися на іншу.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from bot.locales import t, picks_word
from config.settings import settings
from db.models import Match, Prediction
from db.session import SessionLocal

# ── Model versions ─────────────────────────────────────────────────────────────
WS_GAP_CAP        = "ws_gap_v1"
WS_GAP_KELLY      = "ws_gap_kelly_v1"
MONSTER_CAP       = "monster_v1"
MONSTER_KELLY     = "monster_v1_kelly"
AQUAMARINE_CAP    = "aquamarine_v1"
AQUAMARINE_KELLY  = "aquamarine_v1_kelly"

KELLY_CAP     = 0.04
SCHEDULE_DAYS = 7
REFRESH_DAYS  = 7  # window for manual refresh

_refresh_executor = ThreadPoolExecutor(max_workers=1)

router = Router()

FINISHED_STATUSES = {"Finished", "FT", "finished", "ft", "Match Finished"}

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
    "Ukraine":     "🇺🇦",
}

HIGH_RISK_ODDS = 2.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flag(match: Match) -> str:
    if match.league and match.league.country:
        return LEAGUE_FLAGS.get(match.league.country, "🌍")
    return "🌍"


def _result(match: Match, pred: Prediction) -> str:
    if match.status not in FINISHED_STATUSES or match.home_score is None:
        return "pending"
    hs, as_ = match.home_score, match.away_score
    if pred.outcome == "home":
        return "win" if hs > as_ else "loss"
    return "win" if as_ > hs else "loss"


def _team_label(match: Match, outcome: str) -> str:
    home = match.home_team.name if match.home_team else "?"
    away = match.away_team.name if match.away_team else "?"
    return home if outcome == "home" else away


def _preds(db, version: str, from_date=None, to_date=None):
    q = db.query(Prediction).join(Match).filter(Prediction.model_version == version)
    if from_date:
        q = q.filter(Match.date >= str(from_date))
    if to_date:
        q = q.filter(Match.date < str(to_date))
    return q.order_by(Match.date.asc()).all()


def _compound_ladder(preds: list, initial: float, cap: float | None = KELLY_CAP) -> list[dict]:
    bankroll = initial
    ladder = []
    for pred in preds:
        res = _result(pred.match, pred)
        kf = pred.kelly_fraction or 0
        stake = round(min(bankroll * kf, bankroll * cap), 2) if cap else round(bankroll * kf, 2)
        stake = max(stake, 0.0) if bankroll > 0 else 0.0

        profit = round(stake * (pred.odds_used - 1), 2) if res == "win" else (-stake if res == "loss" else 0.0)
        if res != "pending":
            bankroll = round(bankroll + profit, 2)

        ladder.append({"pred": pred, "res": res, "stake": stake, "profit": profit, "bankroll": bankroll})
    return ladder


def _last10(db, version: str) -> dict:
    FINISHED = {"Finished", "FT", "finished", "ft", "Match Finished"}
    preds = (
        db.query(Prediction).join(Match)
        .filter(Prediction.model_version == version, Match.status.in_(FINISHED), Match.home_score.isnot(None))
        .order_by(Match.date.desc()).limit(10).all()
    )
    if not preds:
        return {"n": 0, "wins": 0, "roi": 0.0}
    wins = 0
    flat = 0.0
    for p in preds:
        m = p.match
        won = (p.outcome == "home" and m.home_score > m.away_score) or \
              (p.outcome == "away" and m.away_score > m.home_score)
        wins += int(won)
        flat += (p.odds_used - 1) if won else -1.0
    n = len(preds)
    return {"n": n, "wins": wins, "roi": flat / n * 100}


# ── Keyboards ──────────────────────────────────────────────────────────────────

def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📐 WS Gap",       callback_data="wsgap:menu")],
        [InlineKeyboardButton(text="⚡ Monster",      callback_data="monster:menu")],
        [InlineKeyboardButton(text="🐬 Aquamarine",   callback_data="aquamarine:menu")],
    ])


def _wsgap_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=t("btn_picks_wsgap"),  callback_data="wsgap:picks"),
         InlineKeyboardButton(text=t("btn_stats"),        callback_data="wsgap:stats")],
        [InlineKeyboardButton(text=t("btn_history"),      callback_data="wsgap:history"),
         InlineKeyboardButton(text=t("btn_schedule"),     callback_data="wsgap:schedule")],
        [InlineKeyboardButton(text=t("btn_refresh"),      callback_data="wsgap:refresh")],
        [InlineKeyboardButton(text=t("btn_to_monster"),   callback_data="monster:menu"),
         InlineKeyboardButton(text=t("btn_menu"),         callback_data="menu")],
    ])


def _monster_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=t("btn_picks_monster"), callback_data="monster:picks"),
         InlineKeyboardButton(text=t("btn_stats"),         callback_data="monster:stats:kelly")],
        [InlineKeyboardButton(text=t("btn_history"),       callback_data="monster:history"),
         InlineKeyboardButton(text=t("btn_schedule"),      callback_data="monster:schedule")],
        [InlineKeyboardButton(text=t("btn_refresh"),       callback_data="monster:refresh")],
        [InlineKeyboardButton(text=t("btn_to_wsgap"),      callback_data="wsgap:menu"),
         InlineKeyboardButton(text=t("btn_menu"),          callback_data="menu")],
    ])


def _aquamarine_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=t("btn_picks_aquamarine"), callback_data="aquamarine:picks"),
         InlineKeyboardButton(text=t("btn_stats"),            callback_data="aquamarine:stats:kelly")],
        [InlineKeyboardButton(text=t("btn_history"),          callback_data="aquamarine:history"),
         InlineKeyboardButton(text=t("btn_schedule"),         callback_data="aquamarine:schedule")],
        [InlineKeyboardButton(text=t("btn_refresh"),          callback_data="aquamarine:refresh")],
        [InlineKeyboardButton(text=t("btn_to_monster"),       callback_data="monster:menu"),
         InlineKeyboardButton(text=t("btn_menu"),             callback_data="menu")],
    ])


def _aquamarine_stats_keyboard(current: str) -> InlineKeyboardMarkup:
    if current == "kelly":
        switch_btn = InlineKeyboardButton(text=t("btn_stats_cap"),   callback_data="aquamarine:stats:cap")
    else:
        switch_btn = InlineKeyboardButton(text=t("btn_stats_kelly"), callback_data="aquamarine:stats:kelly")
    return InlineKeyboardMarkup(inline_keyboard=[
        [switch_btn],
        [InlineKeyboardButton(text=t("btn_back_aquamarine"), callback_data="aquamarine:menu")],
    ])


def _back_keyboard(model: str) -> InlineKeyboardMarkup:
    if model == "wsgap":
        cb, label = "wsgap:menu", t("btn_back_wsgap")
    elif model == "aquamarine":
        cb, label = "aquamarine:menu", t("btn_back_aquamarine")
    else:
        cb, label = "monster:menu", t("btn_back_monster")
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=cb)],
    ])


def _history_keyboard(model: str, day_str: str, days: list) -> InlineKeyboardMarkup:
    try:
        idx = days.index(day_str)
    except ValueError:
        idx = len(days) - 1
    row = []
    if idx > 0:
        row.append(InlineKeyboardButton(text="◀", callback_data=f"{model}:history_day:{days[idx-1]}"))
    row.append(InlineKeyboardButton(text=f"{idx+1}/{len(days)}", callback_data=f"{model}:history_day:{day_str}"))
    if idx < len(days) - 1:
        row.append(InlineKeyboardButton(text="▶", callback_data=f"{model}:history_day:{days[idx+1]}"))
    if model == "wsgap":
        back_cb, back_label = "wsgap:menu", t("btn_back_wsgap")
    elif model == "aquamarine":
        back_cb, back_label = "aquamarine:menu", t("btn_back_aquamarine")
    else:
        back_cb, back_label = "monster:menu", t("btn_back_monster")
    return InlineKeyboardMarkup(inline_keyboard=[
        row,
        [InlineKeyboardButton(text=back_label, callback_data=back_cb)],
    ])


def _monster_stats_keyboard(current: str) -> InlineKeyboardMarkup:
    """current: 'kelly' or 'cap'"""
    if current == "kelly":
        switch_btn = InlineKeyboardButton(text=t("btn_stats_cap"),   callback_data="monster:stats:cap")
    else:
        switch_btn = InlineKeyboardButton(text=t("btn_stats_kelly"), callback_data="monster:stats:kelly")
    return InlineKeyboardMarkup(inline_keyboard=[
        [switch_btn],
        [InlineKeyboardButton(text=t("btn_back_monster"), callback_data="monster:menu")],
    ])



def _picks_text(db, version_kelly: str, version_cap: str, model: str) -> str:
    today = date.today()
    all_preds = _preds(db, version=version_cap)
    ladder = _compound_ladder(all_preds, settings.bankroll, cap=KELLY_CAP)

    active = [
        item for item in ladder
        if item["pred"].match.date.date() >= today
        and item["pred"].match.date.date() < today + timedelta(days=SCHEDULE_DAYS)
    ]

    if model == "wsgap":
        model_label = "📐 WS Gap"
        no_active_key = "picks_no_active_wsgap"
    elif model == "aquamarine":
        model_label = "🐬 Aquamarine"
        no_active_key = "picks_no_active_aquamarine"
    else:
        model_label = "⚡ Monster"
        no_active_key = "picks_no_active_monster"
    if not active:
        return t(no_active_key)

    by_day: dict[str, list] = {}
    for item in active:
        day = item["pred"].match.date.strftime("%d.%m")
        by_day.setdefault(day, []).append(item)

    n = len(active)
    lines = [t("picks_header", model_label=model_label, n=n, word=picks_word(n))]
    for day in sorted(by_day):
        lines.append(f"📅 <b>{day}</b>")
        for item in by_day[day]:
            pred = item["pred"]
            m = pred.match
            home = m.home_team.name if m.home_team else "?"
            away = m.away_team.name if m.away_team else "?"
            time_str = m.date.strftime("%H:%M")
            bet_team = _team_label(m, pred.outcome)
            side = t("side_home") if pred.outcome == "home" else t("side_away")
            if item["res"] == "win":
                status = t("status_win", profit=item["profit"])
            elif item["res"] == "loss":
                status = t("status_loss", loss=abs(item["profit"]))
            else:
                status = t("status_pending", stake=item["stake"])

            lines.append(
                f"\n{_flag(m)} <b>{home} — {away}</b> · {time_str}\n"
                f"   ➤ {side} <b>{bet_team}</b>\n"
                f"   Коеф: <b>{pred.odds_used:.2f}</b> · EV: {pred.ev*100:.0f}%\n"
                f"   {status}"
            )
        lines.append("")

    return "\n".join(lines)


def _stats_text(db, version: str, cap: float | None, title: str) -> str:
    initial = settings.bankroll
    all_preds = _preds(db, version=version)
    if not all_preds:
        return t("stats_no_data", title=title)

    ladder = _compound_ladder(all_preds, initial, cap=cap)
    finished = [i for i in ladder if i["res"] != "pending"]
    pending  = [i for i in ladder if i["res"] == "pending"]

    wins   = sum(1 for i in finished if i["res"] == "win")
    losses = sum(1 for i in finished if i["res"] == "loss")

    balance  = ladder[-1]["bankroll"]
    profit   = balance - initial
    roi      = (profit / initial) * 100
    win_pct  = wins / len(finished) * 100 if finished else 0
    avg_odds = sum(i["pred"].odds_used for i in ladder) / len(ladder)

    streak_wins = streak_losses = 0
    for item in reversed(finished):
        if item["res"] == "win":
            if streak_losses > 0: break
            streak_wins += 1
        else:
            if streak_wins > 0: break
            streak_losses += 1

    if streak_wins >= 2:
        streak_str = t("streak_wins_n", n=streak_wins)
    elif streak_wins == 1:
        streak_str = t("streak_wins_1")
    elif streak_losses >= 2:
        streak_str = t("streak_losses_n", n=streak_losses)
    else:
        streak_str = t("streak_none")

    since_str = all_preds[0].match.date.strftime("%d.%m")
    to_str    = date.today().strftime("%d.%m.%Y")

    bankrolls = [i["bankroll"] for i in ladder if i["res"] != "pending"]
    peak      = max(bankrolls) if bankrolls else balance
    drawdown  = (peak - balance) / peak * 100 if peak > 0 else 0

    profit_sign = "+" if profit >= 0 else ""
    roi_sign    = "+" if roi >= 0 else ""
    filled      = min(int(win_pct / 10), 10)
    bars        = ("🟩" if win_pct >= 60 else "🟥") * filled + "⬜" * (10 - filled)

    text = (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Capper · {title}</b>\n"
        f"📅 {since_str} → {to_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 <b>${initial:.0f} → ${balance:.0f}</b>\n"
        f"   {profit_sign}${profit:.0f}  ·  ROI {roi_sign}{roi:.1f}%\n\n"
        f"🎯 <b>{wins}W / {losses}L</b>  ({win_pct:.0f}%)\n"
        f"   {bars}\n"
    )
    if pending:
        text += t("stats_pending", n=len(pending))
    text += t("stats_avg_odds", avg=avg_odds, streak=streak_str)
    if drawdown > 5:
        text += t("stats_drawdown", dd=drawdown)
    text += "\n━━━━━━━━━━━━━━━━━━━━"
    return text


def _history_days(db, version: str) -> list[str]:
    preds = _preds(db, version=version, to_date=date.today() + timedelta(days=1))
    seen: dict[str, bool] = {}
    for p in preds:
        seen[p.match.date.strftime("%Y-%m-%d")] = True
    return list(seen.keys())


def _history_day_text(db, day_str: str, version: str) -> str:
    all_preds = _preds(db, version=version)
    if not all_preds:
        return t("history_no_bets_day")

    ladder = _compound_ladder(all_preds, settings.bankroll, cap=KELLY_CAP)
    items  = [i for i in ladder if i["pred"].match.date.strftime("%Y-%m-%d") == day_str]
    if not items:
        return t("history_no_bets_on_day", day=day_str)

    day_label = items[0]["pred"].match.date.strftime("%d.%m.%Y")
    lines = [f"📋 <b>{day_label}</b>\n"]
    day_p = 0.0

    for item in items:
        pred = item["pred"]
        m = pred.match
        home = m.home_team.name if m.home_team else "?"
        away = m.away_team.name if m.away_team else "?"
        bet_team = _team_label(m, pred.outcome)
        side = t("side_home") if pred.outcome == "home" else t("side_away")
        score_str = f" {m.home_score}:{m.away_score}" if m.home_score is not None else ""
        if item["res"] == "win":
            icon, profit_str = "✅", f"+${item['profit']:.0f}"
            day_p += item["profit"]
        elif item["res"] == "loss":
            icon, profit_str = "❌", f"−${abs(item['profit']):.0f}"
            day_p += item["profit"]
        else:
            icon, profit_str = "⏳", t("history_result_pending")

        lines.append(
            f"{icon} {_flag(pred.match)} {home} — {away}{score_str}\n"
            f"   {side} {bet_team} · {pred.odds_used:.2f}\n"
            f"   ${item['stake']:.0f} → <b>{profit_str}</b>"
        )

    end_bankroll = items[-1]["bankroll"]
    sign = "+" if day_p >= 0 else ""
    lines.append(t("history_day_summary", sign=sign, pnl=day_p, bankroll=end_bankroll))
    return "\n".join(lines)


def _run_refresh_wsgap(days: int = REFRESH_DAYS) -> int:
    """Run WS Gap pick generation for next `days` days. Returns count of new picks."""
    from scheduler.tasks.generate_picks_ws_gap import run_generate_picks_ws_gap
    now = datetime.now(timezone.utc)
    before = _count_predictions(WS_GAP_CAP)
    run_generate_picks_ws_gap(
        match_date_from=now,
        match_date_to=now + timedelta(days=days),
    )
    return _count_predictions(WS_GAP_CAP) - before


def _run_refresh_monster(days: int = REFRESH_DAYS) -> int:
    """Run Monster pick generation for next `days` days. Returns count of new picks."""
    from scheduler.tasks.generate_picks_monster import run_generate_picks_monster
    now = datetime.now(timezone.utc)
    before = _count_predictions(MONSTER_CAP)
    run_generate_picks_monster(
        match_date_from=now,
        match_date_to=now + timedelta(days=days),
    )
    return _count_predictions(MONSTER_CAP) - before


def _run_refresh_aquamarine(days: int = REFRESH_DAYS) -> int:
    """Run Aquamarine pick generation for next `days` days. Returns count of new picks."""
    from scheduler.tasks.generate_picks_aquamarine import run_generate_picks_aquamarine
    now = datetime.now(timezone.utc)
    before = _count_predictions(AQUAMARINE_CAP)
    run_generate_picks_aquamarine(
        match_date_from=now,
        match_date_to=now + timedelta(days=days),
    )
    return _count_predictions(AQUAMARINE_CAP) - before


def _count_predictions(version: str) -> int:
    db = SessionLocal()
    try:
        return db.query(Prediction).filter(Prediction.model_version == version).count()
    finally:
        db.close()


def _schedule_text(db, version_cap: str, model: str) -> str:
    """Show upcoming picks for next SCHEDULE_DAYS days with status labels."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    to_date = today + timedelta(days=SCHEDULE_DAYS)

    if model == "wsgap":
        model_label = "📐 WS Gap"
        no_picks_key = "schedule_no_picks_wsgap"
    elif model == "aquamarine":
        model_label = "🐬 Aquamarine"
        no_picks_key = "schedule_no_picks_aquamarine"
    else:
        model_label = "⚡ Monster"
        no_picks_key = "schedule_no_picks_monster"

    all_preds = _preds(db, version=version_cap, from_date=today, to_date=to_date)
    if not all_preds:
        return t(no_picks_key)
    by_day: dict[str, list] = {}
    for pred in all_preds:
        day = pred.match.date.strftime("%d.%m")
        by_day.setdefault(day, []).append(pred)

    lines = [t("schedule_header", model_label=model_label)]
    for day in sorted(by_day):
        lines.append(f"\n📅 <b>{day}</b>")
        for pred in by_day[day]:
            m = pred.match
            home = m.home_team.name if m.home_team else "?"
            away = m.away_team.name if m.away_team else "?"
            time_str = m.date.strftime("%H:%M")
            bet_team = _team_label(m, pred.outcome)
            side = t("side_home") if pred.outcome == "home" else t("side_away")

            # Label by hours to kickoff
            hours_left = (m.date - now).total_seconds() / 3600
            if hours_left <= settings.picks_hours_before + 0.5:
                label = t("schedule_label_final")
            elif hours_left <= 24:
                label = t("schedule_label_soon")
            else:
                label = t("schedule_label_early")

            # High risk label for Monster/Aquamarine in schedule (odds >= threshold)
            hr_tag = f" {t('schedule_hr_tag')}" if model in ("monster", "aquamarine") and pred.odds_used >= HIGH_RISK_ODDS else ""

            lines.append(
                f"  {label}{hr_tag}\n"
                f"  {_flag(m)} <b>{home} — {away}</b> · {time_str}\n"
                f"  ➤ {side} <b>{bet_team}</b> · {pred.odds_used:.2f}"
            )

    return "\n".join(lines)


# ── Handlers ───────────────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(t("main_menu"), reply_markup=_main_menu_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "menu")
async def cb_menu(callback: CallbackQuery):
    await callback.message.edit_text(t("main_menu"), reply_markup=_main_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


# ── WS Gap ─────────────────────────────────────────────────────────────────────

@router.callback_query(F.data == "wsgap:menu")
async def cb_wsgap_menu(callback: CallbackQuery):
    await callback.message.edit_text(t("wsgap_menu"), reply_markup=_wsgap_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "wsgap:picks")
async def cb_wsgap_picks(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _picks_text(db, WS_GAP_KELLY, WS_GAP_CAP, "wsgap")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("wsgap"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "wsgap:stats")
async def cb_wsgap_stats(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _stats_text(db, WS_GAP_KELLY, cap=None, title=t("stats_title_wsgap_kelly"))
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("wsgap"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "wsgap:history")
async def cb_wsgap_history(callback: CallbackQuery):
    db = SessionLocal()
    try:
        days = _history_days(db, WS_GAP_CAP)
        if not days:
            await callback.message.edit_text(
                t("history_no_bets_wsgap"),
                reply_markup=_back_keyboard("wsgap"), parse_mode="HTML"
            )
            await callback.answer()
            return
        day_str = days[-1]
        text = _history_day_text(db, day_str, WS_GAP_CAP)
        kb = _history_keyboard("wsgap", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "wsgap:refresh")
async def cb_wsgap_refresh(callback: CallbackQuery):
    await callback.answer(t("refresh_started"), show_alert=False)
    await callback.message.edit_text(t("refresh_loading_wsgap"), parse_mode="HTML")
    loop = asyncio.get_event_loop()
    try:
        new_n = await loop.run_in_executor(_refresh_executor, _run_refresh_wsgap)
        text = t("refresh_done_wsgap", n=new_n)
    except Exception as e:
        text = t("refresh_error", err=str(e)[:200])
    await callback.message.edit_text(text, reply_markup=_wsgap_menu_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "wsgap:schedule")
async def cb_wsgap_schedule(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _schedule_text(db, WS_GAP_CAP, "wsgap")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("wsgap"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("wsgap:history_day:"))
async def cb_wsgap_history_day(callback: CallbackQuery):
    day_str = callback.data.split(":", 2)[2]
    db = SessionLocal()
    try:
        days = _history_days(db, WS_GAP_CAP)
        text = _history_day_text(db, day_str, WS_GAP_CAP)
        kb   = _history_keyboard("wsgap", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


# ── Monster ────────────────────────────────────────────────────────────────────

@router.callback_query(F.data == "monster:menu")
async def cb_monster_menu(callback: CallbackQuery):
    await callback.message.edit_text(t("monster_menu"), reply_markup=_monster_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "monster:picks")
async def cb_monster_picks(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _picks_text(db, MONSTER_KELLY, MONSTER_CAP, "monster")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("monster"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("monster:stats:"))
async def cb_monster_stats(callback: CallbackQuery):
    mode = callback.data.split(":")[-1]  # "kelly" or "cap"
    db = SessionLocal()
    try:
        if mode == "kelly":
            text = _stats_text(db, MONSTER_KELLY, cap=None, title=t("stats_title_monster_kelly"))
        else:
            text = _stats_text(db, MONSTER_CAP, cap=KELLY_CAP, title=t("stats_title_monster_cap"))
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_monster_stats_keyboard(mode), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "monster:history")
async def cb_monster_history(callback: CallbackQuery):
    db = SessionLocal()
    try:
        days = _history_days(db, MONSTER_CAP)
        if not days:
            await callback.message.edit_text(
                t("history_no_bets_monster"),
                reply_markup=_back_keyboard("monster"), parse_mode="HTML"
            )
            await callback.answer()
            return
        day_str = days[-1]
        text = _history_day_text(db, day_str, MONSTER_CAP)
        kb   = _history_keyboard("monster", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "monster:refresh")
async def cb_monster_refresh(callback: CallbackQuery):
    await callback.answer(t("refresh_started"), show_alert=False)
    await callback.message.edit_text(t("refresh_loading_monster"), parse_mode="HTML")
    loop = asyncio.get_event_loop()
    try:
        new_n = await loop.run_in_executor(_refresh_executor, _run_refresh_monster)
        text = t("refresh_done_monster", n=new_n)
    except Exception as e:
        text = t("refresh_error", err=str(e)[:200])
    await callback.message.edit_text(text, reply_markup=_monster_menu_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "monster:schedule")
async def cb_monster_schedule(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _schedule_text(db, MONSTER_CAP, "monster")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("monster"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("monster:history_day:"))
async def cb_monster_history_day(callback: CallbackQuery):
    day_str = callback.data.split(":", 2)[2]
    db = SessionLocal()
    try:
        days = _history_days(db, MONSTER_CAP)
        text = _history_day_text(db, day_str, MONSTER_CAP)
        kb   = _history_keyboard("monster", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


# ── Aquamarine ─────────────────────────────────────────────────────────────────

@router.callback_query(F.data == "aquamarine:menu")
async def cb_aquamarine_menu(callback: CallbackQuery):
    await callback.message.edit_text(t("aquamarine_menu"), reply_markup=_aquamarine_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "aquamarine:picks")
async def cb_aquamarine_picks(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _picks_text(db, AQUAMARINE_KELLY, AQUAMARINE_CAP, "aquamarine")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("aquamarine"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("aquamarine:stats:"))
async def cb_aquamarine_stats(callback: CallbackQuery):
    mode = callback.data.split(":")[-1]  # "kelly" or "cap"
    db = SessionLocal()
    try:
        if mode == "kelly":
            text = _stats_text(db, AQUAMARINE_KELLY, cap=None, title=t("stats_title_aquamarine_kelly"))
        else:
            text = _stats_text(db, AQUAMARINE_CAP, cap=KELLY_CAP, title=t("stats_title_aquamarine_cap"))
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_aquamarine_stats_keyboard(mode), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "aquamarine:history")
async def cb_aquamarine_history(callback: CallbackQuery):
    db = SessionLocal()
    try:
        days = _history_days(db, AQUAMARINE_CAP)
        if not days:
            await callback.message.edit_text(
                t("history_no_bets_aquamarine"),
                reply_markup=_back_keyboard("aquamarine"), parse_mode="HTML"
            )
            await callback.answer()
            return
        day_str = days[-1]
        text = _history_day_text(db, day_str, AQUAMARINE_CAP)
        kb   = _history_keyboard("aquamarine", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "aquamarine:refresh")
async def cb_aquamarine_refresh(callback: CallbackQuery):
    await callback.answer(t("refresh_started"), show_alert=False)
    await callback.message.edit_text(t("refresh_loading_aquamarine"), parse_mode="HTML")
    loop = asyncio.get_event_loop()
    try:
        new_n = await loop.run_in_executor(_refresh_executor, _run_refresh_aquamarine)
        text = t("refresh_done_aquamarine", n=new_n)
    except Exception as e:
        text = t("refresh_error", err=str(e)[:200])
    await callback.message.edit_text(text, reply_markup=_aquamarine_menu_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "aquamarine:schedule")
async def cb_aquamarine_schedule(callback: CallbackQuery):
    db = SessionLocal()
    try:
        text = _schedule_text(db, AQUAMARINE_CAP, "aquamarine")
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=_back_keyboard("aquamarine"), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("aquamarine:history_day:"))
async def cb_aquamarine_history_day(callback: CallbackQuery):
    day_str = callback.data.split(":", 2)[2]
    db = SessionLocal()
    try:
        days = _history_days(db, AQUAMARINE_CAP)
        text = _history_day_text(db, day_str, AQUAMARINE_CAP)
        kb   = _history_keyboard("aquamarine", day_str, days)
    finally:
        db.close()
    await callback.message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()
