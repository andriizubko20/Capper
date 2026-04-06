from datetime import date, timedelta

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from config.settings import settings
from db.models import Match, Prediction, Odds as OddsModel
from db.session import SessionLocal

WS_GAP_VERSION    = "ws_gap_v1"
SCHEDULE_DAYS     = 5
HISTORY_PAGE_SIZE = 7
KELLY_CAP         = 0.04

router = Router()

FINISHED_STATUSES = {"Finished", "FT", "finished", "ft", "Match Finished"}


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _ws_gap_preds(db, from_date=None, to_date=None):
    """Повертає WS Gap пики від найстарішого до найновішого."""
    q = (
        db.query(Prediction)
        .join(Match)
        .filter(Prediction.model_version == WS_GAP_VERSION)
    )
    if from_date:
        q = q.filter(Match.date >= str(from_date))
    if to_date:
        q = q.filter(Match.date < str(to_date))
    return q.order_by(Match.date.asc()).all()


def _compound_ladder(preds: list, initial: float) -> list[dict]:
    """
    Симулює compound зростання банкролу.
    Для кожної ставки stake = min(bankroll * kelly_fraction, bankroll * KELLY_CAP).
    Повертає список dict з {pred, res, stake, profit, bankroll}.
    """
    bankroll = initial
    ladder = []
    for pred in preds:
        res = _result(pred.match, pred)
        # Перераховуємо стейк з поточного банкролу
        kf = pred.kelly_fraction or 0
        stake = round(min(bankroll * kf, bankroll * KELLY_CAP), 2) if bankroll > 0 else 0.0

        if res == "win":
            profit = round(stake * (pred.odds_used - 1), 2)
        elif res == "loss":
            profit = -stake
        else:
            profit = 0.0  # pending — банкрол не змінюємо

        if res != "pending":
            bankroll = round(bankroll + profit, 2)

        ladder.append({
            "pred":     pred,
            "res":      res,
            "stake":    stake,
            "profit":   profit,
            "bankroll": bankroll,
        })
    return ladder


# ── /start ────────────────────────────────────────────────────────────────────

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 <b>Capper — WS Gap</b>\n\n"
        "Аналізую топ-5 ліг Європи. Ставлю тільки там, де один клуб домінує по всіх "
        "факторах: Elo, xG, форма, таблиця — і ринок дає хороший коефіцієнт.\n\n"
        "📌 <b>Команди:</b>\n"
        "/picks — активні ставки\n"
        "/history — минулі ставки\n"
        "/stats — статистика\n"
        "/schedule — розклад матчів",
        parse_mode="HTML",
    )


# ── /picks ────────────────────────────────────────────────────────────────────

@router.message(Command("picks"))
async def cmd_picks(message: Message):
    db = SessionLocal()
    try:
        today = date.today()
        all_preds = _ws_gap_preds(db)
        ladder    = _compound_ladder(all_preds, settings.bankroll)

        # Тільки активні (найближчі 5 днів)
        active = [
            item for item in ladder
            if item["pred"].match.date.date() >= today
            and item["pred"].match.date.date() < today + timedelta(days=5)
        ]

        if not active:
            await message.answer(
                "🤷 Активних ставок немає.\n\n"
                "<i>Модель шукає матчі з WS Gap ≥ 80 щогодини.</i>",
                parse_mode="HTML",
            )
            return

        by_day: dict[str, list] = {}
        for item in active:
            day = item["pred"].match.date.strftime("%d.%m")
            by_day.setdefault(day, []).append(item)

        lines = [f"📐 <b>WS Gap — {len(active)} ставок</b>\n"]

        for day in sorted(by_day):
            lines.append(f"📅 <b>{day}</b>")
            for item in by_day[day]:
                pred = item["pred"]
                m    = pred.match
                home = m.home_team.name if m.home_team else "?"
                away = m.away_team.name if m.away_team else "?"
                time_str  = m.date.strftime("%H:%M")
                bet_team  = _team_label(m, pred.outcome)
                side      = "П1" if pred.outcome == "home" else "П2"

                if item["res"] == "win":
                    status = f"✅ +${item['profit']:.0f}"
                elif item["res"] == "loss":
                    status = f"❌ −${abs(item['profit']):.0f}"
                else:
                    status = f"⏳ ${item['stake']:.0f}"

                lines.append(
                    f"\n⚽ <b>{home} — {away}</b> · {time_str}\n"
                    f"   ➤ {side} <b>{bet_team}</b>\n"
                    f"   Коеф: <b>{pred.odds_used:.2f}</b> · EV: {pred.ev*100:.0f}%\n"
                    f"   {status}"
                )
            lines.append("")

        await message.answer("\n".join(lines), parse_mode="HTML")
    finally:
        db.close()


# ── /history ──────────────────────────────────────────────────────────────────

def _build_history_text(page: int) -> str:
    db = SessionLocal()
    try:
        # Всі пики від найстаріших до найновіших
        all_preds = _ws_gap_preds(db)
        if not all_preds:
            return "Ставок ще немає."

        ladder = _compound_ladder(all_preds, settings.bankroll)

        # Тільки ті що вже відбулися або сьогодні — без майбутніх
        today = date.today()
        ladder = [i for i in ladder if i["pred"].match.date.date() <= today]

        # Групуємо по даті (від найстаріших — вже відсортовано asc)
        by_day: dict[str, list] = {}
        for item in ladder:
            day = item["pred"].match.date.strftime("%d.%m.%Y")
            by_day.setdefault(day, []).append(item)

        days = list(by_day.keys())
        total_pages = max(1, (len(days) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        page_days = days[page * HISTORY_PAGE_SIZE: (page + 1) * HISTORY_PAGE_SIZE]

        lines = [f"📋 <b>Історія WS Gap</b> · стор. {page + 1}/{total_pages}\n"]
        page_profit = 0.0

        for day in page_days:
            items = by_day[day]
            lines.append(f"📅 <b>{day}</b>")
            day_p = 0.0

            for item in items:
                pred = item["pred"]
                m    = pred.match
                home = m.home_team.name if m.home_team else "?"
                away = m.away_team.name if m.away_team else "?"
                bet_team = _team_label(m, pred.outcome)
                side = "П1" if pred.outcome == "home" else "П2"
                score_str = f" {m.home_score}:{m.away_score}" if m.home_score is not None else ""

                if item["res"] == "win":
                    icon, profit_str = "✅", f"+${item['profit']:.0f}"
                    day_p += item["profit"]
                elif item["res"] == "loss":
                    icon, profit_str = "❌", f"−${abs(item['profit']):.0f}"
                    day_p += item["profit"]
                else:
                    icon, profit_str = "⏳", "очікується"

                lines.append(
                    f"  {icon} {home} — {away}{score_str}\n"
                    f"     {side} {bet_team} · {pred.odds_used:.2f}\n"
                    f"     ${item['stake']:.0f} → <b>{profit_str}</b>"
                )

            # Підсумок дня + поточний банкрол
            end_bankroll = items[-1]["bankroll"]
            sign = "+" if day_p >= 0 else ""
            lines.append(f"  <i>День: {sign}${day_p:.0f} · Банкрол: ${end_bankroll:.0f}</i>\n")
            page_profit += day_p

        sign = "+" if page_profit >= 0 else ""
        lines.append("━━━━━━━━━━━━━━━")
        lines.append(f"Сторінка: <b>{sign}${page_profit:.0f}</b>")

        return "\n".join(lines)
    finally:
        db.close()


def _history_keyboard(page: int) -> InlineKeyboardMarkup:
    db = SessionLocal()
    try:
        preds = _ws_gap_preds(db, to_date=date.today() + timedelta(days=1))
        by_day = {}
        for p in preds:
            by_day[p.match.date.strftime("%d.%m.%Y")] = 1
        total_pages = max(1, (len(by_day) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
    finally:
        db.close()

    row = []
    if page > 0:
        row.append(InlineKeyboardButton(text="◀ Новіші", callback_data=f"history:{page - 1}"))
    if page < total_pages - 1:
        row.append(InlineKeyboardButton(text="Старіші ▶", callback_data=f"history:{page + 1}"))
    return InlineKeyboardMarkup(inline_keyboard=[row]) if row else InlineKeyboardMarkup(inline_keyboard=[])


@router.message(Command("history"))
async def cmd_history(message: Message):
    text = _build_history_text(0)
    await message.answer(text, reply_markup=_history_keyboard(0), parse_mode="HTML")


@router.callback_query(F.data.startswith("history:"))
async def cb_history(callback: CallbackQuery):
    page = int(callback.data.split(":")[1])
    text = _build_history_text(page)
    await callback.message.edit_text(text, reply_markup=_history_keyboard(page), parse_mode="HTML")
    await callback.answer()


# ── /stats ────────────────────────────────────────────────────────────────────

@router.message(Command("stats"))
async def cmd_stats(message: Message):
    db = SessionLocal()
    try:
        initial   = settings.bankroll
        all_preds = _ws_gap_preds(db)

        if not all_preds:
            await message.answer("Статистика ще недоступна — ставок немає.")
            return

        ladder    = _compound_ladder(all_preds, initial)
        finished  = [i for i in ladder if i["res"] != "pending"]
        pending_n = len(ladder) - len(finished)

        wins   = sum(1 for i in finished if i["res"] == "win")
        losses = sum(1 for i in finished if i["res"] == "loss")

        # Поточний банкрол — останній запис в ladder
        balance  = ladder[-1]["bankroll"]
        profit   = balance - initial
        roi      = (profit / initial) * 100

        win_rate = f"{wins}/{len(finished)} ({wins/len(finished)*100:.0f}%)" if finished else "—"
        avg_odds = sum(i["pred"].odds_used for i in ladder) / len(ladder)

        # Серія виграшів/програшів поспіль
        streak_wins = streak_losses = 0
        for item in reversed(finished):
            if item["res"] == "win":
                if streak_losses > 0:
                    break
                streak_wins += 1
            else:
                if streak_wins > 0:
                    break
                streak_losses += 1

        if streak_wins >= 2:
            streak_str = f"🔥 {streak_wins} виграші поспіль"
        elif streak_wins == 1:
            streak_str = "🔥 1 виграш"
        elif streak_losses >= 2:
            streak_str = f"🔴 {streak_losses} програші поспіль"
        else:
            streak_str = "—"

        since_str = all_preds[0].match.date.strftime("%d.%m.%Y")
        profit_sign = "+" if profit >= 0 else ""
        roi_sign    = "+" if roi >= 0 else ""

        await message.answer(
            f"📊 <b>WS Gap · {since_str} → сьогодні</b>\n\n"
            f"💰 <b>${initial:.0f} → ${balance:.2f}</b>\n"
            f"   {profit_sign}${profit:.2f}  ({roi_sign}{roi:.1f}% ROI)\n\n"
            f"🏆 Win rate: <b>{win_rate}</b>\n"
            f"   Очікується: {pending_n} ставок\n\n"
            f"📈 Avg коеф: {avg_odds:.2f}\n"
            f"   {streak_str}\n\n"
            f"<i>Фільтр: WS Gap ≥ 80 · Odds ≥ 1.7 · Kelly 4%</i>",
            parse_mode="HTML",
        )
    finally:
        db.close()


# ── /schedule ─────────────────────────────────────────────────────────────────

def _schedule_keyboard(day_offset: int) -> InlineKeyboardMarkup:
    row = []
    if day_offset > 0:
        row.append(InlineKeyboardButton(text="◀", callback_data=f"schedule:{day_offset - 1}"))
    row.append(InlineKeyboardButton(
        text="Сьогодні" if day_offset == 0 else ("Завтра" if day_offset == 1 else f"+{day_offset}д"),
        callback_data=f"schedule:{day_offset}",
    ))
    if day_offset < SCHEDULE_DAYS - 1:
        row.append(InlineKeyboardButton(text="▶", callback_data=f"schedule:{day_offset + 1}"))
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _build_schedule_text(day_offset: int) -> str:
    target = date.today() + timedelta(days=day_offset)
    db = SessionLocal()
    try:
        matches = (
            db.query(Match)
            .filter(
                Match.date >= str(target),
                Match.date < str(target + timedelta(days=1)),
                Match.status == "Not Started",
            )
            .order_by(Match.date)
            .all()
        )

        day_label = "Сьогодні" if day_offset == 0 else ("Завтра" if day_offset == 1 else target.strftime("%d.%m"))

        if not matches:
            return f"📅 <b>{day_label}</b> — матчів немає"

        picks_match_ids = {
            p.match_id
            for p in db.query(Prediction)
            .filter(
                Prediction.model_version == WS_GAP_VERSION,
                Prediction.match_id.in_([m.id for m in matches]),
            )
            .all()
        }

        by_league: dict[str, list] = {}
        for m in matches:
            league_name = m.league.name if m.league else "Інше"
            by_league.setdefault(league_name, []).append(m)

        lines = [f"📅 <b>{day_label}</b> — {len(matches)} матчів\n"]

        for league in sorted(by_league):
            lines.append(f"🏆 <b>{league}</b>")
            for m in by_league[league]:
                home = m.home_team.name if m.home_team else "?"
                away = m.away_team.name if m.away_team else "?"
                time_str = m.date.strftime("%H:%M")

                odds_rows = db.query(OddsModel).filter_by(match_id=m.id, market="1x2", is_closing=False).all()
                odds_map  = {o.outcome: o.value for o in odds_rows}

                if odds_map.get("home") and odds_map.get("away"):
                    odds_str = f"{odds_map['home']:.2f} · {odds_map.get('draw', 0):.2f} · {odds_map['away']:.2f}"
                else:
                    odds_str = "—"

                pick_marker = " 🎯" if m.id in picks_match_ids else ""
                lines.append(f"  {time_str}  {home} — {away}{pick_marker}\n  <i>{odds_str}</i>")

            lines.append("")

        return "\n".join(lines)
    finally:
        db.close()


@router.message(Command("schedule"))
async def cmd_schedule(message: Message):
    text = _build_schedule_text(0)
    await message.answer(text, reply_markup=_schedule_keyboard(0), parse_mode="HTML")


@router.callback_query(F.data.startswith("schedule:"))
async def cb_schedule(callback: CallbackQuery):
    day_offset = int(callback.data.split(":")[1])
    text = _build_schedule_text(day_offset)
    await callback.message.edit_text(text, reply_markup=_schedule_keyboard(day_offset), parse_mode="HTML")
    await callback.answer()
