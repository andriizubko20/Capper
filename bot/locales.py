"""
bot/locales.py

Всі текстові рядки бота в двох мовах.
Використання:
    from bot.locales import t
    text = t("main_menu")           # Ukrainian (default)
    text = t("main_menu", lang="ru")
"""

DEFAULT_LANG = "uk"

STRINGS: dict[str, dict[str, str]] = {

    # ── Головне меню ────────────────────────────────────────────────────────────
    "main_menu": {
        "uk": (
            "🤖 <b>Capper</b>\n\n"
            "Два алгоритми:\n"
            "📐 <b>WS Gap</b> — домінування по зважених факторах (Elo, xG, форма)\n"
            "⚡ <b>Monster</b> — нішеві правила по лігах (xG + Elo + форма + ринок)\n\n"
            "Оберіть модель:"
        ),
        "ru": (
            "🤖 <b>Capper</b>\n\n"
            "Два алгоритма:\n"
            "📐 <b>WS Gap</b> — доминирование по взвешенным факторам (Elo, xG, форма)\n"
            "⚡ <b>Monster</b> — нишевые правила по лигам (xG + Elo + форма + рынок)\n\n"
            "Выберите модель:"
        ),
    },

    # ── WS Gap меню ─────────────────────────────────────────────────────────────
    "wsgap_menu": {
        "uk": (
            "📐 <b>WS Gap</b>\n\n"
            "Ставимо тільки там, де одна команда домінує по всіх факторах: "
            "Elo, xG, форма, таблиця — і ринок дає хороший коефіцієнт.\n\n"
            "Top-5 + ЛЧ + Eredivisie + Jupiler + УПЛ"
        ),
        "ru": (
            "📐 <b>WS Gap</b>\n\n"
            "Ставим только там, где одна команда доминирует по всем факторам: "
            "Elo, xG, форма, таблица — и рынок даёт хороший коэффициент.\n\n"
            "Top-5 + ЛЧ + Eredivisie + Jupiler + УПЛ"
        ),
    },

    # ── Aquamarine меню ─────────────────────────────────────────────────────────
    "aquamarine_menu": {
        "uk": (
            "🐬 <b>Aquamarine</b>\n\n"
            "Алгоритм відбирає матчі з підтвердженою статистичною перевагою "
            "по вузьких нішах кожної ліги.\n\n"
            "⚠️ Ставки з коефіцієнтом ≥ 2.5 позначаються як High Risk у нотифікаціях."
        ),
        "ru": (
            "🐬 <b>Aquamarine</b>\n\n"
            "Алгоритм отбирает матчи с подтверждённым статистическим преимуществом "
            "по узким нишам каждой лиги.\n\n"
            "⚠️ Ставки с коэффициентом ≥ 2.5 помечаются как High Risk в уведомлениях."
        ),
    },

    # ── Monster меню ────────────────────────────────────────────────────────────
    "monster_menu": {
        "uk": (
            "⚡ <b>Monster</b>\n\n"
            "Алгоритм аналізує статистичні патерни по кожній лізі окремо "
            "та відбирає матчі з підтвердженою edge-перевагою.\n\n"
            "⚠️ Ставки з коефіцієнтом ≥ 2.5 позначаються як High Risk у нотифікаціях."
        ),
        "ru": (
            "⚡ <b>Monster</b>\n\n"
            "Алгоритм анализирует статистические паттерны по каждой лиге отдельно "
            "и отбирает матчи с подтверждённым edge-преимуществом.\n\n"
            "⚠️ Ставки с коэффициентом ≥ 2.5 помечаются как High Risk в уведомлениях."
        ),
    },

    # ── Кнопки ──────────────────────────────────────────────────────────────────
    "btn_schedule":           {"uk": "📅 Розклад",            "ru": "📅 Расписание"},
    "btn_refresh":            {"uk": "🔄 Оновити пики",       "ru": "🔄 Обновить пики"},
    "btn_picks_wsgap":        {"uk": "📐 Пики",              "ru": "📐 Пики"},
    "btn_picks_monster":      {"uk": "⚡ Пики",              "ru": "⚡ Пики"},
    "btn_picks_aquamarine":   {"uk": "🐬 Пики",              "ru": "🐬 Пики"},
    "btn_stats":              {"uk": "📊 Статистика",         "ru": "📊 Статистика"},
    "btn_history":            {"uk": "📋 Історія",            "ru": "📋 История"},
    "btn_to_monster":         {"uk": "⚡ → Monster",          "ru": "⚡ → Monster"},
    "btn_to_aquamarine":      {"uk": "🐬 → Aquamarine",       "ru": "🐬 → Aquamarine"},
    "btn_to_wsgap":           {"uk": "📐 → WS Gap",           "ru": "📐 → WS Gap"},
    "btn_menu":               {"uk": "🏠 Меню",               "ru": "🏠 Меню"},
    "btn_back_wsgap":         {"uk": "◀ WS Gap",              "ru": "◀ WS Gap"},
    "btn_back_monster":       {"uk": "◀ Monster",             "ru": "◀ Monster"},
    "btn_back_aquamarine":    {"uk": "◀ Aquamarine",          "ru": "◀ Aquamarine"},
    "btn_stats_kelly":        {"uk": "📊 Kelly 25% (без кепу)","ru": "📊 Kelly 25% (без кепа)"},
    "btn_stats_cap":          {"uk": "📊 Kelly 4% Cap",        "ru": "📊 Kelly 4% Cap"},

    # ── Піки ────────────────────────────────────────────────────────────────────
    "picks_no_active_wsgap": {
        "uk": "📐 <b>Пики WS Gap</b>\n\n🤷 Немає активних ставок.",
        "ru": "📐 <b>Пики WS Gap</b>\n\n🤷 Нет активных ставок.",
    },
    "picks_no_active_monster": {
        "uk": "⚡ <b>Пики Monster</b>\n\n🤷 Немає активних ставок.",
        "ru": "⚡ <b>Пики Monster</b>\n\n🤷 Нет активных ставок.",
    },
    "picks_no_active_aquamarine": {
        "uk": "🐬 <b>Пики Aquamarine</b>\n\n🤷 Немає активних ставок.",
        "ru": "🐬 <b>Пики Aquamarine</b>\n\n🤷 Нет активных ставок.",
    },
    # {model_label} {n} {word}
    "picks_header": {
        "uk": "{model_label} <b>Пики — {n} {word}</b>\n",
        "ru": "{model_label} <b>Пики — {n} {word}</b>\n",
    },
    # plurals for n ставок
    "picks_word_1":  {"uk": "ставка",  "ru": "ставка"},
    "picks_word_24": {"uk": "ставки",  "ru": "ставки"},
    "picks_word_5":  {"uk": "ставок",  "ru": "ставок"},

    "side_home": {"uk": "П1", "ru": "П1"},
    "side_away": {"uk": "П2", "ru": "П2"},

    "high_risk_label": {"uk": " ⚠️ <b>HIGH RISK</b>", "ru": " ⚠️ <b>HIGH RISK</b>"},

    "status_win":     {"uk": "✅ +${profit:.0f}",    "ru": "✅ +${profit:.0f}"},
    "status_loss":    {"uk": "❌ −${loss:.0f}",      "ru": "❌ −${loss:.0f}"},
    "status_pending": {"uk": "⏳ ${stake:.0f}",      "ru": "⏳ ${stake:.0f}"},

    # ── Статистика ──────────────────────────────────────────────────────────────
    "stats_no_data": {
        "uk": "📊 <b>Статистика · {title}</b>\n\nСтатистика поки недоступна — ставок немає.",
        "ru": "📊 <b>Статистика · {title}</b>\n\nСтатистика пока недоступна — ставок нет.",
    },
    "stats_title_wsgap_kelly":       {"uk": "WS Gap · Kelly 25%",       "ru": "WS Gap · Kelly 25%"},
    "stats_title_monster_kelly":     {"uk": "Monster · Kelly 25%",      "ru": "Monster · Kelly 25%"},
    "stats_title_monster_cap":       {"uk": "Monster · Kelly 4% Cap",   "ru": "Monster · Kelly 4% Cap"},
    "stats_title_aquamarine_kelly":  {"uk": "Aquamarine · Kelly 25%",   "ru": "Aquamarine · Kelly 25%"},
    "stats_title_aquamarine_cap":    {"uk": "Aquamarine · Kelly 4% Cap","ru": "Aquamarine · Kelly 4% Cap"},

    "stats_pending": {
        "uk": "   ⏳ {n} очікується\n",
        "ru": "   ⏳ {n} ожидается\n",
    },
    "stats_avg_odds": {
        "uk": "\n📈 Avg коеф: {avg:.2f}\n   {streak}\n",
        "ru": "\n📈 Avg коэф: {avg:.2f}\n   {streak}\n",
    },
    "stats_drawdown": {
        "uk": "   📉 Просадка від піку: {dd:.1f}%\n",
        "ru": "   📉 Просадка от пика: {dd:.1f}%\n",
    },

    "streak_wins_n":  {"uk": "🔥 {n} перемог поспіль", "ru": "🔥 {n} побед подряд"},
    "streak_wins_1":  {"uk": "🔥 1 перемога",           "ru": "🔥 1 победа"},
    "streak_losses_n":{"uk": "🔴 {n} поразок поспіль",  "ru": "🔴 {n} поражений подряд"},
    "streak_none":    {"uk": "—",                        "ru": "—"},

    # ── Історія ─────────────────────────────────────────────────────────────────
    "history_no_bets_wsgap": {
        "uk": "📋 <b>Історія WS Gap</b>\n\nСтавок ще немає.",
        "ru": "📋 <b>История WS Gap</b>\n\nСтавок ещё нет.",
    },
    "history_no_bets_monster": {
        "uk": "📋 <b>Історія Monster</b>\n\nСтавок ще немає.",
        "ru": "📋 <b>История Monster</b>\n\nСтавок ещё нет.",
    },
    "history_no_bets_aquamarine": {
        "uk": "📋 <b>Історія Aquamarine</b>\n\nСтавок ще немає.",
        "ru": "📋 <b>История Aquamarine</b>\n\nСтавок ещё нет.",
    },
    "history_no_bets_day": {
        "uk": "Ставок ще немає.",
        "ru": "Ставок ещё нет.",
    },
    "history_no_bets_on_day": {
        "uk": "На {day} ставок немає.",
        "ru": "На {day} ставок нет.",
    },
    "history_day_summary": {
        "uk": "\n<i>День: {sign}${pnl:.0f} · Банкрол: ${bankroll:.0f}</i>",
        "ru": "\n<i>День: {sign}${pnl:.0f} · Банкролл: ${bankroll:.0f}</i>",
    },
    "history_result_pending": {"uk": "очікується", "ru": "ожидается"},
    "history_hr_tag":         {"uk": " ⚠️ HR",     "ru": " ⚠️ HR"},

    # ── Розклад ─────────────────────────────────────────────────────────────────
    "schedule_header": {
        "uk": "{model_label} <b>Розклад ставок</b>\n",
        "ru": "{model_label} <b>Расписание ставок</b>\n",
    },
    "schedule_no_picks_wsgap": {
        "uk": "📐 <b>Розклад WS Gap</b>\n\n📭 Немає запланованих ставок.",
        "ru": "📐 <b>Расписание WS Gap</b>\n\n📭 Нет запланированных ставок.",
    },
    "schedule_no_picks_monster": {
        "uk": "⚡ <b>Розклад Monster</b>\n\n📭 Немає запланованих ставок.",
        "ru": "⚡ <b>Расписание Monster</b>\n\n📭 Нет запланированных ставок.",
    },
    "schedule_no_picks_aquamarine": {
        "uk": "🐬 <b>Розклад Aquamarine</b>\n\n📭 Немає запланованих ставок.",
        "ru": "🐬 <b>Расписание Aquamarine</b>\n\n📭 Нет запланированных ставок.",
    },
    # ── Refresh ──────────────────────────────────────────────────────────────────
    "refresh_started":       {"uk": "Запускаю...",         "ru": "Запускаю..."},
    "refresh_loading_wsgap": {
        "uk": "📐 <b>WS Gap</b>\n\n🔄 Генерую пики на 7 днів вперед...",
        "ru": "📐 <b>WS Gap</b>\n\n🔄 Генерирую пики на 7 дней вперёд...",
    },
    "refresh_loading_monster": {
        "uk": "⚡ <b>Monster</b>\n\n🔄 Генерую пики на 7 днів вперед...",
        "ru": "⚡ <b>Monster</b>\n\n🔄 Генерирую пики на 7 дней вперёд...",
    },
    "refresh_loading_aquamarine": {
        "uk": "🐬 <b>Aquamarine</b>\n\n🔄 Генерую пики на 7 днів вперед...",
        "ru": "🐬 <b>Aquamarine</b>\n\n🔄 Генерирую пики на 7 дней вперёд...",
    },
    "refresh_done_wsgap": {
        "uk": "📐 <b>WS Gap</b>\n\n✅ Оновлено. Нових пиків: <b>{n}</b>",
        "ru": "📐 <b>WS Gap</b>\n\n✅ Обновлено. Новых пиков: <b>{n}</b>",
    },
    "refresh_done_monster": {
        "uk": "⚡ <b>Monster</b>\n\n✅ Оновлено. Нових пиків: <b>{n}</b>",
        "ru": "⚡ <b>Monster</b>\n\n✅ Обновлено. Новых пиков: <b>{n}</b>",
    },
    "refresh_done_aquamarine": {
        "uk": "🐬 <b>Aquamarine</b>\n\n✅ Оновлено. Нових пиків: <b>{n}</b>",
        "ru": "🐬 <b>Aquamarine</b>\n\n✅ Обновлено. Новых пиков: <b>{n}</b>",
    },
    "refresh_error": {
        "uk": "❌ Помилка при оновленні:\n<code>{err}</code>",
        "ru": "❌ Ошибка при обновлении:\n<code>{err}</code>",
    },

    "schedule_label_final": {"uk": "🔒 Фінальна",        "ru": "🔒 Финальная"},
    "schedule_label_soon":  {"uk": "⏳ Скоро (~24г)",     "ru": "⏳ Скоро (~24ч)"},
    "schedule_label_early": {"uk": "🔮 Рання",            "ru": "🔮 Ранняя"},
    "schedule_hr_tag":      {"uk": "⚠️ High Risk",        "ru": "⚠️ High Risk"},
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """
    Get localized string by key.
    Falls back to Ukrainian if key/lang not found.
    Supports format kwargs: t("stats_pending", n=3)
    """
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANG) or key
    return text.format(**kwargs) if kwargs else text


def picks_word(n: int, lang: str = DEFAULT_LANG) -> str:
    """Ukrainian/Russian plural for number of bets."""
    if n == 1:
        return t("picks_word_1", lang)
    if 2 <= n <= 4:
        return t("picks_word_24", lang)
    return t("picks_word_5", lang)
