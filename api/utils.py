"""Спільні утиліти для API: маппінги моделей, ліг, форматування пікс."""
from __future__ import annotations
from datetime import datetime, timezone

# model_version → display name + color
MODEL_META = {
    "ws_gap_v1":           {"name": "WS Gap",  "color": "#F472B6"},
    "ws_gap_kelly_v1":     {"name": "WS Gap",  "color": "#F472B6"},
    "monster_v1":          {"name": "Monster", "color": "#F59E0B"},
    "monster_v1_kelly":    {"name": "Monster", "color": "#F59E0B"},
    "aquamarine_v1":       {"name": "Aqua",    "color": "#22D3EE"},
    "aquamarine_v1_kelly": {"name": "Aqua",    "color": "#22D3EE"},
}

# model query param → list of model_version values
MODEL_VERSIONS: dict[str, list[str]] = {
    "WS Gap":  ["ws_gap_kelly_v1"],
    "Monster": ["monster_v1_kelly"],
    "Aqua":    ["aquamarine_v1_kelly"],
}

LEAGUE_FLAGS: dict[str, str] = {
    "Premier League":    "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":           "🇪🇸",
    "Bundesliga":        "🇩🇪",
    "Serie A":           "🇮🇹",
    "Ligue 1":           "🇫🇷",
    "Champions League":  "🏆",
    "UEFA Champions League": "🏆",
}

# SStats status codes that mean "live"
LIVE_STATUSES_EXCLUDE = {"Finished", "FT", "Not Started", "Scheduled", ""}


def format_side(market: str, outcome: str) -> str:
    """Перетворює market+outcome на відображуваний side для фронта."""
    m = market.lower()
    if m == "1x2":
        return outcome.upper()           # HOME / AWAY / DRAW
    if m == "total":
        return outcome.upper()           # OVER 2.5 / UNDER 2.5
    if m == "btts":
        return f"BTTS {outcome.upper()}" # BTTS YES / BTTS NO
    if m == "double_chance":
        return outcome.upper()
    if m == "handicap":
        return f"HCP {outcome}"
    return outcome.upper()


def match_status_label(match_status: str, match_date: datetime) -> str:
    """Повертає 'live' | 'pending' | 'finished'."""
    s = match_status.strip()
    if s in ("Finished", "FT"):
        return "finished"
    now = datetime.now(timezone.utc)
    match_dt = match_date.replace(tzinfo=timezone.utc) if match_date.tzinfo is None else match_date
    if match_dt <= now and s not in ("Not Started", "Scheduled", ""):
        return "live"
    return "pending"


def format_time(match_date: datetime, elapsed: int | None = None) -> str:
    """Повертає рядок часу: для live — хвилину, для pending — HH:MM."""
    if elapsed is not None:
        return f"{elapsed}'"
    return match_date.strftime("%H:%M")
