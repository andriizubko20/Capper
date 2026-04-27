"""Спільні утиліти для API: маппінги моделей, ліг, форматування пікс."""
from __future__ import annotations
from datetime import datetime, timezone

# model_version → display name + color
MODEL_META = {
    "ws_gap_kelly_v1":     {"name": "WS Gap",   "color": "#F472B6"},
    "monster_v1_kelly":    {"name": "Monster",  "color": "#F59E0B"},
    "aquamarine_v1_kelly": {"name": "Aqua",     "color": "#22D3EE"},
    "pure_v1":             {"name": "Pure",     "color": "#9D4EDD"},
    "gem_v1":              {"name": "Gem",      "color": "#10B981"},
    "gem_v2_kmeans3":      {"name": "Gem v2",   "color": "#34D399"},
}

# model query param → list of model_version values
MODEL_VERSIONS: dict[str, list[str]] = {
    "WS Gap":  ["ws_gap_kelly_v1"],
    "Monster": ["monster_v1_kelly"],
    "Aqua":    ["aquamarine_v1_kelly"],
    "Pure":    ["pure_v1"],
    "Gem":     ["gem_v1"],
    "Gem v2":  ["gem_v2_kmeans3"],
}

# Flag lookup is hybrid:
#   - (name, country) tuple keys disambiguate collisions ("Premier League" lives
#     in both England and Ukraine).
#   - Plain-string keys are used as a fallback when country is unknown or for
#     leagues whose name is globally unique.
# api/main.py::prediction_to_pick tries the tuple key first, then the string.
LEAGUE_FLAGS: dict[tuple[str, str] | str, str] = {
    # Disambiguated by country
    ("Premier League", "England"): "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    ("Premier League", "Ukraine"): "🇺🇦",
    ("Champions League", "Europe"): "🏆",
    # Globally unique names (fallback)
    "Premier League":        "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":               "🇪🇸",
    "Bundesliga":            "🇩🇪",
    "Serie A":               "🇮🇹",
    "Ligue 1":               "🇫🇷",
    "Champions League":      "🏆",
    "UEFA Champions League": "🏆",
    "Eredivisie":            "🇳🇱",
    "Jupiler Pro League":    "🇧🇪",
    "Premier League (UA)":   "🇺🇦",
    "Ukrainian Premier League": "🇺🇦",
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
