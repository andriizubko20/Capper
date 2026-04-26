"""
data/collectors/apifootball_fallback.py

Fallback fixture status lookup via API-Football. Used when SStats returns
stale data (e.g. status='Not Started' for matches that have already kicked
off — observed for postponed/rescheduled fixtures).

Same fixture IDs as SStats (both on api-sports.io platform).

Free tier: 100 req/day, 10 req/min. Use sparingly — only call as fallback,
not as primary.
"""
from __future__ import annotations

import httpx
from loguru import logger

from config.settings import settings

API_HOST = "https://v3.football.api-sports.io"

# API-Football status codes that mean the match has finished.
FINISHED_STATUSES = {"FT", "AET", "PEN"}


def fetch_fixture_status(fixture_api_id: int) -> dict | None:
    """
    Returns dict with keys: status_short, status_long, home_score, away_score,
    elapsed (or None if request fails / fixture not found).

    Example:
        {"status_short": "FT", "status_long": "Match Finished",
         "home_score": 2, "away_score": 3, "elapsed": 90}
    """
    key = settings.api_football_key
    if not key:
        logger.warning("[ApiFootball] API_FOOTBALL_KEY not set — cannot fall back")
        return None
    try:
        resp = httpx.get(
            f"{API_HOST}/fixtures",
            params={"id": fixture_api_id},
            headers={
                "x-rapidapi-key": key,
                "x-rapidapi-host": "v3.football.api-sports.io",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        d = resp.json()
        if not d.get("response"):
            return None
        fx = d["response"][0]
        return {
            "status_short": fx["fixture"]["status"]["short"],
            "status_long":  fx["fixture"]["status"]["long"],
            "home_score":   fx["goals"]["home"],
            "away_score":   fx["goals"]["away"],
            "elapsed":      fx["fixture"]["status"].get("elapsed"),
        }
    except httpx.RequestError as e:
        logger.warning(f"[ApiFootball] Network error for {fixture_api_id}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.warning(f"[ApiFootball] Bad response shape for {fixture_api_id}: {e}")
        return None


def is_finished(status: dict | None) -> bool:
    return bool(status and status.get("status_short") in FINISHED_STATUSES)
