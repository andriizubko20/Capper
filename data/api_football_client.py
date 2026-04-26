"""
data/api_football_client.py

Thin client for API-Football v3 (api-sports.io).

Free tier: 100 requests/day — caller must batch and cache aggressively.
Mirrors the structure of SStatsClient (httpx.Client + retry/backoff on 429
and network errors).

Endpoints currently used:
  - GET /injuries?team={id}&season={season}
  - GET /players/topscorers?league={id}&season={season}
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from config.settings import settings


NETWORK_RETRY_DELAY = 60   # seconds between retries on transport-level errors
NETWORK_MAX_RETRIES = 10   # ~10 minutes max before bailing out


class APIFootballError(Exception):
    pass


class APIFootballClient:
    """Synchronous client for the API-Football v3 endpoints we consume.

    The free tier allows only ~100 calls/day, so callers should:
      * deduplicate requests within a job (one call per team/league/day),
      * persist results to the DB and refresh on a daily/weekly cadence,
      * never hit the API in tight loops without DELAY between calls.
    """

    BASE_URL = "https://v3.football.api-sports.io/"

    def __init__(self, api_key: str | None = None):
        key = api_key or getattr(settings, "api_football_key", "") or ""
        if not key:
            raise APIFootballError(
                "API_FOOTBALL_KEY is not set — add `api_football_key` to "
                "config/settings.py or export API_FOOTBALL_KEY in .env"
            )
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "x-apisports-key": key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    # ---- low-level transport ------------------------------------------------

    def get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """GET with 429 backoff and transport retry, mirroring SStatsClient."""
        attempt = 0
        while True:
            try:
                resp = self._client.get(endpoint, params=params)
                resp.raise_for_status()
                payload = resp.json()
                # API-Football wraps everything in {"errors": ..., "response": [...]}
                errors = payload.get("errors")
                if errors:
                    # `errors` is sometimes a dict, sometimes a list — both falsy when empty
                    if isinstance(errors, dict) and errors:
                        logger.warning(f"API-Football returned errors: {errors} ({endpoint})")
                    elif isinstance(errors, list) and errors:
                        logger.warning(f"API-Football returned errors: {errors} ({endpoint})")
                return payload
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    attempt += 1
                    wait = min(10 * attempt, 60)
                    logger.warning(
                        f"API-Football rate limit 429 {endpoint} — "
                        f"retry in {wait}s (attempt {attempt})"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"API-Football error: {status} {endpoint}")
                    raise APIFootballError(f"HTTP {status}: {endpoint}") from e
            except httpx.RequestError as e:
                attempt += 1
                if attempt >= NETWORK_MAX_RETRIES:
                    logger.error(
                        f"API-Football connection failed after {NETWORK_MAX_RETRIES} "
                        f"retries: {endpoint}"
                    )
                    raise APIFootballError(
                        f"Connection failed after {NETWORK_MAX_RETRIES} retries: {endpoint}"
                    ) from e
                logger.warning(
                    f"API-Football connection error: {endpoint} — "
                    f"retry in {NETWORK_RETRY_DELAY}s (attempt {attempt}/{NETWORK_MAX_RETRIES})"
                )
                time.sleep(NETWORK_RETRY_DELAY)

    # ---- typed endpoints ----------------------------------------------------

    def get_injuries(self, team_id: int, season: int | None = None) -> list[dict]:
        """Return current injury reports for a team.

        Free-tier note: each call costs 1 request — call once per team per day.
        """
        params: dict[str, Any] = {"team": team_id}
        if season is not None:
            params["season"] = season
        payload = self.get("injuries", params=params)
        return payload.get("response", []) or []

    def get_top_scorers(self, league_id: int, season: int) -> list[dict]:
        """Return the top scorers for a league/season — used to derive xg_share."""
        payload = self.get(
            "players/topscorers",
            params={"league": league_id, "season": season},
        )
        return payload.get("response", []) or []

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "APIFootballClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()
