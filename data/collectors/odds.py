from loguru import logger
from data.api_client import SStatsClient

# Reference bookmaker — used for CLV (sharpest line preferred). Keep stable
# across collections so closing-line comparisons are apples-to-apples.
REFERENCE_BOOKMAKER = "Pinnacle"
REFERENCE_BOOKMAKER_FALLBACKS = ["Bet365", "William Hill", "Bwin", "1xBet", "Unibet"]

# Маппінг ринків SStats → наші назви
MARKET_MAP = {
    "Match Winner": "1x2",
    "Goals Over/Under": "total",
    "Both Teams Score": "btts",
    "Asian Handicap": "handicap",
    "Double Chance": "double_chance",
}

# Маппінг outcomes Double Chance → наші назви
DC_OUTCOME_MAP = {
    "home/draw": "1X",
    "draw/away": "2X",
    "home/away": "12",
}


def _parse_bookmaker_block(fixture_id: int, bookmaker_name: str, markets: list[dict]) -> list[dict]:
    """Parse one bookmaker's odds block into normalized rows."""
    rows: list[dict] = []
    for market in markets or []:
        market_name = market.get("marketName", "")
        market_key = MARKET_MAP.get(market_name)
        if not market_key:
            continue
        for outcome in market.get("odds", []) or []:
            try:
                raw_outcome = outcome["name"].lower()
                if market_key == "double_chance":
                    mapped = DC_OUTCOME_MAP.get(raw_outcome)
                    if not mapped:
                        continue
                    outcome_name = mapped
                else:
                    outcome_name = raw_outcome
                rows.append({
                    "fixture_id": fixture_id,
                    "market": market_key,
                    "bookmaker": bookmaker_name,
                    "outcome": outcome_name,
                    "odds": float(outcome["value"]),
                    "opening_odds": float(outcome["openingValue"]) if outcome.get("openingValue") else None,
                    "is_closing": False,
                })
            except (ValueError, TypeError, KeyError):
                continue
    return rows


def fetch_odds(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Fetch odds across ALL bookmakers (one row per bookmaker × market × outcome).

    Bookmaker shopping: we keep every bookmaker so downstream code can pick the
    BEST price per outcome. Previously we filtered to one bookmaker via priority
    list and threw the rest away (~3-5% EV uplift left on the table)."""
    logger.info(f"Fetching odds for fixture {fixture_id}")
    data = client.get(f"/Odds/{fixture_id}")

    results: list[dict] = []
    bookmakers_seen: set[str] = set()
    for bookmaker in data.get("data", []) or []:
        name = bookmaker.get("bookmakerName", "") or ""
        if not name:
            continue
        bookmakers_seen.add(name)
        results.extend(_parse_bookmaker_block(fixture_id, name, bookmaker.get("odds", [])))

    logger.info(
        f"Fetched {len(results)} odds rows for fixture {fixture_id} "
        f"across {len(bookmakers_seen)} bookmakers"
    )
    return results


def fetch_closing_odds(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Closing odds — фінальні коефіцієнти перед стартом матчу (для розрахунку CLV)."""
    results = fetch_odds(fixture_id, client)
    for r in results:
        r["is_closing"] = True
    return results


def fetch_reference_closing_odds(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Closing odds from REFERENCE_BOOKMAKER only (for CLV comparison).

    Falls back through REFERENCE_BOOKMAKER_FALLBACKS if Pinnacle is missing,
    then to the first available bookmaker. Returns rows with is_closing=True."""
    all_rows = fetch_closing_odds(fixture_id, client)
    if not all_rows:
        return []

    # Group available bookmakers
    by_bm: dict[str, list[dict]] = {}
    for r in all_rows:
        by_bm.setdefault(r["bookmaker"], []).append(r)

    selected = None
    for candidate in [REFERENCE_BOOKMAKER, *REFERENCE_BOOKMAKER_FALLBACKS]:
        if candidate in by_bm:
            selected = candidate
            break
    if selected is None:
        selected = next(iter(by_bm))

    return by_bm[selected]


def fetch_line_movement(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Рух лінії — зміни коефіцієнтів (sharp money signal)."""
    logger.info(f"Fetching line movement for fixture {fixture_id}")
    data = client.get(f"/Odds/live-changes/{fixture_id}")
    return data.get("data") or []


def get_best_odds(odds_list: list[dict], market: str, outcome: str) -> float | None:
    """Повертає найкращий коефіцієнт по ринку і виходу серед всіх букмекерів."""
    relevant = [
        o["odds"] for o in odds_list
        if o["market"] == market and o["outcome"] == outcome and o["odds"] > 1
    ]
    return max(relevant) if relevant else None
