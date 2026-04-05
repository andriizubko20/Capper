from loguru import logger
from data.api_client import SStatsClient

REFERENCE_BOOKMAKER = "Bet365"

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


def fetch_odds(fixture_id: int, client: SStatsClient) -> list[dict]:
    logger.info(f"Fetching odds for fixture {fixture_id}")
    data = client.get(f"/Odds/{fixture_id}")

    results = []
    for bookmaker in data.get("data", []) or []:
        bookmaker_name = bookmaker.get("bookmakerName", "")
        if bookmaker_name != REFERENCE_BOOKMAKER:
            continue
        for market in bookmaker.get("odds", []):
            market_name = market.get("marketName", "")
            market_key = MARKET_MAP.get(market_name)
            if not market_key:
                continue
            for outcome in market.get("odds", []):
                try:
                    raw_outcome = outcome["name"].lower()
                    if market_key == "double_chance":
                        mapped = DC_OUTCOME_MAP.get(raw_outcome)
                        if not mapped:
                            continue
                        outcome_name = mapped
                    else:
                        outcome_name = raw_outcome
                    results.append({
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

    logger.info(f"Fetched {len(results)} odds entries for fixture {fixture_id}")
    return results


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
