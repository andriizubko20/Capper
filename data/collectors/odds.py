from loguru import logger
from data.api_client import SStatsClient

# Пріоритет букмекерів — беремо першого доступного
BOOKMAKER_PRIORITY = ["Bet365", "Pinnacle", "William Hill", "Bwin", "1xBet", "Unibet"]
REFERENCE_BOOKMAKER = "Bet365"  # для зворотної сумісності

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

    # Індексуємо букмекерів по імені
    bookmakers_by_name = {}
    for bookmaker in data.get("data", []) or []:
        name = bookmaker.get("bookmakerName", "")
        if name:
            bookmakers_by_name[name] = bookmaker

    # Вибираємо найкращого доступного букмекера
    selected_name = None
    for candidate in BOOKMAKER_PRIORITY:
        if candidate in bookmakers_by_name:
            selected_name = candidate
            break
    # Якщо нікого з пріоритетного списку немає — беремо першого доступного
    if selected_name is None and bookmakers_by_name:
        selected_name = next(iter(bookmakers_by_name))

    if selected_name is None:
        logger.info(f"Fetched 0 odds entries for fixture {fixture_id}")
        return []

    results = []
    for market in bookmakers_by_name[selected_name].get("odds", []):
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
                    "bookmaker": selected_name,
                    "outcome": outcome_name,
                    "odds": float(outcome["value"]),
                    "opening_odds": float(outcome["openingValue"]) if outcome.get("openingValue") else None,
                    "is_closing": False,
                })
            except (ValueError, TypeError, KeyError):
                continue

    logger.info(f"Fetched {len(results)} odds entries for fixture {fixture_id} via {selected_name}")
    return results


def fetch_closing_odds(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Closing odds — фінальні коефіцієнти перед стартом матчу (для розрахунку CLV)."""
    results = fetch_odds(fixture_id, client)
    for r in results:
        r["is_closing"] = True
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
