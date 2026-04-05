from data.api_client import SStatsClient
from loguru import logger

# Топ-5 + Ліга чемпіонів — ID з SStats.net
TRACKED_LEAGUES = {
    39: ("Premier League", "England"),
    140: ("La Liga", "Spain"),
    78: ("Bundesliga", "Germany"),
    135: ("Serie A", "Italy"),
    61: ("Ligue 1", "France"),
    2: ("Champions League", "Europe"),
}


def fetch_leagues(client: SStatsClient) -> list[dict]:
    logger.info("Fetching all leagues")
    data = client.get("/Leagues")
    leagues = []
    for item in data.get("data", []):
        leagues.append({
            "api_id": item["id"],
            "name": item["name"],
            "country": item.get("country", {}).get("name", ""),
        })
    logger.info(f"Fetched {len(leagues)} leagues")
    return leagues


def fetch_teams(league_id: int, season: int, client: SStatsClient) -> list[dict]:
    logger.info(f"Fetching teams for league {league_id}, season {season}")
    data = client.get("/Teams/list", params={"LeagueId": league_id, "Year": season})
    teams = []
    for item in data.get("data", []):
        teams.append({
            "api_id": item["id"],
            "name": item["name"],
            "country": item.get("country", {}).get("name", ""),
        })
    logger.info(f"Fetched {len(teams)} teams for league {league_id}")
    return teams
