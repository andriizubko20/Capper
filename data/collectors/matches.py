from datetime import date
from loguru import logger
from data.api_client import SStatsClient
from data.collectors.leagues import TRACKED_LEAGUES

# Статус матчу: 8 = Finished, 0/1 = Not Started
STATUS_FINISHED = 8
STATUS_SCHEDULED = [0, 1]


def fetch_fixtures(
    league_id: int,
    season: int,
    client: SStatsClient,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict]:
    params = {"LeagueId": league_id, "Year": season}
    if date_from:
        params["dateFrom"] = date_from.isoformat()
    if date_to:
        params["dateTo"] = date_to.isoformat()

    logger.info(f"Fetching fixtures for league {league_id}, season {season}")
    data = client.get("/Games/list", params=params)

    matches = []
    for item in data.get("data", []):
        home = item.get("homeTeam", {})
        away = item.get("awayTeam", {})
        matches.append({
            "api_id": item["id"],
            "league_id": league_id,
            "date": item.get("date"),
            "status": item.get("statusName", ""),
            "home_team_api_id": home.get("id"),
            "home_team_name": home.get("name", ""),
            "home_team_country": home.get("country", {}).get("name", "") if isinstance(home.get("country"), dict) else "",
            "away_team_api_id": away.get("id"),
            "away_team_name": away.get("name", ""),
            "away_team_country": away.get("country", {}).get("name", "") if isinstance(away.get("country"), dict) else "",
            "home_score": item.get("homeFTResult"),
            "away_score": item.get("awayFTResult"),
        })

    logger.info(f"Fetched {len(matches)} fixtures for league {league_id}")
    return matches


def fetch_fixture_glicko(fixture_id: int, client: SStatsClient) -> dict:
    """Glicko-2 рейтинги + xG для матчу."""
    logger.info(f"Fetching glicko/xG for fixture {fixture_id}")
    data = client.get(f"/Games/glicko/{fixture_id}")
    glicko = (data.get("data") or {}).get("glicko") or {}

    return {
        "home_xg": glicko.get("homeXg"),
        "away_xg": glicko.get("awayXg"),
        "home_glicko": glicko.get("homeRating"),
        "away_glicko": glicko.get("awayRating"),
        "home_win_prob": glicko.get("homeWinProbability"),
        "draw_prob": glicko.get("drawProbability"),
        "away_win_prob": glicko.get("awayWinProbability"),
    }


def fetch_injuries(fixture_id: int, client: SStatsClient) -> list[dict]:
    """Травми і дискваліфікації для матчу."""
    logger.info(f"Fetching injuries for fixture {fixture_id}")
    data = client.get("/Games/injuries", params={"gameId": fixture_id})
    injuries = []
    for item in data.get("data", []) or []:
        injuries.append({
            "player_id": item["player"]["id"],
            "player_name": item["player"]["name"],
            "team_id": item["teamId"],
            "reason": item.get("reason", ""),
        })
    return injuries


def fetch_all_fixtures_today(season: int, client: SStatsClient) -> list[dict]:
    today = date.today()
    all_fixtures = []
    for league_id in TRACKED_LEAGUES:
        fixtures = fetch_fixtures(
            league_id=league_id,
            season=season,
            client=client,
            date_from=today,
            date_to=today,
        )
        all_fixtures.extend(fixtures)
    return all_fixtures
