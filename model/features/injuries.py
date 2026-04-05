import pandas as pd


def compute_injury_features(
    injuries_df: pd.DataFrame,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
) -> dict:
    """
    Кількість травмованих гравців по командах для конкретного матчу.
    injuries_df: DataFrame з колонками match_id, team_id, player_api_id.
    """
    match_injuries = injuries_df[injuries_df["match_id"] == match_id]

    if match_injuries.empty:
        return {"home_injured_count": 0, "away_injured_count": 0, "injured_count_diff": 0}

    home_count = int((match_injuries["team_id"] == home_team_id).sum())
    away_count = int((match_injuries["team_id"] == away_team_id).sum())

    return {
        "home_injured_count": home_count,
        "away_injured_count": away_count,
        "injured_count_diff": home_count - away_count,
    }
