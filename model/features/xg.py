import pandas as pd


def compute_xg_features(
    stats: pd.DataFrame, team_id: int, before_date: pd.Timestamp, n: int = 5
) -> dict:
    """
    xG ознаки команди за останні n матчів.
    stats — DataFrame з колонками: date, home_team_id, away_team_id, home_xg, away_xg
    """
    team_matches = stats[
        ((stats["home_team_id"] == team_id) | (stats["away_team_id"] == team_id))
        & (stats["date"] < before_date)
        & (stats["home_xg"].notna())
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return _empty_xg()

    xg_for = []
    xg_against = []

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id
        xg_for.append(row["home_xg"] if is_home else row["away_xg"])
        xg_against.append(row["away_xg"] if is_home else row["home_xg"])

    xg_for_avg = sum(xg_for) / len(xg_for)
    xg_against_avg = sum(xg_against) / len(xg_against)

    return {
        "xg_for_avg": xg_for_avg,
        "xg_against_avg": xg_against_avg,
        "xg_diff_avg": xg_for_avg - xg_against_avg,
        "xg_ratio": xg_for_avg / xg_against_avg if xg_against_avg > 0 else 1.0,
    }


def compute_xg_overperformance(
    stats: pd.DataFrame, team_id: int, before_date: pd.Timestamp, n: int = 10
) -> dict:
    """
    xG overperformance: (реальні голи - xG) за останні n матчів.
    Позитивне = команда реалізує більше ніж очікується (клінічна завершуваність).
    Негативне = недореалізовує моменти.
    """
    team_matches = stats[
        ((stats["home_team_id"] == team_id) | (stats["away_team_id"] == team_id))
        & (stats["date"] < before_date)
        & (stats["home_xg"].notna())
        & (stats["home_score"].notna())
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return {"xg_overperformance": 0.0}

    diffs = []
    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id
        goals = row["home_score"] if is_home else row["away_score"]
        xg = row["home_xg"] if is_home else row["away_xg"]
        diffs.append(float(goals) - float(xg))

    return {"xg_overperformance": sum(diffs) / len(diffs)}


def _empty_xg() -> dict:
    return {
        "xg_for_avg": 1.3,
        "xg_against_avg": 1.3,
        "xg_diff_avg": 0.0,
        "xg_ratio": 1.0,
    }
