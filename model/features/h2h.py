import pandas as pd


def compute_h2h(
    matches: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    before_date: pd.Timestamp,
    n: int = 5,
) -> dict:
    """
    Head-to-head статистика між двома командами.
    Враховує матчі в обох напрямках.
    """
    h2h = matches[
        (
            ((matches["home_team_id"] == home_team_id) & (matches["away_team_id"] == away_team_id))
            | ((matches["home_team_id"] == away_team_id) & (matches["away_team_id"] == home_team_id))
        )
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    ].sort_values("date", ascending=False).head(n)

    if h2h.empty:
        return _empty_h2h()

    home_wins = 0
    away_wins = 0
    draws = 0
    goals_home_team = []
    goals_away_team = []

    for _, row in h2h.iterrows():
        if row["home_team_id"] == home_team_id:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        goals_home_team.append(gf)
        goals_away_team.append(ga)

        if gf > ga:
            home_wins += 1
        elif gf == ga:
            draws += 1
        else:
            away_wins += 1

    total = len(h2h)
    return {
        "h2h_home_win_rate": home_wins / total,
        "h2h_away_win_rate": away_wins / total,
        "h2h_draw_rate": draws / total,
        "h2h_home_goals_avg": sum(goals_home_team) / total,
        "h2h_away_goals_avg": sum(goals_away_team) / total,
        "h2h_matches_count": total,
    }


def _empty_h2h() -> dict:
    return {
        "h2h_home_win_rate": 0.45,
        "h2h_away_win_rate": 0.30,
        "h2h_draw_rate": 0.25,
        "h2h_home_goals_avg": 1.3,
        "h2h_away_goals_avg": 1.1,
        "h2h_matches_count": 0,
    }
