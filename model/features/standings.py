from collections import defaultdict
import pandas as pd


def build_standings_snapshots(matches_df: pd.DataFrame) -> dict[int, dict]:
    """
    Pre-computes standings features for every finished match in O(n) time.
    Returns {match_id: standings_features_dict} where features are taken BEFORE the match.
    """
    finished = matches_df[matches_df["home_score"].notna()].sort_values("date")

    # Per-league running state: {league_id: {team_id: {points, goal_diff, goals_for, played}}}
    league_tables: dict[int, dict[int, dict]] = defaultdict(dict)
    snapshots: dict[int, dict] = {}

    _neutral = {
        "home_table_position": 10.0, "home_table_points": 0, "home_table_goal_diff": 0,
        "away_table_position": 10.0, "away_table_points": 0, "away_table_goal_diff": 0,
        "table_position_diff": 0.0, "table_points_diff": 0.0,
    }

    for _, row in finished.iterrows():
        match_id = int(row["id"])
        league_id = int(row["league_id"])
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        hs, as_ = int(row["home_score"]), int(row["away_score"])

        table = league_tables[league_id]
        n_teams = len(table)

        if n_teams >= 4:
            sorted_teams = sorted(
                table.items(),
                key=lambda x: (-x[1]["points"], -x[1]["goal_diff"], -x[1]["goals_for"])
            )
            pos_map = {tid: i + 1 for i, (tid, _) in enumerate(sorted_teams)}
            mid = (n_teams + 1) / 2
            h_entry = table.get(h, {"points": 0, "goal_diff": 0})
            a_entry = table.get(a, {"points": 0, "goal_diff": 0})
            snapshots[match_id] = {
                "home_table_position": float(pos_map.get(h, mid)),
                "home_table_points": h_entry["points"],
                "home_table_goal_diff": h_entry["goal_diff"],
                "away_table_position": float(pos_map.get(a, mid)),
                "away_table_points": a_entry["points"],
                "away_table_goal_diff": a_entry["goal_diff"],
                "table_position_diff": float(pos_map.get(h, mid)) - float(pos_map.get(a, mid)),
                "table_points_diff": float(h_entry["points"] - a_entry["points"]),
            }
        else:
            snapshots[match_id] = dict(_neutral)

        # Update table with this match result
        for tid in (h, a):
            if tid not in table:
                table[tid] = {"points": 0, "goal_diff": 0, "goals_for": 0, "played": 0}

        table[h]["played"] += 1
        table[a]["played"] += 1
        table[h]["goals_for"] += hs
        table[a]["goals_for"] += as_
        table[h]["goal_diff"] += hs - as_
        table[a]["goal_diff"] += as_ - hs

        if hs > as_:
            table[h]["points"] += 3
        elif hs == as_:
            table[h]["points"] += 1
            table[a]["points"] += 1
        else:
            table[a]["points"] += 3

    return snapshots


def compute_standings(
    matches: pd.DataFrame,
    league_id: int,
    before_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Таблиця ліги для league_id на дату before_date.
    Повертає DataFrame: team_id, points, goal_diff, goals_for, played, position.
    """
    relevant = matches[
        (matches["league_id"] == league_id)
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    ]

    if relevant.empty:
        return pd.DataFrame(columns=["team_id", "points", "goal_diff", "goals_for", "played", "position"])

    records: dict[int, dict] = {}

    def _entry(team_id: int) -> dict:
        return records.setdefault(team_id, {"points": 0, "goal_diff": 0, "goals_for": 0, "played": 0})

    for _, row in relevant.iterrows():
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        hs, as_ = int(row["home_score"]), int(row["away_score"])

        _entry(h)["played"] += 1
        _entry(a)["played"] += 1
        _entry(h)["goals_for"] += hs
        _entry(a)["goals_for"] += as_
        _entry(h)["goal_diff"] += hs - as_
        _entry(a)["goal_diff"] += as_ - hs

        if hs > as_:
            _entry(h)["points"] += 3
        elif hs == as_:
            _entry(h)["points"] += 1
            _entry(a)["points"] += 1
        else:
            _entry(a)["points"] += 3

    df = pd.DataFrame([{"team_id": tid, **stats} for tid, stats in records.items()])
    df = df.sort_values(["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
    df["position"] = df.index + 1
    return df


_NEUTRAL = {
    "home_table_position": 10.0,
    "away_table_position": 10.0,
    "table_points_diff": 0.0,
}


def compute_standings_features(
    matches: pd.DataFrame,
    league_id: int,
    home_team_id: int,
    away_team_id: int,
    before_date: pd.Timestamp,
) -> dict:
    standings = compute_standings(matches, league_id, before_date)
    n_teams = len(standings)

    # Менше 4 команд — CL плей-оф або початок сезону, таблиця не має сенсу
    if n_teams < 4:
        return _NEUTRAL

    mid_position = (n_teams + 1) / 2

    def _team_row(team_id: int) -> dict:
        row = standings[standings["team_id"] == team_id]
        if row.empty:
            return {"position": mid_position, "points": 0, "goal_diff": 0}
        r = row.iloc[0]
        return {"position": int(r["position"]), "points": int(r["points"]), "goal_diff": int(r["goal_diff"])}

    home = _team_row(home_team_id)
    away = _team_row(away_team_id)

    return {
        "home_table_position": home["position"],
        "home_table_points": home["points"],
        "home_table_goal_diff": home["goal_diff"],
        "away_table_position": away["position"],
        "away_table_points": away["points"],
        "away_table_goal_diff": away["goal_diff"],
        "table_position_diff": home["position"] - away["position"],
        "table_points_diff": home["points"] - away["points"],
    }
