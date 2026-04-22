import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0
K_FACTOR = 32


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    home_elo: float, away_elo: float, home_score: int, away_score: int
) -> tuple[float, float]:
    """
    Оновлює Elo рейтинги після матчу.
    Повертає (новий_home_elo, новий_away_elo).
    """
    exp_home = expected_score(home_elo, away_elo)
    exp_away = 1 - exp_home

    if home_score > away_score:
        actual_home, actual_away = 1.0, 0.0
    elif home_score == away_score:
        actual_home, actual_away = 0.5, 0.5
    else:
        actual_home, actual_away = 0.0, 1.0

    new_home = home_elo + K_FACTOR * (actual_home - exp_home)
    new_away = away_elo + K_FACTOR * (actual_away - exp_away)

    return new_home, new_away


def elo_features(home_elo: float, away_elo: float) -> dict:
    return {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "elo_home_win_prob": expected_score(home_elo, away_elo),
    }


def build_elo_snapshots(matches_df: pd.DataFrame) -> dict[int, dict[int, float]]:
    """
    Повертає {match_id: {team_id: elo_до_матчу}} для кожного завершеного матчу.
    Без data leakage — кожен матч бачить Elo тільки з попередніх матчів.
    """
    elos: dict[int, float] = {}
    snapshots: dict[int, dict[int, float]] = {}
    finished = matches_df[matches_df["home_score"].notna()].sort_values("date")

    for _, row in finished.iterrows():
        match_id = int(row["id"])
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        elo_h = elos.get(h, DEFAULT_ELO)
        elo_a = elos.get(a, DEFAULT_ELO)
        snapshots[match_id] = {h: elo_h, a: elo_a}
        new_h, new_a = update_elo(elo_h, elo_a, int(row["home_score"]), int(row["away_score"]))
        elos[h] = new_h
        elos[a] = new_a

    return snapshots


def compute_elo_momentum(
    elo_snapshots: dict[int, dict[int, float]],
    matches_df: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    n: int = 10,
) -> float:
    """
    Динаміка Elo: поточний Elo мінус Elo n матчів тому.
    Позитивне = команда в підйомі, негативне = спад форми.
    Повертає 0.0 якщо недостатньо даних.
    """
    team_matches = matches_df[
        ((matches_df["home_team_id"] == team_id) | (matches_df["away_team_id"] == team_id))
        & (matches_df["date"] < before_date)
        & (matches_df["home_score"].notna())
    ].sort_values("date", ascending=False)

    if len(team_matches) < n:
        return np.nan

    recent_id = int(team_matches.iloc[0]["id"])
    old_id    = int(team_matches.iloc[n - 1]["id"])

    recent_elo = elo_snapshots.get(recent_id, {}).get(team_id, DEFAULT_ELO)
    old_elo    = elo_snapshots.get(old_id,    {}).get(team_id, DEFAULT_ELO)

    return recent_elo - old_elo


def compute_dynamic_elo(matches_df: pd.DataFrame) -> dict[int, float]:
    """Повертає {team_id: поточний_elo} після всіх матчів. Для predict path."""
    elos: dict[int, float] = {}
    finished = matches_df[matches_df["home_score"].notna()].sort_values("date")
    for _, row in finished.iterrows():
        h, a = int(row["home_team_id"]), int(row["away_team_id"])
        elo_h = elos.get(h, DEFAULT_ELO)
        elo_a = elos.get(a, DEFAULT_ELO)
        new_h, new_a = update_elo(elo_h, elo_a, int(row["home_score"]), int(row["away_score"]))
        elos[h] = new_h
        elos[a] = new_a
    return elos
