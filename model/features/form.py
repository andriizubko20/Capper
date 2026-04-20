import pandas as pd


def compute_form(
    matches: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    n: int = 5,
    league_id: int | None = None,
) -> dict:
    """
    Розраховує форму команди за останні n матчів до вказаної дати.
    league_id: якщо вказано — тільки матчі цієї ліги (для league-only форми).
    """
    mask = (
        ((matches["home_team_id"] == team_id) | (matches["away_team_id"] == team_id))
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    )
    if league_id is not None and "league_id" in matches.columns:
        mask = mask & (matches["league_id"] == league_id)
    team_matches = matches[mask].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return _empty_form(prefix="")

    results = []
    goals_for = []
    goals_against = []

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]
        goals_for.append(gf)
        goals_against.append(ga)
        if gf > ga:
            results.append(3)
        elif gf == ga:
            results.append(1)
        else:
            results.append(0)

    return {
        "form_points": sum(results) / (len(results) * 3),           # нормалізовано 0..1
        "form_goals_for_avg": sum(goals_for) / len(goals_for),
        "form_goals_against_avg": sum(goals_against) / len(goals_against),
        "form_wins": results.count(3) / len(results),
        "form_draws": results.count(1) / len(results),
        "form_losses": results.count(0) / len(results),
    }


def compute_home_away_form(
    matches: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    side: str,
    n: int = 5,
    league_id: int | None = None,
) -> dict:
    """
    Форма тільки вдома або тільки на виїзді.
    side: 'home' або 'away'
    league_id: якщо вказано — тільки матчі цієї ліги.
    """
    col = "home_team_id" if side == "home" else "away_team_id"
    mask = (
        (matches[col] == team_id)
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    )
    if league_id is not None and "league_id" in matches.columns:
        mask = mask & (matches["league_id"] == league_id)
    team_matches = matches[mask].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return _empty_form(prefix=f"{side}_")

    results = []
    goals_for = []
    goals_against = []

    for _, row in team_matches.iterrows():
        is_home = side == "home"
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]
        goals_for.append(gf)
        goals_against.append(ga)
        if gf > ga:
            results.append(3)
        elif gf == ga:
            results.append(1)
        else:
            results.append(0)

    return {
        f"{side}_form_points": sum(results) / (len(results) * 3),
        f"{side}_form_goals_for_avg": sum(goals_for) / len(goals_for),
        f"{side}_form_goals_against_avg": sum(goals_against) / len(goals_against),
        f"{side}_form_wins": results.count(3) / len(results),
        f"{side}_form_losses": results.count(0) / len(results),
    }


def compute_form_advanced(
    matches: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """
    Розширені показники форми:
    - clean_sheet_rate — % матчів без пропущеного гола
    - failed_to_score_rate — % матчів без забитого гола
    - btts_rate — % матчів, де обидві команди забили
    - win_streak — поточна серія перемог
    - loss_streak — поточна серія поразок
    - goals_for_std — варіативність атаки (непостійна команда = ненадійна)
    """
    mask = (
        ((matches["home_team_id"] == team_id) | (matches["away_team_id"] == team_id))
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    )
    team_matches = matches[mask].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return {
            "clean_sheet_rate": 0.25,
            "failed_to_score_rate": 0.2,
            "btts_rate": 0.5,
            "win_streak": 0,
            "loss_streak": 0,
            "goals_for_std": 1.0,
        }

    clean_sheets = 0
    failed_to_score = 0
    btts = 0
    goals_for = []

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id
        gf = int(row["home_score"] if is_home else row["away_score"])
        ga = int(row["away_score"] if is_home else row["home_score"])
        goals_for.append(gf)
        if ga == 0:
            clean_sheets += 1
        if gf == 0:
            failed_to_score += 1
        if gf > 0 and ga > 0:
            btts += 1

    # Поточна серія (найостанніші матчі)
    win_streak = 0
    loss_streak = 0
    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id
        gf = int(row["home_score"] if is_home else row["away_score"])
        ga = int(row["away_score"] if is_home else row["home_score"])
        if gf > ga:
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        elif gf < ga:
            if win_streak == 0:
                loss_streak += 1
            else:
                break
        else:
            break

    n_matches = len(team_matches)
    import statistics
    goals_std = statistics.stdev(goals_for) if len(goals_for) >= 2 else 1.0

    return {
        "clean_sheet_rate": clean_sheets / n_matches,
        "failed_to_score_rate": failed_to_score / n_matches,
        "btts_rate": btts / n_matches,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "goals_for_std": goals_std,
    }


def compute_rest_days(matches: pd.DataFrame, team_id: int, before_date: pd.Timestamp) -> dict:
    """Дні відпочинку з останнього матчу до before_date."""
    prior = matches[
        ((matches["home_team_id"] == team_id) | (matches["away_team_id"] == team_id))
        & (matches["date"] < before_date)
        & (matches["home_score"].notna())
    ].sort_values("date", ascending=False)

    if prior.empty:
        return {"rest_days": 7}

    last_date = pd.Timestamp(prior.iloc[0]["date"])
    delta = (before_date - last_date).days
    return {"rest_days": min(delta, 30)}


def _empty_form(prefix: str) -> dict:
    return {
        f"{prefix}form_points": 0.5,
        f"{prefix}form_goals_for_avg": 1.3,
        f"{prefix}form_goals_against_avg": 1.3,
        f"{prefix}form_wins": 0.33,
        f"{prefix}form_draws": 0.25,
        f"{prefix}form_losses": 0.33,
    }
