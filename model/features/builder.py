import pandas as pd
from model.features.form import compute_form, compute_home_away_form, compute_rest_days
from model.features.xg import compute_xg_features, compute_xg_overperformance
from model.features.elo import elo_features, build_elo_snapshots
from model.features.h2h import compute_h2h
from model.features.odds_features import market_implied_features, odds_movement_features
from model.features.standings import build_standings_snapshots
from model.features.injuries import compute_injury_features


def build_match_features(
    match: dict,
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    teams: dict,  # {team_id: {"elo": float}}
    odds: dict | None = None,
    opening_odds: dict | None = None,
    injuries_df: pd.DataFrame | None = None,
    standings_snap: dict | None = None,
) -> dict:
    date = pd.Timestamp(match["date"])
    home_id = match["home_team_id"]
    away_id = match["away_team_id"]

    features = {}

    # Форма (загальна, останні 5)
    home_form = compute_form(matches_df, home_id, date, n=5)
    away_form = compute_form(matches_df, away_id, date, n=5)
    features.update({f"home_{k}": v for k, v in home_form.items()})
    features.update({f"away_{k}": v for k, v in away_form.items()})

    # Форма вдома / на виїзді
    home_home_form = compute_home_away_form(matches_df, home_id, date, side="home", n=5)
    away_away_form = compute_home_away_form(matches_df, away_id, date, side="away", n=5)
    features.update({f"home_{k}": v for k, v in home_home_form.items()})
    features.update({f"away_{k}": v for k, v in away_away_form.items()})

    # xG останні 5
    home_xg5 = compute_xg_features(stats_df, home_id, date, n=5)
    away_xg5 = compute_xg_features(stats_df, away_id, date, n=5)
    features.update({f"home_{k}": v for k, v in home_xg5.items()})
    features.update({f"away_{k}": v for k, v in away_xg5.items()})

    # xG останні 10
    home_xg10 = compute_xg_features(stats_df, home_id, date, n=10)
    away_xg10 = compute_xg_features(stats_df, away_id, date, n=10)
    features.update({f"home_{k}_10": v for k, v in home_xg10.items()})
    features.update({f"away_{k}_10": v for k, v in away_xg10.items()})

    # xG overperformance (клінічність реалізації)
    home_xg_op = compute_xg_overperformance(stats_df, home_id, date)
    away_xg_op = compute_xg_overperformance(stats_df, away_id, date)
    features["home_xg_overperformance"] = home_xg_op["xg_overperformance"]
    features["away_xg_overperformance"] = away_xg_op["xg_overperformance"]

    # Elo (динамічний, передається через teams)
    home_elo = teams.get(home_id, {}).get("elo", 1500.0)
    away_elo = teams.get(away_id, {}).get("elo", 1500.0)
    features.update(elo_features(home_elo, away_elo))

    # H2H прибрано з фіч — нерепрезентативний (мало зустрічей, давні дані)

    # Дні відпочинку
    home_rest = compute_rest_days(matches_df, home_id, date)
    away_rest = compute_rest_days(matches_df, away_id, date)
    features["home_rest_days"] = home_rest["rest_days"]
    features["away_rest_days"] = away_rest["rest_days"]
    features["rest_days_diff"] = home_rest["rest_days"] - away_rest["rest_days"]

    # Місце в таблиці (pre-computed snapshot or fallback)
    if standings_snap is not None:
        features.update(standings_snap)
    else:
        from model.features.standings import compute_standings_features
        features.update(compute_standings_features(
            matches_df, match["league_id"], home_id, away_id, date
        ))

    # Травми
    if injuries_df is not None and not injuries_df.empty:
        inj = compute_injury_features(
            injuries_df,
            match_id=match.get("id"),
            home_team_id=home_id,
            away_team_id=away_id,
        )
        features.update(inj)

    # Ринкові коефіцієнти
    if odds:
        market = market_implied_features(odds["home"], odds["draw"], odds["away"])
        features.update(market)

        if opening_odds:
            for side in ("home", "draw", "away"):
                movement = odds_movement_features(opening_odds.get(side, 0), odds.get(side, 0))
                features.update({f"{side}_{k}": v for k, v in movement.items()})

    features["league_id"] = match["league_id"]

    return features


def build_dataset(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    teams: dict,
    injuries_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    finished = matches_df[matches_df["home_score"].notna()].copy()

    # Pre-compute snapshots once (O(n) each instead of O(n²))
    elo_snapshots = build_elo_snapshots(matches_df)
    standings_snapshots = build_standings_snapshots(matches_df)

    # Pre-index odds by match_id (avoids 1.6M-row scan per match)
    odds_1x2 = odds_df[odds_df["market"] == "1x2"]
    odds_by_match: dict[int, pd.DataFrame] = {}
    for mid, grp in odds_1x2.groupby("match_id"):
        odds_by_match[int(mid)] = grp

    rows = []
    for _, match in finished.iterrows():
        match_id = int(match["id"])

        # Elo з снепшоту (без data leakage)
        elo_snap = elo_snapshots.get(match_id, {})
        dynamic_teams = {
            match["home_team_id"]: {"elo": elo_snap.get(match["home_team_id"], 1500.0)},
            match["away_team_id"]: {"elo": elo_snap.get(match["away_team_id"], 1500.0)},
        }

        match_odds_grp = odds_by_match.get(match_id)
        odds_dict = None
        if match_odds_grp is not None and not match_odds_grp.empty:
            closing = match_odds_grp[match_odds_grp["is_closing"] == True]
            source = closing if not closing.empty else match_odds_grp
            best_odds: dict[str, float] = {}
            for _, row in source.iterrows():
                outcome = row["outcome"]
                if outcome not in best_odds or row["value"] > best_odds[outcome]:
                    best_odds[outcome] = row["value"]
            if {"home", "draw", "away"}.issubset(best_odds):
                odds_dict = best_odds

        feats = build_match_features(
            match=match.to_dict(),
            matches_df=matches_df,
            stats_df=stats_df,
            teams=dynamic_teams,
            odds=odds_dict,
            injuries_df=injuries_df,
            standings_snap=standings_snapshots.get(match_id),
        )

        if match["home_score"] > match["away_score"]:
            feats["target"] = "home"
        elif match["home_score"] == match["away_score"]:
            feats["target"] = "draw"
        else:
            feats["target"] = "away"

        feats["match_id"] = match_id
        feats["date"] = match["date"]
        rows.append(feats)

    return pd.DataFrame(rows)
