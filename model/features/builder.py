import pandas as pd
from model.features.form import compute_form, compute_home_away_form, compute_rest_days, compute_form_advanced
from model.features.xg import compute_xg_features, compute_xg_overperformance
from model.features.elo import elo_features, build_elo_snapshots, compute_elo_momentum
from model.features.h2h import compute_h2h
from model.features.odds_features import market_implied_features, odds_movement_features
from model.features.standings import build_standings_snapshots
from model.features.injuries import compute_injury_features
from model.features.match_stats_features import compute_match_stats_features, compute_efficiency_features


def build_match_features(
    match: dict,
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    teams: dict,  # {team_id: {"elo": float}}
    odds: dict | None = None,
    opening_odds: dict | None = None,
    injuries_df: pd.DataFrame | None = None,
    standings_snap: dict | None = None,
    elo_snapshots: dict | None = None,
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

    # Розширена форма: clean sheet, btts, streak, variability
    home_adv_form = compute_form_advanced(matches_df, home_id, date, n=10)
    away_adv_form = compute_form_advanced(matches_df, away_id, date, n=10)
    features.update({f"home_{k}": v for k, v in home_adv_form.items()})
    features.update({f"away_{k}": v for k, v in away_adv_form.items()})
    features["delta_clean_sheet_rate"]     = home_adv_form["clean_sheet_rate"]     - away_adv_form["clean_sheet_rate"]
    features["delta_failed_to_score_rate"] = home_adv_form["failed_to_score_rate"] - away_adv_form["failed_to_score_rate"]
    features["delta_btts_rate"]            = home_adv_form["btts_rate"]            - away_adv_form["btts_rate"]

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

    # Match stats rolling avg (shots, possession, corners, saves, passes) — останні 5
    home_ms5 = compute_match_stats_features(stats_df, home_id, date, n=5)
    away_ms5 = compute_match_stats_features(stats_df, away_id, date, n=5)
    features.update({f"home_{k}": v for k, v in home_ms5.items()})
    features.update({f"away_{k}": v for k, v in away_ms5.items()})
    for k in home_ms5:
        features[f"delta_{k}"] = home_ms5[k] - away_ms5[k]

    # Ефективність (conversion, accuracy, save%, pass%) — останні 10
    home_eff = compute_efficiency_features(stats_df, home_id, date, n=10)
    away_eff = compute_efficiency_features(stats_df, away_id, date, n=10)
    features.update({f"home_{k}": v for k, v in home_eff.items()})
    features.update({f"away_{k}": v for k, v in away_eff.items()})
    for k in home_eff:
        features[f"delta_{k}"] = home_eff[k] - away_eff[k]

    # Elo (динамічний, передається через teams)
    home_elo = teams.get(home_id, {}).get("elo", 1500.0)
    away_elo = teams.get(away_id, {}).get("elo", 1500.0)
    features.update(elo_features(home_elo, away_elo))

    # Elo momentum (тренд за останні 10 матчів)
    if elo_snapshots is not None:
        features["home_elo_momentum"] = compute_elo_momentum(elo_snapshots, matches_df, home_id, date)
        features["away_elo_momentum"] = compute_elo_momentum(elo_snapshots, matches_df, away_id, date)
        features["delta_elo_momentum"] = features["home_elo_momentum"] - features["away_elo_momentum"]
    else:
        features["home_elo_momentum"] = 0.0
        features["away_elo_momentum"] = 0.0
        features["delta_elo_momentum"] = 0.0

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

    # Pre-index DC odds (1X, 2X)
    odds_dc = odds_df[odds_df["market"] == "double_chance"]
    dc_by_match: dict[int, pd.DataFrame] = {}
    for mid, grp in odds_dc.groupby("match_id"):
        dc_by_match[int(mid)] = grp

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
            elo_snapshots=elo_snapshots,
        )

        if match["home_score"] > match["away_score"]:
            feats["target"] = "home"
        elif match["home_score"] == match["away_score"]:
            feats["target"] = "draw"
        else:
            feats["target"] = "away"

        # DC odds (1X, 2X) з реального ринку
        dc_grp = dc_by_match.get(match_id)
        if dc_grp is not None and not dc_grp.empty:
            for outcome in ("1X", "2X"):
                best = dc_grp[dc_grp["outcome"] == outcome]["value"].max()
                if best and not pd.isna(best):
                    feats[f"dc_{outcome.lower()}_odds"] = best

        # Зберігаємо сирі odds для бектестів (до margin adjustment)
        if odds_dict:
            feats["home_odds"] = odds_dict["home"]
            feats["away_odds"] = odds_dict["away"]
            feats["draw_odds"] = odds_dict["draw"]

        feats["match_id"] = match_id
        feats["date"] = match["date"]
        rows.append(feats)

    return pd.DataFrame(rows)
