"""
Rolling averages of match statistics (shots, possession, corners, saves, passes).
Requires stats_df to have columns: home_shots_on_target, away_shots_on_target,
home_shots_inside_box, away_shots_inside_box, home_possession, away_possession,
home_corners, away_corners, home_gk_saves, away_gk_saves,
home_passes_accurate, away_passes_accurate.
"""
import pandas as pd


_EMPTY_EFF = {
    "shot_conversion_rate": 0.30,
    "shot_accuracy": 0.35,
    "shots_box_ratio": 0.50,
    "save_pct": 0.70,
    "pass_accuracy_pct": 0.82,
}

_EMPTY = {
    "shots_ot_for_avg": 3.5,
    "shots_ot_against_avg": 3.5,
    "shots_box_for_avg": 5.0,
    "shots_box_against_avg": 5.0,
    "possession_avg": 50.0,
    "corners_for_avg": 4.5,
    "gk_saves_avg": 2.5,
    "passes_acc_for_avg": 250.0,
}


def compute_efficiency_features(
    stats: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    n: int = 10,
) -> dict:
    """
    Ефективність команди на основі shots + goals + passes.
    - shot_conversion_rate = goals / shots_on_target
    - shot_accuracy         = shots_on_target / total_shots
    - shots_box_ratio       = shots_inside_box / total_shots
    - save_pct              = gk_saves / opp_shots_on_target  (якість воротаря)
    - pass_accuracy_pct     = passes_accurate / passes_total
    Потребує: home_shots, home_shots_on_target, home_shots_inside_box,
              home_gk_saves, home_passes_accurate, home_passes_total,
              home_score, away_score у stats_df.
    """
    required_cols = {"home_shots_on_target", "home_shots", "home_score"}
    if not required_cols.issubset(stats.columns):
        return _EMPTY_EFF.copy()

    team_matches = stats[
        ((stats["home_team_id"] == team_id) | (stats["away_team_id"] == team_id))
        & (stats["date"] < before_date)
        & (stats["home_shots_on_target"].notna())
        & (stats["home_score"].notna())
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return _EMPTY_EFF.copy()

    conversions, accuracies, box_ratios, save_pcts, pass_accs = [], [], [], [], []

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id

        goals      = float(row["home_score"]  if is_home else row["away_score"])
        sot_for    = row["home_shots_on_target"]  if is_home else row["away_shots_on_target"]
        shots_for  = row.get("home_shots")        if is_home else row.get("away_shots")
        sib_for    = row.get("home_shots_inside_box") if is_home else row.get("away_shots_inside_box")
        sot_against = row["away_shots_on_target"] if is_home else row["home_shots_on_target"]
        gk_saves   = row.get("home_gk_saves")     if is_home else row.get("away_gk_saves")
        passes_acc = row.get("home_passes_accurate") if is_home else row.get("away_passes_accurate")
        passes_tot = row.get("home_passes_total")  if is_home else row.get("away_passes_total")

        if pd.notna(sot_for) and sot_for > 0:
            conversions.append(goals / sot_for)
        if pd.notna(sot_for) and pd.notna(shots_for) and shots_for > 0:
            accuracies.append(sot_for / shots_for)
        if pd.notna(sib_for) and pd.notna(shots_for) and shots_for > 0:
            box_ratios.append(sib_for / shots_for)
        if pd.notna(gk_saves) and pd.notna(sot_against) and sot_against > 0:
            save_pcts.append(gk_saves / sot_against)
        if pd.notna(passes_acc) and pd.notna(passes_tot) and passes_tot > 0:
            pass_accs.append(passes_acc / passes_tot)

    def avg(lst, default):
        return sum(lst) / len(lst) if lst else default

    return {
        "shot_conversion_rate": avg(conversions, _EMPTY_EFF["shot_conversion_rate"]),
        "shot_accuracy":        avg(accuracies,  _EMPTY_EFF["shot_accuracy"]),
        "shots_box_ratio":      avg(box_ratios,  _EMPTY_EFF["shots_box_ratio"]),
        "save_pct":             avg(save_pcts,   _EMPTY_EFF["save_pct"]),
        "pass_accuracy_pct":    avg(pass_accs,   _EMPTY_EFF["pass_accuracy_pct"]),
    }


def compute_match_stats_features(
    stats: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
    n: int = 5,
) -> dict:
    """Rolling averages of match stats for a team over last n games."""
    required = "home_shots_on_target"
    if required not in stats.columns:
        return _EMPTY.copy()

    team_matches = stats[
        ((stats["home_team_id"] == team_id) | (stats["away_team_id"] == team_id))
        & (stats["date"] < before_date)
        & (stats["home_shots_on_target"].notna())
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return _EMPTY.copy()

    shots_ot_for, shots_ot_against = [], []
    shots_box_for, shots_box_against = [], []
    possession, corners_for = [], []
    gk_saves, passes_acc_for = [], []

    for _, row in team_matches.iterrows():
        is_home = row["home_team_id"] == team_id

        shots_ot_for.append(row["home_shots_on_target"] if is_home else row["away_shots_on_target"])
        shots_ot_against.append(row["away_shots_on_target"] if is_home else row["home_shots_on_target"])

        if pd.notna(row.get("home_shots_inside_box")):
            shots_box_for.append(row["home_shots_inside_box"] if is_home else row["away_shots_inside_box"])
            shots_box_against.append(row["away_shots_inside_box"] if is_home else row["home_shots_inside_box"])

        if pd.notna(row.get("home_possession")):
            possession.append(row["home_possession"] if is_home else row["away_possession"])

        if pd.notna(row.get("home_corners")):
            corners_for.append(row["home_corners"] if is_home else row["away_corners"])

        if pd.notna(row.get("home_gk_saves")):
            gk_saves.append(row["home_gk_saves"] if is_home else row["away_gk_saves"])

        if pd.notna(row.get("home_passes_accurate")):
            passes_acc_for.append(row["home_passes_accurate"] if is_home else row["away_passes_accurate"])

    def avg(lst, default):
        return sum(lst) / len(lst) if lst else default

    return {
        "shots_ot_for_avg":      avg(shots_ot_for, _EMPTY["shots_ot_for_avg"]),
        "shots_ot_against_avg":  avg(shots_ot_against, _EMPTY["shots_ot_against_avg"]),
        "shots_box_for_avg":     avg(shots_box_for, _EMPTY["shots_box_for_avg"]),
        "shots_box_against_avg": avg(shots_box_against, _EMPTY["shots_box_against_avg"]),
        "possession_avg":        avg(possession, _EMPTY["possession_avg"]),
        "corners_for_avg":       avg(corners_for, _EMPTY["corners_for_avg"]),
        "gk_saves_avg":          avg(gk_saves, _EMPTY["gk_saves_avg"]),
        "passes_acc_for_avg":    avg(passes_acc_for, _EMPTY["passes_acc_for_avg"]),
    }
