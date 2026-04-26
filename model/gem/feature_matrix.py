"""
model/gem/feature_matrix.py

Assembles (X, y, info) from load_historical() output.

X    : np.ndarray (n, n_features)  — feature matrix for training
y    : np.ndarray (n,)              — 0=H, 1=D, 2=A
info : pd.DataFrame                 — match_id, date, league, teams, scores,
                                       result, odds (for simulation only)

League priors are injected BY FOLD at training time via the
`league_encoder` argument — here we only leave those 3 columns as NaN
so a per-fold encoder can fill them. At inference time, pass the
final encoder fit on all training data.
"""
import numpy as np
import pandas as pd
from loguru import logger

from model.gem.features import build_gem_features, expected_feature_names
from model.gem.preprocessing import LeagueTargetEncoder

RESULT_MAP = {"H": 0, "D": 1, "A": 2}


def build_feature_matrix(
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    odds: pd.DataFrame,
    injuries: pd.DataFrame,
    team_state: dict,
    h2h: dict,
    league_encoder: LeagueTargetEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds the full X/y/info triple.

    If `league_encoder` is provided, its priors are injected into every row.
    At training time, leave it as None for the initial pass — the trainer
    will recompute X per-fold with a fold-fit encoder to avoid leakage.
    """
    stats_by_mid = stats.set_index("match_id", drop=False) if not stats.empty else stats
    odds_by_mid = odds.set_index("match_id", drop=False) if not odds.empty else odds
    injury_set = set(zip(injuries["match_id"], injuries["team_id"])) if not injuries.empty else set()

    feature_names = expected_feature_names()

    rows: list[list[float | None]] = []
    labels: list[int] = []
    info_rows: list[dict] = []

    for m in matches.itertuples(index=False):
        mid = m.match_id

        home_state = team_state.get((mid, "home"))
        away_state = team_state.get((mid, "away"))
        if home_state is None or away_state is None:
            continue

        home_glicko_prob = None
        away_glicko_prob = None
        if mid in stats_by_mid.index:
            s = stats_by_mid.loc[mid]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            home_glicko_prob = s["home_win_prob"]
            away_glicko_prob = s["away_win_prob"]

        home_odds = draw_odds = away_odds = None
        if mid in odds_by_mid.index:
            o = odds_by_mid.loc[mid]
            if isinstance(o, pd.DataFrame):
                o = o.iloc[0]
            home_odds = o["home_odds"]
            draw_odds = o["draw_odds"]
            away_odds = o["away_odds"]

        home_inj = (mid, m.home_team_id) in injury_set
        away_inj = (mid, m.away_team_id) in injury_set

        priors = league_encoder.transform_row(m.league_name) if league_encoder else None

        feat = build_gem_features(
            match_date=m.date,
            league_canonical=m.league_name,  # already in canonical "Country: Name" form
            home_state=home_state,
            away_state=away_state,
            h2h=h2h.get(mid, {}),
            home_glicko_prob=home_glicko_prob,
            away_glicko_prob=away_glicko_prob,
            home_has_injuries=home_inj,
            away_has_injuries=away_inj,
            league_priors=priors,
        )

        rows.append([feat.get(f) for f in feature_names])
        labels.append(RESULT_MAP[m.result])
        info_rows.append({
            "match_id":       mid,
            "date":           m.date,
            "league_name":    m.league_name,
            "home_team_id":   m.home_team_id,
            "away_team_id":   m.away_team_id,
            "home_score":     m.home_score,
            "away_score":     m.away_score,
            "result":         m.result,
            "home_odds":      home_odds,
            "draw_odds":      draw_odds,
            "away_odds":      away_odds,
        })

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    info = pd.DataFrame(info_rows)

    logger.info(
        f"Feature matrix: X={X.shape}, y={y.shape} "
        f"(H={int((y==0).sum()):,} D={int((y==1).sum()):,} A={int((y==2).sum()):,})"
    )
    return X, y, info


def inject_league_priors(
    X: np.ndarray,
    info: pd.DataFrame,
    encoder: LeagueTargetEncoder,
    feature_names: list[str],
) -> np.ndarray:
    """
    In-place replacement of the 3 prior columns using a (possibly fold-fit) encoder.
    Returns a NEW array (does not mutate input).
    """
    X = X.copy()
    idx_home = feature_names.index("league_prior_home_wr")
    idx_draw = feature_names.index("league_prior_draw_rate")
    idx_away = feature_names.index("league_prior_away_wr")
    for i, lg in enumerate(info["league_name"].to_numpy()):
        r = encoder.transform_row(lg)
        X[i, idx_home] = r["home_wr"]
        X[i, idx_draw] = r["draw_rate"]
        X[i, idx_away] = r["away_wr"]
    return X
