"""
BETA/models/m3_poisson.py

M3 — Dixon-Coles Poisson goal model.

Models goals as:
  Home goals ~ Poisson(lambda_h = attack_h × defense_a × home_advantage × mu)
  Away goals ~ Poisson(lambda_a = attack_a × defense_h × mu)

With Dixon-Coles correction for low-score matches (0:0, 1:0, 0:1, 1:1).

Team parameters are estimated via MLE using scipy.optimize.
The model has no sklearn-style fit/predict_proba interface — it works
directly on match-level data and returns P(H/D/A) per match.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from collections import defaultdict


# ── Dixon-Coles correction ─────────────────────────────────────────────────

def _dc_correction(h_goals: int, a_goals: int, lambda_h: float,
                   lambda_a: float, rho: float) -> float:
    """
    Bivariate Poisson correction factor for low-scoring matches.
    rho ≈ -0.13 (typical fitted value).
    """
    if h_goals == 0 and a_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif h_goals == 1 and a_goals == 0:
        return 1 + lambda_a * rho
    elif h_goals == 0 and a_goals == 1:
        return 1 + lambda_h * rho
    elif h_goals == 1 and a_goals == 1:
        return 1 - rho
    return 1.0


def _match_prob_matrix(lambda_h: float, lambda_a: float,
                        rho: float = -0.13, max_goals: int = 8) -> np.ndarray:
    """Returns (max_goals+1) × (max_goals+1) matrix of P(h_goals, a_goals)."""
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
            p *= _dc_correction(h, a, lambda_h, lambda_a, rho)
            mat[h, a] = max(p, 1e-10)
    return mat


def _hda_from_matrix(mat: np.ndarray) -> tuple[float, float, float]:
    """Extract P(home win), P(draw), P(away win) from score matrix."""
    p_home = np.sum(np.tril(mat, -1))  # h > a
    p_draw = np.sum(np.diag(mat))
    p_away = np.sum(np.triu(mat, 1))
    total  = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ── Model class ────────────────────────────────────────────────────────────

class M3Poisson:
    """
    Dixon-Coles Poisson model.

    sklearn-like interface:
      fit(X, y, matches_df=...) — ignores X/y, fits from matches_df directly
      predict_proba(X, matches_df=...) — returns (n, 3) P(H/D/A)
    """

    def __init__(
        self,
        weight_days: float = 365.0,   # time-decay half-life in days
        rho:         float = -0.13,   # DC correction
        max_goals:   int   = 8,
        home_adv:    float = 0.1,     # initial home advantage log-param
    ):
        self.weight_days = weight_days
        self.rho         = rho
        self.max_goals   = max_goals
        self.home_adv    = home_adv

        self.attack_:  dict[int, float] = {}
        self.defense_: dict[int, float] = {}
        self.home_adv_: float = home_adv
        self.mu_:       float = 1.3   # average goals
        self.fitted_:   bool  = False

        # pseudo-sklearn
        self.feature_importances_ = None
        self.classes_ = np.array([0, 1, 2])

    def _time_weight(self, match_date, latest_date) -> float:
        days = (latest_date - match_date).days
        return np.exp(-days / self.weight_days)

    def _neg_log_likelihood(self, params, matches_df, team_ids, latest_date):
        """Negative log-likelihood for optimization."""
        n_teams = len(team_ids)
        idx = {tid: i for i, tid in enumerate(team_ids)}

        # params layout: [attack × n_teams, defense × n_teams, home_adv, rho]
        attack  = np.exp(params[:n_teams])
        defense = np.exp(params[n_teams:2*n_teams])
        home    = np.exp(params[2*n_teams])
        rho     = params[2*n_teams + 1]

        total_nll = 0.0
        for _, row in matches_df.iterrows():
            h, a = int(row.home_team_id), int(row.away_team_id)
            if h not in idx or a not in idx:
                continue
            hi, ai = idx[h], idx[a]

            lh = attack[hi] * defense[ai] * home
            la = attack[ai] * defense[hi]

            h_g = int(row.home_score)
            a_g = int(row.away_score)

            p = (poisson.logpmf(h_g, max(lh, 1e-6)) +
                 poisson.logpmf(a_g, max(la, 1e-6)))

            dc = _dc_correction(h_g, a_g, lh, la, rho)
            p += np.log(max(dc, 1e-10))

            w = self._time_weight(row.date, latest_date)
            total_nll -= w * p

        return total_nll

    def fit(self, X, y, matches_df=None, **kwargs):
        """Fit team attack/defense parameters from historical matches."""
        if matches_df is None:
            raise ValueError("M3Poisson.fit() requires matches_df= kwarg")

        df = matches_df.dropna(subset=['home_score','away_score']).copy()
        df['date'] = pd.to_datetime(df['date']) if hasattr(df['date'].iloc[0], 'date') else df['date']
        latest_date = df['date'].max()

        team_ids = sorted(set(df.home_team_id) | set(df.away_team_id))
        n = len(team_ids)

        # Initial params: attack=1, defense=1, home=1.1, rho=-0.13
        x0 = np.zeros(2 * n + 2)
        x0[2*n]   = np.log(1.1)   # home advantage
        x0[2*n+1] = -0.13         # rho

        # Bounds: rho in (-0.5, 0.5)
        bounds = [(None, None)] * (2*n) + [(None, None), (-0.5, 0.5)]

        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(df, team_ids, latest_date),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6},
        )

        idx = {tid: i for i, tid in enumerate(team_ids)}
        self.attack_   = {tid: np.exp(res.x[i])          for tid, i in idx.items()}
        self.defense_  = {tid: np.exp(res.x[n + i])      for tid, i in idx.items()}
        self.home_adv_ = np.exp(res.x[2*n])
        self.rho_      = res.x[2*n + 1]
        self.fitted_   = True

        return self

    def predict_hda(self, home_id: int, away_id: int) -> tuple[float, float, float]:
        """Returns (P_home, P_draw, P_away) for a single match."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted")

        att_h = self.attack_.get(home_id,  1.0)
        def_h = self.defense_.get(home_id, 1.0)
        att_a = self.attack_.get(away_id,  1.0)
        def_a = self.defense_.get(away_id, 1.0)

        lh = att_h * def_a * self.home_adv_
        la = att_a * def_h

        mat = _match_prob_matrix(lh, la, self.rho_, self.max_goals)
        return _hda_from_matrix(mat)

    def predict_proba(self, X, match_ids=None, home_ids=None, away_ids=None, **kwargs) -> np.ndarray:
        """
        Returns (n, 3) array [P(H), P(D), P(A)].
        Requires home_ids and away_ids lists of same length as X.
        """
        if home_ids is None or away_ids is None:
            raise ValueError("predict_proba() requires home_ids= and away_ids= kwargs")

        result = np.zeros((len(home_ids), 3))
        for i, (h, a) in enumerate(zip(home_ids, away_ids)):
            ph, pd_, pa = self.predict_hda(int(h), int(a))
            result[i] = [ph, pd_, pa]
        return result


# Import needed inside fit
import pandas as pd
