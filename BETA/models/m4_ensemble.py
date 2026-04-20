"""
BETA/models/m4_ensemble.py

M4 — Ensemble of M1 + M2 + M3.
Averages probabilities from all three models.
Bet only when 2/3 models agree on the same side being +EV.
"""
import numpy as np


class M4Ensemble:
    """
    Wraps M1, M2, M3. Averages their probability outputs.
    Agreement filter: bet only when at least 2/3 models independently
    flag the same side as EV-positive.

    fit() and predict_proba() take extra kwargs passed through to M3.
    """

    def __init__(self, m1, m2, m3, weights=(1.0, 1.0, 1.0)):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.weights = np.array(weights) / sum(weights)
        self.feature_importances_ = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y, matches_df=None, **kwargs):
        self.m1.fit(X, y)
        self.m2.fit(X, y)
        self.m3.fit(X, y, matches_df=matches_df)

        # Average feature importances from M1+M2
        fi1 = getattr(self.m1, 'feature_importances_', None)
        fi2 = getattr(self.m2, 'feature_importances_', None)
        if fi1 is not None and fi2 is not None:
            self.feature_importances_ = (fi1 + fi2) / 2.0
        return self

    def predict_proba(self, X, home_ids=None, away_ids=None, **kwargs) -> np.ndarray:
        p1 = self.m1.predict_proba(X)
        p2 = self.m2.predict_proba(X)
        p3 = self.m3.predict_proba(X, home_ids=home_ids, away_ids=away_ids)

        # Weighted average
        combined = (self.weights[0] * p1 +
                    self.weights[1] * p2 +
                    self.weights[2] * p3)
        # Re-normalize rows
        combined = combined / combined.sum(axis=1, keepdims=True)
        return combined

    def predict_proba_with_agreement(
        self, X, home_ids=None, away_ids=None,
        odds_home=None, odds_away=None, min_ev=0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (proba, agreement_mask).
        agreement_mask[i, side] = True if ≥2 models say this side is +EV.
        side: 0=home, 2=away
        """
        p1 = self.m1.predict_proba(X)
        p2 = self.m2.predict_proba(X)
        p3 = self.m3.predict_proba(X, home_ids=home_ids, away_ids=away_ids)

        combined = (self.weights[0] * p1 +
                    self.weights[1] * p2 +
                    self.weights[2] * p3)
        combined = combined / combined.sum(axis=1, keepdims=True)

        n = len(X)
        agreement = np.zeros((n, 3), dtype=bool)

        if odds_home is not None and odds_away is not None:
            for i in range(n):
                h_odds = odds_home[i] if odds_home[i] else 0
                a_odds = odds_away[i] if odds_away[i] else 0

                # Count models that see +EV for home
                votes_h = sum([
                    1 if p1[i, 0] * h_odds - 1 >= min_ev else 0,
                    1 if p2[i, 0] * h_odds - 1 >= min_ev else 0,
                    1 if p3[i, 0] * h_odds - 1 >= min_ev else 0,
                ])
                votes_a = sum([
                    1 if p1[i, 2] * a_odds - 1 >= min_ev else 0,
                    1 if p2[i, 2] * a_odds - 1 >= min_ev else 0,
                    1 if p3[i, 2] * a_odds - 1 >= min_ev else 0,
                ])
                agreement[i, 0] = votes_h >= 2
                agreement[i, 2] = votes_a >= 2

        return combined, agreement
