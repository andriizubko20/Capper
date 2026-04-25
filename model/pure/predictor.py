"""
model/pure/predictor.py

Production interface: given an upcoming match, decide whether Pure picks it.

Usage:
  from model.pure.predictor import PurePredictor
  pp = PurePredictor.load()
  pick = pp.predict(match_features)   # PurePick or None
  if pick:
      print(pick.side, pick.odds, pick.p_is, pick.ev, pick.kelly_fraction)

`match_features` must contain (per-match perspective):
  league_name, home_odds, draw_odds, away_odds,
  home_glicko, away_glicko, home_glicko_prob, away_glicko_prob,
  xg_diff_home, xg_diff_away, attack_vs_def_home, attack_vs_def_away,
  form_advantage  (home - away PPG)
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model.pure.niches import features_for_side, matches_niche

ARTIFACTS = Path(__file__).parent / "artifacts"

KELLY_FRAC = 0.25
KELLY_CAP  = 0.10


@dataclass
class PurePick:
    league_name: str
    niche_id:    str
    side:        str
    odds:        float
    p_is:        float
    p_is_lower_95: float
    ev:          float
    kelly_fraction: float
    matched_filters: dict   # the niche dict for transparency


class PurePredictor:
    """Loads niches.json once and answers predict(match)."""

    def __init__(self, niches_by_league: dict):
        self._niches = niches_by_league

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PurePredictor":
        path = Path(path or ARTIFACTS / "niches.json")
        with open(path) as f:
            niches = json.load(f)
        return cls(niches)

    @property
    def supported_leagues(self) -> list[str]:
        return list(self._niches.keys())

    def predict(self, match: dict) -> Optional[PurePick]:
        """Return best matching PurePick or None."""
        league = match.get("league_name")
        if league is None or league not in self._niches:
            return None
        league_niches = self._niches[league]

        # De-vig market probs (used in matching)
        home_odds = match.get("home_odds")
        draw_odds = match.get("draw_odds")
        away_odds = match.get("away_odds")
        if not (home_odds and draw_odds and away_odds):
            return None
        raw_sum = 1 / home_odds + 1 / draw_odds + 1 / away_odds
        match["home_market_prob"] = (1 / home_odds) / raw_sum
        match["away_market_prob"] = (1 / away_odds) / raw_sum

        # glicko_gap from home perspective + form_advantage too
        if match.get("glicko_gap") is None and match.get("home_glicko") is not None and match.get("away_glicko") is not None:
            match["glicko_gap"] = match["home_glicko"] - match["away_glicko"]

        best: Optional[dict] = None
        for side in ("home", "away"):
            feats = features_for_side(match, side)
            if feats.get("odds") is None:
                continue
            for niche in league_niches.get(side, []):
                if matches_niche(feats, niche, side):
                    if best is None or niche["p_is_lower_95"] > best["p_is_lower_95"]:
                        best = {**niche, "_side": side, "_odds": feats["odds"]}

        if best is None:
            return None

        side = best["_side"]
        odds = best["_odds"]
        p = best["p_is"]
        ev = p * odds - 1
        if ev <= 0:
            return None

        b = odds - 1
        f_star = max(0.0, (p * b - (1 - p)) / b) if b > 0 else 0.0
        kelly = min(KELLY_FRAC * f_star, KELLY_CAP)

        return PurePick(
            league_name=match["league_name"],
            niche_id=best["niche_id"],
            side=side,
            odds=odds,
            p_is=p,
            p_is_lower_95=best["p_is_lower_95"],
            ev=ev,
            kelly_fraction=kelly,
            matched_filters={k: v for k, v in best.items() if not k.startswith("_") and k != "niche_id"},
        )
