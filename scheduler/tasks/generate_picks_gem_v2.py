"""
scheduler/tasks/generate_picks_gem_v2.py

A/B variant of Gem ML pick generator. Same XGB+LGB+CatBoost ensemble as
gem_v1 but uses experimental calibrator (kmeans-3 clustering, mpl=20,
tail=0.10 — Grid #2 winner). Backtested at +52.49 u/yr vs prod V1 +49.16.

Picks saved with model_version='gem_v2_kmeans3'. Mini-app shows them
side-by-side with gem_v1 in the Compare screen — 30-day live shoot-out
decides which calibrator wins.

Architecture: imports nearly everything from generate_picks_gem; only
overrides:
  - calibrator artifact (calibrator_v2_kmeans3.pkl)
  - per-league thresholds path (per_league_thresholds_v2.json)
  - MODEL_VERSION ('gem_v2_kmeans3')
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import tuple_

from config.settings import settings
from data.best_odds import best_1x2_odds  # noqa: F401  (re-exported via gem_v1 shared)
from db.models import League as LeagueModel, Match, Prediction
from db.session import SessionLocal
from model.gem.calibration import GemCalibrator
from model.gem.ensemble import GemEnsemble
from model.gem.features import expected_feature_names, market_probs
from model.gem.niches import (
    MAX_DRAW_PROB,  # global fallback constants
    MAX_ODDS,
    MIN_BET_PROB,
    MIN_GEM_SCORE,
    MIN_ODDS,
    TARGET_LEAGUES,
    to_canonical,
)
from model.gem.team_state import build_h2h, build_team_state
from scheduler.tasks.generate_picks_gem import (
    KELLY_CAP,
    KELLY_FRAC,
    _build_match_row,
    _compute_bankroll as _v1_compute_bankroll,
    _latest_1x2_odds,
    _load_historical_for_state,
)

MODEL_VERSION = "gem_v2_kmeans3"

ARTIFACTS = Path(__file__).parents[2] / "model" / "gem" / "artifacts"
PER_LEAGUE_PATH_V2 = ARTIFACTS / "per_league_thresholds_v2.json"
CALIBRATOR_V2_PATH = ARTIFACTS / "calibrator_v2_kmeans3.pkl"

# Module singletons (separate from gem_v1)
_ENSEMBLE: GemEnsemble | None = None
_CALIBRATOR_V2: GemCalibrator | None = None
_PER_LEAGUE_V2: dict[str, dict] | None = None


def _load_per_league_v2() -> dict[str, dict]:
    global _PER_LEAGUE_V2
    if _PER_LEAGUE_V2 is not None:
        return _PER_LEAGUE_V2
    import json
    if PER_LEAGUE_PATH_V2.exists():
        with open(PER_LEAGUE_PATH_V2) as f:
            _PER_LEAGUE_V2 = json.load(f)
        logger.info(f"[Gem v2] Loaded per-league thresholds for {len(_PER_LEAGUE_V2)} leagues")
    else:
        logger.warning("[Gem v2] per_league_thresholds_v2.json missing — using universal niches.py config")
        _PER_LEAGUE_V2 = {}
    return _PER_LEAGUE_V2


def _thresholds_v2(league_canonical: str) -> dict | None:
    cfg = _load_per_league_v2()
    if not cfg:
        return {
            "max_draw_prob":  MAX_DRAW_PROB,
            "min_bet_prob":   MIN_BET_PROB,
            "min_gem_score":  MIN_GEM_SCORE,
        }
    return cfg.get(league_canonical)


def _load_ensemble_and_v2_cal() -> tuple[GemEnsemble, GemCalibrator]:
    global _ENSEMBLE, _CALIBRATOR_V2
    if _ENSEMBLE is None:
        logger.info(f"[Gem v2] Loading ensemble from {ARTIFACTS}/ …")
        _ENSEMBLE = GemEnsemble.load(ARTIFACTS)
    if _CALIBRATOR_V2 is None:
        logger.info(f"[Gem v2] Loading V2 calibrator from {CALIBRATOR_V2_PATH}")
        if not CALIBRATOR_V2_PATH.exists():
            raise FileNotFoundError(
                f"V2 calibrator missing at {CALIBRATOR_V2_PATH} — retrain Gem with new train.py"
            )
        _CALIBRATOR_V2 = joblib.load(CALIBRATOR_V2_PATH)
    return _ENSEMBLE, _CALIBRATOR_V2


def _gem_v2_pick_for_match(
    proba_cal: np.ndarray,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    league: str | None = None,
) -> dict | None:
    """Same filter logic as gem_v1 but reads thresholds from v2 JSON."""
    thr = _thresholds_v2(league) if league is not None else None
    if thr is None:
        return None
    max_draw_prob = thr["max_draw_prob"]
    min_bet_prob  = thr["min_bet_prob"]
    min_gem_score = thr["min_gem_score"]
    devig_method  = thr.get("devig", "proportional")
    league_min_odds = thr.get("min_odds", MIN_ODDS)
    league_max_odds = thr.get("max_odds", MAX_ODDS)

    p_h, p_d, p_a = float(proba_cal[0]), float(proba_cal[1]), float(proba_cal[2])
    if p_d >= max_draw_prob:
        return None

    market = market_probs(home_odds, draw_odds, away_odds, method=devig_method)
    if market["home"] is None:
        return None

    candidates: list[dict] = []
    for side, p_side, odds_side, mp in (
        ("home", p_h, home_odds, market["home"]),
        ("away", p_a, away_odds, market["away"]),
    ):
        if p_side <= min_bet_prob:
            continue
        if not (league_min_odds <= odds_side <= league_max_odds):
            continue
        gem_score = p_side - mp
        if gem_score <= min_gem_score:
            continue
        candidates.append({
            "side":      side,
            "p":         p_side,
            "odds":      odds_side,
            "market_p":  mp,
            "gem_score": gem_score,
            "p_draw":    p_d,
        })
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["gem_score"])


def _compute_bankroll_v2(db, initial: float) -> float:
    """Bankroll for gem_v2_kmeans3 (separate ledger)."""
    FINISHED = {"Finished", "FT", "finished", "ft", "Match Finished"}
    preds = (
        db.query(Prediction)
        .join(Match)
        .filter(
            Prediction.model_version == MODEL_VERSION,
            Match.status.in_(FINISHED),
            Match.home_score.isnot(None),
        )
        .order_by(Match.date.asc())
        .all()
    )
    bankroll = initial
    for pred in preds:
        m = pred.match
        kf = pred.kelly_fraction or 0
        stake = min(bankroll * kf, bankroll * KELLY_CAP)
        won = (pred.outcome == "home" and m.home_score > m.away_score) or \
              (pred.outcome == "away" and m.away_score > m.home_score)
        bankroll += stake * (pred.odds_used - 1) if won else -stake
    return round(bankroll, 2)


def run_generate_picks_gem_v2(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(days=settings.early_picks_days_ahead)

    logger.info(
        f"[Gem v2] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    ensemble, calibrator_v2 = _load_ensemble_and_v2_cal()
    feature_names = ensemble.feature_names or expected_feature_names()

    db = SessionLocal()
    try:
        upcoming = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status == "Not Started",
            tuple_(LeagueModel.name, LeagueModel.country).in_(list(TARGET_LEAGUES)),
        ).all()
        if not upcoming:
            logger.info("[Gem v2] No matches in window, skipping")
            return

        logger.info(f"[Gem v2] Found {len(upcoming)} upcoming matches")
        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll_v2(db, settings.bankroll)
        logger.info(f"[Gem v2] Bankroll = ${bankroll:.0f}")

        match_ids = [m.id for m in upcoming]
        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        rows: list[np.ndarray] = []
        infos: list[dict] = []
        match_objs: list[Match] = []
        odds_cache: dict[int, tuple[float, float, float]] = {}

        for match in upcoming:
            if match.id in existing:
                continue
            home_odds, draw_odds, away_odds = _latest_1x2_odds(db, match.id)
            if not (home_odds and draw_odds and away_odds):
                continue
            built = _build_match_row(db, match, team_state, h2h_dict, feature_names)
            if built is None:
                continue
            X_row, info = built
            rows.append(X_row)
            infos.append(info)
            match_objs.append(match)
            odds_cache[match.id] = (home_odds, draw_odds, away_odds)

        if not rows:
            logger.info("[Gem v2] No matches with full features in window")
            return

        X = np.vstack(rows)
        info_df = pd.DataFrame(infos)
        proba_raw = ensemble.predict_proba_from_info(X, info_df)
        proba_cal = calibrator_v2.transform(
            proba_raw, leagues=info_df["league_name"].to_numpy()
        )

        new_picks = 0
        for i, match in enumerate(match_objs):
            home_odds, draw_odds, away_odds = odds_cache[match.id]
            league_canonical = (
                to_canonical(match.league.name, match.league.country) if match.league else None
            )
            decision = _gem_v2_pick_for_match(
                proba_cal[i], home_odds, draw_odds, away_odds, league=league_canonical,
            )
            if decision is None:
                continue
            side = decision["side"]
            p_side = decision["p"]
            odds_val = decision["odds"]
            ev = round(p_side * odds_val - 1, 4)
            if ev <= 0:
                continue
            b = odds_val - 1
            f_star = max(0.0, (p_side * b - (1 - p_side)) / b) if b > 0 else 0.0
            if f_star <= 0:
                continue
            kelly = KELLY_FRAC * f_star
            stake = round(min(bankroll * kelly, bankroll * KELLY_CAP), 2)

            home = match.home_team.name if match.home_team else "?"
            away = match.away_team.name if match.away_team else "?"
            logger.info(
                f"[Gem v2] Pick: {home} vs {away} → {side} | odds={odds_val:.2f} "
                f"p={p_side:.3f} gem={decision['gem_score']:.3f} EV={ev*100:.1f}%"
            )
            db.add(Prediction(
                match_id=match.id, market="1x2",
                outcome=side,
                probability=round(p_side, 4),
                odds_used=float(odds_val),
                ev=ev,
                kelly_fraction=round(kelly, 4),
                stake=stake,
                model_version=MODEL_VERSION,
                league_name=match.league.name if match.league else None,
                home_name=home,
                away_name=away,
                match_date=match.date,
            ))
            new_picks += 1

        db.commit()
        logger.info(f"[Gem v2] Generated {new_picks} picks")
    finally:
        db.close()


if __name__ == "__main__":
    run_generate_picks_gem_v2()
