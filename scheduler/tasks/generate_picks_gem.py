"""
scheduler/tasks/generate_picks_gem.py

Gem model — 3-model XGBoost+LGB+CatBoost ensemble + L2 stacking + isotonic
calibration → niche-style filter (high-confidence "gems" with positive market gap).

For each upcoming match in target leagues:
  1. Build pre-match team_state (Glicko, xG splits, form, momentum, style, h2h, rest).
  2. Build the full Gem feature row (50+ features including league one-hots / priors).
  3. Predict calibrated [P(H), P(D), P(A)] via the saved ensemble + calibrator.
  4. Apply gem filter:
        P(draw) < MAX_DRAW_PROB
        P(side) > MIN_BET_PROB
        gem_score = our_p(side) - market_p(side) > MIN_GEM_SCORE
        MIN_ODDS <= odds(side) <= MAX_ODDS
  5. Stake: Kelly 25% × cap 10% (same as Pure).
  6. Save to predictions table with model_version='gem_v1' — picked up by API
     and shown in the mini-app (no Telegram broadcast).

Performance:
  - Ensemble (XGB + LGB + CatBoost + LR meta + isotonic) is loaded ONCE per
    process and cached as a module-level singleton (~1 s cold-start).
  - team_state / h2h are built once per task run from full historical (~2 s).
  - NO SStats API calls; uses only DB + saved Gem artifacts.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from db.models import League as LeagueModel, Match, Odds, Prediction, TeamRating
from db.session import SessionLocal
from model.gem.calibration import GemCalibrator
from model.gem.ensemble import GemEnsemble
from model.gem.features import build_gem_features, expected_feature_names, market_probs
from model.gem.niches import (
    FLAT_STAKE_FRAC,  # noqa: F401  (kept for parity with niches.py constants)
    MAX_DRAW_PROB,
    MAX_ODDS,
    MIN_BET_PROB,
    MIN_GEM_SCORE,
    MIN_ODDS,
    TARGET_LEAGUES,
)
from model.gem.team_state import build_h2h, build_team_state

MODEL_VERSION = "gem_v1"
KELLY_FRAC = 0.25
KELLY_CAP = 0.10

# Same allowlist as Pure — country guard against name collisions
# (e.g. German vs Austrian Bundesliga, English vs Ukrainian Premier League).
LEAGUE_COUNTRY: dict[str, str] = {
    "Premier League":     "England",
    "La Liga":            "Spain",
    "Bundesliga":         "Germany",
    "Serie A":            "Italy",
    "Serie B":            "Italy",
    "Ligue 1":            "France",
    "Primeira Liga":      "Portugal",
    "Eredivisie":         "Netherlands",
    "Jupiler Pro League": "Belgium",
    "Champions League":   "World",   # accept any country for UCL
}

ARTIFACTS = Path(__file__).parents[2] / "model" / "gem" / "artifacts"
PER_LEAGUE_PATH = ARTIFACTS / "per_league_thresholds.json"

# ── Module-level cached singletons ────────────────────────────────────────
_ENSEMBLE: GemEnsemble | None = None
_CALIBRATOR: GemCalibrator | None = None
_PER_LEAGUE: dict[str, dict] | None = None


def _load_per_league_thresholds() -> dict[str, dict]:
    """
    Load per-league optimal thresholds from JSON.
    If file missing, fall back to universal config from niches.py for ALL leagues.
    """
    global _PER_LEAGUE
    if _PER_LEAGUE is not None:
        return _PER_LEAGUE
    import json
    if PER_LEAGUE_PATH.exists():
        with open(PER_LEAGUE_PATH) as f:
            _PER_LEAGUE = json.load(f)
        logger.info(
            f"[Gem] Loaded per-league thresholds for {len(_PER_LEAGUE)} leagues"
        )
    else:
        logger.warning("[Gem] per_league_thresholds.json missing — using universal niches.py config")
        _PER_LEAGUE = {}
    return _PER_LEAGUE


def _thresholds_for(league: str) -> dict | None:
    """Return per-league thresholds, or None if league not whitelisted for Gem."""
    cfg = _load_per_league_thresholds()
    if not cfg:
        # Fall back to universal config (legacy mode)
        return {
            "max_draw_prob":  MAX_DRAW_PROB,
            "min_bet_prob":   MIN_BET_PROB,
            "min_gem_score":  MIN_GEM_SCORE,
        }
    return cfg.get(league)


def _load_ensemble() -> tuple[GemEnsemble, GemCalibrator]:
    """Load and cache the Gem ensemble + calibrator (~1 s cold start)."""
    global _ENSEMBLE, _CALIBRATOR
    if _ENSEMBLE is None:
        logger.info(f"[Gem] Loading ensemble from {ARTIFACTS}/ …")
        _ENSEMBLE = GemEnsemble.load(ARTIFACTS)
    if _CALIBRATOR is None:
        logger.info("[Gem] Loading calibrator …")
        _CALIBRATOR = GemCalibrator.load(ARTIFACTS)
    return _ENSEMBLE, _CALIBRATOR


# ── Historical loaders (DB-only, no SStats) ───────────────────────────────

def _load_historical_for_state(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all matches (finished + upcoming) + match_stats for team_state/h2h.

    Mirrors generate_picks_pure._load_historical_for_state.
    """
    from sqlalchemy import text
    matches = pd.DataFrame(db.execute(text(
        """
        SELECT m.id AS match_id, m.date, m.league_id, l.name AS league_name,
               m.home_team_id, m.away_team_id, m.home_score, m.away_score
        FROM matches m JOIN leagues l ON l.id = m.league_id
        ORDER BY m.date ASC
        """
    )).fetchall(), columns=[
        "match_id", "date", "league_id", "league_name",
        "home_team_id", "away_team_id", "home_score", "away_score",
    ])
    matches["date"] = pd.to_datetime(matches["date"])

    def _result(r):
        if r.home_score is None or r.away_score is None or pd.isna(r.home_score) or pd.isna(r.away_score):
            return None
        if r.home_score > r.away_score:
            return "H"
        if r.away_score > r.home_score:
            return "A"
        return "D"
    matches["result"] = matches.apply(_result, axis=1)

    stats = pd.DataFrame(db.execute(text(
        """
        SELECT s.match_id,
               s.home_xg, s.away_xg,
               s.home_possession, s.away_possession,
               s.home_shots_on_target, s.away_shots_on_target,
               s.home_passes_accurate, s.away_passes_accurate,
               s.home_passes_total, s.away_passes_total,
               s.home_glicko, s.away_glicko,
               s.home_win_prob, s.away_win_prob
        FROM match_stats s JOIN matches m ON m.id = s.match_id
        WHERE m.home_score IS NOT NULL
        """
    )).fetchall(), columns=[
        "match_id", "home_xg", "away_xg",
        "home_possession", "away_possession",
        "home_sot", "away_sot",
        "home_pass_acc", "away_pass_acc",
        "home_pass_total", "away_pass_total",
        "home_glicko", "away_glicko",
        "home_win_prob", "away_win_prob",
    ])
    return matches, stats


def _load_match_stats_row(db, match_id: int) -> dict | None:
    """Pre-match Glicko win probabilities from match_stats (one row per match)."""
    from sqlalchemy import text
    row = db.execute(text(
        "SELECT home_glicko, away_glicko, home_win_prob, away_win_prob "
        "FROM match_stats WHERE match_id = :mid"
    ), {"mid": match_id}).fetchone()
    if row is None:
        return None
    return {
        "home_glicko":  row.home_glicko,
        "away_glicko":  row.away_glicko,
        "home_win_prob": row.home_win_prob,
        "away_win_prob": row.away_win_prob,
    }


def _self_glicko(db, team_id: int):
    """Self-computed Glicko rating from team_ratings table. Returns ratings tuple
    (rating, rd, vol) or None.
    """
    row = db.query(TeamRating).filter_by(team_id=team_id).first()
    if row is None:
        return None
    return row


def _has_injury(db, match_id: int, team_id: int) -> bool:
    """True if team has any injury report for this match."""
    from sqlalchemy import text
    row = db.execute(text(
        "SELECT 1 FROM injury_reports WHERE match_id = :mid AND team_id = :tid LIMIT 1"
    ), {"mid": match_id, "tid": team_id}).fetchone()
    return row is not None


def _latest_1x2_odds(db, match_id: int) -> tuple[float | None, float | None, float | None]:
    """Latest 1x2 odds per outcome (first row per outcome assumed freshest)."""
    odds_rows = db.query(Odds).filter(
        Odds.match_id == match_id, Odds.market == "1x2",
    ).all()
    by_outcome: dict[str, float] = {}
    for o in odds_rows:
        if o.outcome not in by_outcome:
            by_outcome[o.outcome] = o.value
    return by_outcome.get("home"), by_outcome.get("draw"), by_outcome.get("away")


# ── Feature row construction for inference ───────────────────────────────

def _build_match_row(
    db,
    match: Match,
    team_state: dict,
    h2h_dict: dict,
    feature_names: list[str],
) -> tuple[np.ndarray, dict] | None:
    """
    Build a single (1, n_features) row for the Gem ensemble.

    Returns (X_row, info_row) or None if features cannot be built (skip match).

    Glicko priority: SELF (team_ratings) > SStats (match_stats). If neither is
    present, snapshot Glicko stays as None inside team_state and downstream
    feature row will have None for those columns; we still allow the row through
    because tree models handle NaN, but we DROP if rolling xG/PPG history is
    missing (those are core features).
    """
    h_state = team_state.get((match.id, "home"))
    a_state = team_state.get((match.id, "away"))
    if h_state is None or a_state is None:
        logger.debug(f"[Gem] No team state for match {match.id}, skip")
        return None

    if h_state.get("xg_for_10") is None or a_state.get("xg_for_10") is None:
        logger.debug(f"[Gem] No xG history for match {match.id}, skip")
        return None
    if h_state.get("ppg_10") is None or a_state.get("ppg_10") is None:
        logger.debug(f"[Gem] No PPG history for match {match.id}, skip")
        return None

    self_h = _self_glicko(db, match.home_team_id)
    self_a = _self_glicko(db, match.away_team_id)
    s_row = _load_match_stats_row(db, match.id)

    # Override snapshot glicko_now with self-Glicko if available (preferred:
    # deterministic & not subject to upstream API outages).
    if self_h is not None and self_a is not None:
        h_state = {**h_state, "glicko_now": self_h.rating}
        a_state = {**a_state, "glicko_now": self_a.rating}

    home_glicko_prob = (s_row or {}).get("home_win_prob")
    away_glicko_prob = (s_row or {}).get("away_win_prob")

    home_inj = _has_injury(db, match.id, match.home_team_id)
    away_inj = _has_injury(db, match.id, match.away_team_id)

    league_name = match.league.name if match.league else None
    feat = build_gem_features(
        match_date=pd.Timestamp(match.date),
        league_name=league_name,
        home_state=h_state,
        away_state=a_state,
        h2h=h2h_dict.get(match.id, {}),
        home_glicko_prob=home_glicko_prob,
        away_glicko_prob=away_glicko_prob,
        home_has_injuries=home_inj,
        away_has_injuries=away_inj,
        league_priors=None,  # injected by ensemble.predict_proba_from_info via final encoder
    )

    row = np.array([[feat.get(f) for f in feature_names]], dtype=float)
    info = {
        "match_id":    match.id,
        "date":        pd.Timestamp(match.date),
        "league_name": league_name,
    }
    return row, info


# ── Core decision logic ──────────────────────────────────────────────────

def _gem_pick_for_match(
    proba_cal: np.ndarray,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    league: str | None = None,
) -> dict | None:
    """
    Apply gem filter to calibrated [P_H, P_D, P_A]. Pick the better of HOME/AWAY
    if both pass the filter, scored by gem_score.

    Per-league thresholds are loaded from artifacts/per_league_thresholds.json.
    If `league` is not in the per-league config, the match is skipped (returns
    None) — leagues without a viable sweep combo are excluded from Gem.

    Returns dict {side, p, odds, market_p, gem_score, p_draw} or None.
    """
    thr = _thresholds_for(league) if league is not None else None
    if thr is None:
        return None
    max_draw_prob = thr["max_draw_prob"]
    min_bet_prob  = thr["min_bet_prob"]
    min_gem_score = thr["min_gem_score"]
    devig_method  = thr.get("devig", "proportional")
    # Per-league odds range (sweet-spot of model edge); fall back to global
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
    # If both sides pass the filter (rare), keep the one with higher gem_score.
    return max(candidates, key=lambda c: c["gem_score"])


def _compute_bankroll(db, initial: float) -> float:
    """Same routine as Pure: bankroll = initial + cumulative settled PnL."""
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


# ── Main entry point ─────────────────────────────────────────────────────

def run_generate_picks_gem(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(hours=settings.picks_hours_before + 0.5)

    logger.info(
        f"[Gem] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    ensemble, calibrator = _load_ensemble()
    feature_names = ensemble.feature_names or expected_feature_names()

    db = SessionLocal()
    try:
        upcoming = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status == "Not Started",
            LeagueModel.name.in_(TARGET_LEAGUES),
        ).all()
        upcoming = [
            m for m in upcoming
            if m.league and (
                m.league.name == "Champions League"
                or m.league.country == LEAGUE_COUNTRY.get(m.league.name)
            )
        ]
        if not upcoming:
            logger.info("[Gem] No matches in window, skipping")
            return

        logger.info(f"[Gem] Found {len(upcoming)} upcoming matches")

        logger.info("[Gem] Building team_state + h2h …")
        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll(db, settings.bankroll)
        logger.info(f"[Gem] Bankroll = ${bankroll:.0f}")

        match_ids = [m.id for m in upcoming]
        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        # Batch features → ensemble → calibrator (one matrix vs. per-match calls
        # is faster for tree models too).
        rows: list[np.ndarray] = []
        infos: list[dict] = []
        match_objs: list[Match] = []
        odds_cache: dict[int, tuple[float, float, float]] = {}

        for match in upcoming:
            if match.id in existing:
                continue
            home_odds, draw_odds, away_odds = _latest_1x2_odds(db, match.id)
            if not (home_odds and draw_odds and away_odds):
                logger.debug(f"[Gem] No 1x2 odds for match {match.id}, skip")
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
            logger.info("[Gem] No matches with full features in window")
            return

        X = np.vstack(rows)
        info_df = pd.DataFrame(infos)
        proba_raw = ensemble.predict_proba_from_info(X, info_df)
        proba_cal = calibrator.transform(proba_raw)

        new_picks = 0
        for i, match in enumerate(match_objs):
            home_odds, draw_odds, away_odds = odds_cache[match.id]
            league_name = match.league.name if match.league else None
            decision = _gem_pick_for_match(
                proba_cal[i], home_odds, draw_odds, away_odds, league=league_name,
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
                f"[Gem] Pick: {home} vs {away} → {side} | odds={odds_val:.2f} "
                f"p={p_side:.3f} mp={decision['market_p']:.3f} "
                f"gem={decision['gem_score']:.3f} p_draw={decision['p_draw']:.3f} "
                f"EV={ev*100:.1f}%"
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
                league_name=league_name,
                home_name=home,
                away_name=away,
                match_date=match.date,
            ))
            new_picks += 1

        db.commit()
        logger.info(f"[Gem] Generated {new_picks} picks")

    finally:
        db.close()


if __name__ == "__main__":
    run_generate_picks_gem()
