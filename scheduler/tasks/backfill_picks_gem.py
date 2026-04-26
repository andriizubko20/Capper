"""
scheduler/tasks/backfill_picks_gem.py

Backfill Gem picks for already-played matches in a date range.
Useful to populate the mini-app with historical Gem performance immediately
without waiting for new matches.

Usage:
  python -m scheduler.tasks.backfill_picks_gem [--from 2026-04-24] [--to today]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from db.models import League as LeagueModel, Match, Prediction
from db.session import SessionLocal
from model.gem.features import expected_feature_names
from model.gem.niches import TARGET_LEAGUES
from model.gem.team_state import build_h2h, build_team_state
from scheduler.tasks.generate_picks_gem import (
    KELLY_CAP,
    KELLY_FRAC,
    LEAGUE_COUNTRY,
    MODEL_VERSION,
    _build_match_row,
    _compute_bankroll,
    _gem_pick_for_match,
    _latest_1x2_odds,
    _load_ensemble,
    _load_historical_for_state,
)


def run_backfill(date_from: datetime | str, date_to: datetime | str | None = None) -> None:
    if isinstance(date_from, str):
        date_from = datetime.strptime(date_from, "%Y-%m-%d")
    if date_to is None:
        date_to = datetime.utcnow()
    elif isinstance(date_to, str):
        date_to = datetime.strptime(date_to, "%Y-%m-%d")

    logger.info(f"[Gem backfill] {date_from.date()} → {date_to.date()}")

    ensemble, calibrator = _load_ensemble()
    feature_names = ensemble.feature_names or expected_feature_names()

    db = SessionLocal()
    try:
        # Pull all matches in range, regardless of status — backfill includes
        # finished too so picks are immediately settled.
        matches = db.query(Match).join(LeagueModel).filter(
            Match.date >= date_from,
            Match.date <= date_to,
            LeagueModel.name.in_(TARGET_LEAGUES),
        ).all()
        matches = [
            m for m in matches
            if m.league and (
                m.league.name == "Champions League"
                or m.league.country == LEAGUE_COUNTRY.get(m.league.name)
            )
        ]
        logger.info(f"[Gem backfill] Found {len(matches)} matches in window")

        # Build state from full history (finished + upcoming).
        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll(db, settings.bankroll)
        logger.info(f"[Gem backfill] Bankroll = ${bankroll:.0f}")

        match_ids = [m.id for m in matches]
        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        rows: list[np.ndarray] = []
        infos: list[dict] = []
        match_objs: list[Match] = []
        odds_cache: dict[int, tuple[float, float, float]] = {}

        skipped_exists = skipped_no_features = skipped_no_odds = 0
        for match in matches:
            if match.id in existing:
                skipped_exists += 1
                continue
            home_odds, draw_odds, away_odds = _latest_1x2_odds(db, match.id)
            if not (home_odds and draw_odds and away_odds):
                skipped_no_odds += 1
                continue
            built = _build_match_row(db, match, team_state, h2h_dict, feature_names)
            if built is None:
                skipped_no_features += 1
                continue
            X_row, info = built
            rows.append(X_row)
            infos.append(info)
            match_objs.append(match)
            odds_cache[match.id] = (home_odds, draw_odds, away_odds)

        if not rows:
            logger.info(
                f"[Gem backfill] No eligible rows. Skipped existing={skipped_exists}, "
                f"no_features={skipped_no_features}, no_odds={skipped_no_odds}"
            )
            return

        X = np.vstack(rows)
        info_df = pd.DataFrame(infos)
        proba_raw = ensemble.predict_proba_from_info(X, info_df)
        proba_cal = calibrator.transform(proba_raw)

        added = 0
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
            added += 1
            won = ""
            if match.home_score is not None and match.away_score is not None:
                w = (side == "home" and match.home_score > match.away_score) or \
                    (side == "away" and match.away_score > match.home_score)
                won = "WIN" if w else "LOSS"
            logger.info(
                f"[Gem backfill] {match.date.strftime('%m-%d')} {home} vs {away} → {side} "
                f"@ {odds_val:.2f} (p={p_side:.3f}, gem={decision['gem_score']:.3f}, "
                f"EV {ev*100:.1f}%) {won}"
            )

        db.commit()
        logger.info(
            f"[Gem backfill] Done. Added {added}, skipped: "
            f"existing={skipped_exists}, no_features={skipped_no_features}, no_odds={skipped_no_odds}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", type=str, default="2026-04-24")
    ap.add_argument("--to",   dest="d_to",   type=str, default=None)
    args = ap.parse_args()

    run_backfill(args.d_from, args.d_to)
