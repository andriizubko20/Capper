"""
scheduler/tasks/backfill_picks_gem_v2.py

Backfill historical Gem v2 picks (kmeans-3 calibrator A/B variant).
Mirrors backfill_picks_gem.py but uses V2 calibrator + V2 thresholds.

Usage:
  python -m scheduler.tasks.backfill_picks_gem_v2 [--from 2026-04-21] [--to today]
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import tuple_

from config.settings import settings
from db.models import League as LeagueModel, Match, Prediction
from db.session import SessionLocal
from model.gem.features import expected_feature_names
from model.gem.niches import TARGET_LEAGUES, to_canonical
from model.gem.team_state import build_h2h, build_team_state
from scheduler.tasks.generate_picks_gem import (
    _build_match_row,
    _latest_1x2_odds,
    _load_historical_for_state,
)
from scheduler.tasks.generate_picks_gem_v2 import (
    KELLY_CAP,
    KELLY_FRAC,
    MODEL_VERSION,
    _compute_bankroll_v2,
    _gem_v2_pick_for_match,
    _load_ensemble_and_v2_cal,
)


def run_backfill(date_from: datetime | str, date_to: datetime | str | None = None) -> None:
    if isinstance(date_from, str):
        date_from = datetime.strptime(date_from, "%Y-%m-%d")
    if date_to is None:
        date_to = datetime.utcnow()
    elif isinstance(date_to, str):
        date_to = datetime.strptime(date_to, "%Y-%m-%d")

    logger.info(f"[Gem v2 backfill] {date_from.date()} → {date_to.date()}")

    ensemble, calibrator = _load_ensemble_and_v2_cal()
    feature_names = ensemble.feature_names or expected_feature_names()

    db = SessionLocal()
    try:
        matches = db.query(Match).join(LeagueModel).filter(
            Match.date >= date_from,
            Match.date <= date_to,
            tuple_(LeagueModel.name, LeagueModel.country).in_(list(TARGET_LEAGUES)),
        ).all()
        logger.info(f"[Gem v2 backfill] Found {len(matches)} matches in window")

        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll_v2(db, settings.bankroll)
        logger.info(f"[Gem v2 backfill] Bankroll = ${bankroll:.0f}")

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
                f"[Gem v2 backfill] No eligible rows. Skipped existing={skipped_exists}, "
                f"no_features={skipped_no_features}, no_odds={skipped_no_odds}"
            )
            return

        X = np.vstack(rows)
        info_df = pd.DataFrame(infos)
        proba_raw = ensemble.predict_proba_from_info(X, info_df)
        proba_cal = calibrator.transform(
            proba_raw, leagues=info_df["league_name"].to_numpy()
        )

        added = 0
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
            added += 1
            won = ""
            if match.home_score is not None and match.away_score is not None:
                w = (side == "home" and match.home_score > match.away_score) or \
                    (side == "away" and match.away_score > match.home_score)
                won = "WIN" if w else "LOSS"
            logger.info(
                f"[Gem v2 backfill] {match.date.strftime('%m-%d')} {home} vs {away} → {side} "
                f"@ {odds_val:.2f} (p={p_side:.3f}, gem={decision['gem_score']:.3f}, "
                f"EV {ev*100:.1f}%) {won}"
            )

        db.commit()
        logger.info(
            f"[Gem v2 backfill] Done. Added {added}, skipped: "
            f"existing={skipped_exists}, no_features={skipped_no_features}, no_odds={skipped_no_odds}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", type=str, default="2026-04-21")
    ap.add_argument("--to",   dest="d_to",   type=str, default=None)
    args = ap.parse_args()

    run_backfill(args.d_from, args.d_to)
