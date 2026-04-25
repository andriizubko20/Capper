"""
scheduler/tasks/backfill_picks_pure.py

Backfill Pure picks for already-played matches in a date range.
Useful to populate the mini-app with historical Pure performance immediately
without waiting for new matches.

Usage:
  python -m scheduler.tasks.backfill_picks_pure [--from 2026-04-24] [--to today]
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from db.models import League as LeagueModel, Match, Odds, Prediction
from db.session import SessionLocal
from model.gem.team_state import build_h2h, build_team_state
from scheduler.tasks.generate_picks_pure import (
    LEAGUE_API_IDS, MODEL_VERSION, KELLY_FRAC, KELLY_CAP,
    _load_historical_for_state, _build_match_features, _matches_niche, _load_niches,
    _compute_bankroll,
)

from config.settings import settings


def run_backfill(date_from: datetime, date_to: datetime) -> None:
    logger.info(f"[Pure backfill] {date_from.date()} → {date_to.date()}")

    niches_by_league = _load_niches()
    pure_leagues = set(niches_by_league.keys())

    db = SessionLocal()
    try:
        # Pull all matches in range, regardless of status — backfill includes
        # finished too so picks are immediately settled.
        matches = db.query(Match).join(LeagueModel).filter(
            Match.date >= date_from,
            Match.date <= date_to,
            LeagueModel.name.in_(pure_leagues),
            LeagueModel.api_id.in_(LEAGUE_API_IDS),
        ).all()
        logger.info(f"[Pure backfill] Found {len(matches)} matches in window")

        # Build state from full history (finished + upcoming)
        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll(db, settings.bankroll)
        logger.info(f"[Pure backfill] Bankroll = ${bankroll:.0f}")

        match_ids = [m.id for m in matches]
        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        added = skipped_exists = skipped_no_features = 0
        for match in matches:
            if match.id in existing:
                skipped_exists += 1
                continue

            league_name = match.league.name if match.league else None
            if league_name not in pure_leagues:
                continue
            league_niches = niches_by_league[league_name]

            features_pair = _build_match_features(db, match, team_state, h2h_dict)
            if features_pair is None:
                skipped_no_features += 1
                continue
            home_features, away_features = features_pair

            best = None
            for side, feats in (("home", home_features), ("away", away_features)):
                for niche in league_niches:
                    if niche["side"] != side:
                        continue
                    if niche.get("p_is") is None or niche["n"] < 15:
                        continue
                    if _matches_niche(feats, niche):
                        if best is None or niche["p_is"] > best["niche"]["p_is"]:
                            best = {"side": side, "niche": niche, "odds": feats["odds"]}

            if best is None:
                continue

            niche = best["niche"]
            side = best["side"]
            odds_val = best["odds"]
            p_is = niche["p_is"]
            ev = round(p_is * odds_val - 1, 4)
            if ev <= 0:
                continue

            b = odds_val - 1
            f_star = max(0.0, (p_is * b - (1 - p_is)) / b) if b > 0 else 0.0
            if f_star <= 0:
                continue
            kelly = KELLY_FRAC * f_star
            stake = round(min(bankroll * kelly, bankroll * KELLY_CAP), 2)

            home = match.home_team.name if match.home_team else "?"
            away = match.away_team.name if match.away_team else "?"

            db.add(Prediction(
                match_id=match.id, market="1x2",
                outcome=side,
                probability=round(p_is, 4),
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
                won = "✅ WIN" if w else "❌ LOSS"
            logger.info(
                f"[Pure backfill] {match.date.strftime('%m-%d')} {home} vs {away} → {side} "
                f"@ {odds_val:.2f} (EV {ev*100:.1f}%) {won}"
            )

        db.commit()
        logger.info(
            f"[Pure backfill] Done. Added {added}, "
            f"skipped {skipped_exists} (existing), {skipped_no_features} (no features)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", type=str, default="2026-04-24")
    ap.add_argument("--to",   dest="d_to",   type=str, default=None)
    args = ap.parse_args()

    date_from = datetime.strptime(args.d_from, "%Y-%m-%d")
    date_to = datetime.strptime(args.d_to, "%Y-%m-%d") if args.d_to else datetime.utcnow()
    run_backfill(date_from, date_to)
