"""
scheduler/tasks/generate_picks_pure.py

Pure model — niche-based picks for upcoming matches.

For each upcoming match in target leagues:
  1. Build pre-match team_state (Glicko, xG splits, form, momentum, style, h2h, rest).
  2. Compute side features (home + away perspective).
  3. For each side: test against all niches → pick the one with highest p_is.
  4. Compute EV = p_is * odds - 1, Kelly stake (25% × cap 10%).
  5. Save to predictions table with model_version='pure_v1' — picked up by API
     and shown in the mini-app (no Telegram broadcast).
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from db.models import League as LeagueModel, Match, Odds, Prediction
from db.session import SessionLocal
from model.gem.team_state import build_h2h, build_team_state

MODEL_VERSION = "pure_v1"
KELLY_FRAC = 0.25
KELLY_CAP = 0.10

# API-Football league IDs to avoid name collisions (e.g. German Bundesliga
# vs Austrian Bundesliga — both stored as "Bundesliga" in DB).
LEAGUE_API_IDS = {
    39,   # Premier League (England)
    78,   # Bundesliga (Germany)
    135,  # Serie A (Italy)
    140,  # La Liga (Spain)
    61,   # Ligue 1 (France)
    94,   # Primeira Liga (Portugal)
    136,  # Serie B (Italy)
    88,   # Eredivisie (Netherlands)
    144,  # Jupiler Pro League (Belgium)
    2,    # Champions League
}

ARTIFACTS = Path(__file__).parents[2] / "model" / "pure" / "artifacts"


def _load_niches() -> dict[str, list[dict]]:
    with open(ARTIFACTS / "selected_niches_with_pis.json") as f:
        return json.load(f)


def _load_historical_for_state(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all matches (finished + upcoming) + stats for team_state/h2h building.

    team_state takes pre-match snapshots BEFORE updating history; for upcoming
    matches we still want a snapshot so the predictor can read it. _update_history
    skips when scores are NULL.
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
    from sqlalchemy import text
    row = db.execute(text(
        "SELECT home_glicko, away_glicko, home_win_prob, away_win_prob FROM match_stats WHERE match_id = :mid"
    ), {"mid": match_id}).fetchone()
    if row is None:
        return None
    return {
        "home_glicko": row.home_glicko,
        "away_glicko": row.away_glicko,
        "home_win_prob": row.home_win_prob,
        "away_win_prob": row.away_win_prob,
    }


def _build_match_features(db, match: Match, team_state: dict, h2h_dict: dict) -> tuple[dict, dict] | None:
    """Build (home_side_features, away_side_features) for a single match."""
    h_state = team_state.get((match.id, "home"))
    a_state = team_state.get((match.id, "away"))

    # If team_state lacks this match (it shouldn't if it's pre-match snapshot),
    # try team_state by team_id from the latest snapshot built incrementally
    if h_state is None or a_state is None:
        logger.debug(f"[Pure] No team state for match {match.id}, skip")
        return None

    if h_state.get("glicko_now") is None or a_state.get("glicko_now") is None:
        logger.debug(f"[Pure] No Glicko in team state for match {match.id}, skip")
        return None
    if h_state.get("xg_for_10") is None or a_state.get("xg_for_10") is None:
        logger.debug(f"[Pure] No xG in team state for match {match.id}, skip")
        return None
    if h_state.get("ppg_10") is None or a_state.get("ppg_10") is None:
        return None

    # Match stats from DB (may be None for upcoming matches)
    s_row = _load_match_stats_row(db, match.id)
    home_glicko_prob = (s_row or {}).get("home_win_prob")
    away_glicko_prob = (s_row or {}).get("away_win_prob")

    # Odds — from latest snapshot
    odds_rows = db.query(Odds).filter(
        Odds.match_id == match.id, Odds.market == "1x2",
    ).all()
    by_outcome: dict[str, float] = {}
    for o in odds_rows:
        # Take freshest (highest recorded_at) per outcome
        if o.outcome not in by_outcome:
            by_outcome[o.outcome] = o.value
    home_odds = by_outcome.get("home")
    draw_odds = by_outcome.get("draw")
    away_odds = by_outcome.get("away")
    if not (home_odds and draw_odds and away_odds):
        logger.debug(f"[Pure] No 1x2 odds for match {match.id}, skip")
        return None

    raw_sum = 1 / home_odds + 1 / draw_odds + 1 / away_odds
    home_market_prob = (1 / home_odds) / raw_sum
    away_market_prob = (1 / away_odds) / raw_sum

    h2h = h2h_dict.get(match.id, {})
    h2h_home_wr = h2h.get("h2h_home_wr")

    # Rest
    match_dt = pd.Timestamp(match.date)
    home_rest = (match_dt - h_state["last_match_date"]).days if h_state.get("last_match_date") else None
    away_rest = (match_dt - a_state["last_match_date"]).days if a_state.get("last_match_date") else None
    rest_advantage_h = (home_rest - away_rest) if home_rest is not None and away_rest is not None else None
    rest_advantage_a = -rest_advantage_h if rest_advantage_h is not None else None

    glicko_gap = h_state["glicko_now"] - a_state["glicko_now"]
    form_advantage = h_state["ppg_10"] - a_state["ppg_10"]

    home_features = {
        "odds":            home_odds,
        "glicko_gap":      glicko_gap,
        "glicko_prob":     home_glicko_prob,
        "market_prob":     home_market_prob,
        "xg_diff":         h_state["xg_for_10"] - h_state["xg_against_10"],
        "attack_vs_def":   h_state["xg_for_10"] - a_state["xg_against_10"],
        "form_advantage":  form_advantage,
        "ppg":             h_state["ppg_10"],
        "xg_trend":        h_state.get("xg_trend"),
        "glicko_momentum": h_state.get("glicko_momentum"),
        "win_streak":      h_state.get("win_streak"),
        "opp_lose_streak": a_state.get("lose_streak"),
        "possession_10":   h_state.get("possession_10"),
        "sot_10":          h_state.get("sot_10"),
        "pass_acc_10":     h_state.get("pass_acc_10"),
        "rest_advantage":  rest_advantage_h,
        "h2h_wr":          h2h_home_wr,
    }
    away_features = {
        "odds":            away_odds,
        "glicko_gap":      -glicko_gap,
        "glicko_prob":     away_glicko_prob,
        "market_prob":     away_market_prob,
        "xg_diff":         a_state["xg_for_10"] - a_state["xg_against_10"],
        "attack_vs_def":   a_state["xg_for_10"] - h_state["xg_against_10"],
        "form_advantage":  -form_advantage,
        "ppg":             a_state["ppg_10"],
        "xg_trend":        a_state.get("xg_trend"),
        "glicko_momentum": a_state.get("glicko_momentum"),
        "win_streak":      a_state.get("win_streak"),
        "opp_lose_streak": h_state.get("lose_streak"),
        "possession_10":   a_state.get("possession_10"),
        "sot_10":          a_state.get("sot_10"),
        "pass_acc_10":     a_state.get("pass_acc_10"),
        "rest_advantage":  rest_advantage_a,
        "h2h_wr":          (1 - h2h_home_wr) if h2h_home_wr is not None else None,
    }
    return home_features, away_features


def _matches_niche(features: dict, niche: dict) -> bool:
    odds = features.get("odds")
    if odds is None:
        return False
    lo, hi = niche["odds_range"]
    if not (lo <= odds <= hi):
        return False

    checks = [
        ("glicko_gap",       "min_glicko_gap",      ">="),
        ("glicko_prob",      "min_glicko_prob",     ">="),
        ("xg_diff",          "min_xg_diff",         ">="),
        ("attack_vs_def",    "min_attack_vs_def",   ">="),
        ("form_advantage",   "min_form_advantage",  ">="),
        ("ppg",              "min_ppg",             ">="),
        ("xg_trend",         "min_xg_trend",        ">="),
        ("glicko_momentum",  "min_glicko_momentum", ">="),
        ("win_streak",       "min_win_streak",      ">="),
        ("opp_lose_streak",  "min_opp_lose_streak", ">="),
        ("possession_10",    "min_possession_10",   ">="),
        ("sot_10",           "min_sot_10",          ">="),
        ("pass_acc_10",      "min_pass_acc_10",     ">="),
        ("rest_advantage",   "min_rest_advantage",  ">="),
        ("h2h_wr",           "min_h2h_wr",          ">="),
        ("market_prob",      "max_market_prob",     "<="),
    ]
    for col, key, op in checks:
        thr = niche.get(key)
        if thr is None:
            continue
        v = features.get(col)
        if v is None:
            return False
        if op == ">=" and v < thr:
            return False
        if op == "<=" and v > thr:
            return False
    return True


def _compute_bankroll(db, initial: float) -> float:
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


def run_generate_picks_pure(
    match_date_from: datetime | None = None,
    match_date_to: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if match_date_from is None:
        match_date_from = now + timedelta(hours=settings.picks_hours_before - 0.5)
    if match_date_to is None:
        match_date_to = now + timedelta(hours=settings.picks_hours_before + 0.5)

    logger.info(
        f"[Pure] Generating picks | window: "
        f"{match_date_from.strftime('%d.%m %H:%M')} – {match_date_to.strftime('%d.%m %H:%M')} UTC"
    )

    niches_by_league = _load_niches()
    pure_leagues = set(niches_by_league.keys())

    db = SessionLocal()
    try:
        upcoming = db.query(Match).join(LeagueModel).filter(
            Match.date >= match_date_from.replace(tzinfo=None),
            Match.date <= match_date_to.replace(tzinfo=None),
            Match.status == "Not Started",
            LeagueModel.name.in_(pure_leagues),
            LeagueModel.api_id.in_(LEAGUE_API_IDS),
        ).all()

        if not upcoming:
            logger.info("[Pure] No matches in window, skipping")
            return

        logger.info(f"[Pure] Found {len(upcoming)} upcoming matches")

        # Build team_state/h2h once from full historical
        logger.info("[Pure] Building team_state + h2h …")
        hist_matches, hist_stats = _load_historical_for_state(db)
        team_state = build_team_state(hist_matches, hist_stats)
        h2h_dict = build_h2h(hist_matches)

        bankroll = _compute_bankroll(db, settings.bankroll)
        logger.info(f"[Pure] Bankroll = ${bankroll:.0f}")

        # Existing picks dedup
        match_ids = [m.id for m in upcoming]
        existing = {p.match_id for p in db.query(Prediction.match_id).filter(
            Prediction.match_id.in_(match_ids),
            Prediction.model_version == MODEL_VERSION,
        ).all()}

        new_picks = []

        for match in upcoming:
            if match.id in existing:
                continue
            league_name = match.league.name if match.league else None
            if league_name not in pure_leagues:
                continue
            league_niches = niches_by_league[league_name]

            features_pair = _build_match_features(db, match, team_state, h2h_dict)
            if features_pair is None:
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
                        # Prefer highest p_is (most confident)
                        if best is None or niche["p_is"] > best["niche"]["p_is"]:
                            best = {
                                "side": side, "niche": niche,
                                "odds": feats["odds"],
                            }

            if best is None:
                continue

            niche = best["niche"]
            side = best["side"]
            odds_val = best["odds"]
            p_is = niche["p_is"]
            ev = round(p_is * odds_val - 1, 4)
            if ev <= 0:
                logger.debug(f"[Pure] Negative EV for match {match.id}, niche={niche['niche_id']}, skip")
                continue

            b = odds_val - 1
            f_star = max(0.0, (p_is * b - (1 - p_is)) / b) if b > 0 else 0.0
            if f_star <= 0:
                continue

            kelly = KELLY_FRAC * f_star
            stake = round(min(bankroll * kelly, bankroll * KELLY_CAP), 2)

            home = match.home_team.name if match.home_team else "?"
            away = match.away_team.name if match.away_team else "?"
            logger.info(
                f"[Pure] Pick: {home} vs {away} → {side} | niche={niche['niche_id']} "
                f"odds={odds_val:.2f} p_is={p_is:.3f} f*={f_star:.3f} EV={ev*100:.1f}%"
            )

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

            new_picks.append((match, {
                "outcome": side, "odds": odds_val, "ev": ev,
                "p_is": p_is, "niche": niche["niche_id"],
                "stake": stake,
            }))

        db.commit()
        logger.info(f"[Pure] Generated {len(new_picks)} picks")

    finally:
        db.close()


if __name__ == "__main__":
    run_generate_picks_pure()
