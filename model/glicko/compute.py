"""
model/glicko/compute.py

Iterate ALL finished matches in DB chronologically and compute Glicko-2
rating for every team after every match.

Output: dict[(match_id, team_id, side)] = TeamRating  (PRE-MATCH snapshot)
        dict[team_id]                    = TeamRating  (current state)

Storage:
  - In-memory dict on each rebuild (fast)
  - Persist current ratings to `team_ratings` table (so live picks read from DB)
  - PRE-match snapshot per (match, side) NOT stored — recomputed when needed
    (negligible cost vs full recompute)

Rating period: each match is its own period with one game.
Football outcome → score:
  Home wins → home=1.0, away=0.0
  Draw      → both = 0.5
  Away wins → home=0.0, away=1.0

Why: simpler than weekly batches, more responsive to recent form.
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text

from db.session import SessionLocal
from model.glicko.algorithm import TeamRating, update_rating, football_score


def load_finished_matches() -> pd.DataFrame:
    """Load all finished matches in chronological order."""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            """
            SELECT id AS match_id, date, league_id, home_team_id, away_team_id,
                   home_score, away_score
            FROM matches
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
              AND status IN ('Finished','FT','finished','ft','Match Finished')
            ORDER BY date ASC, id ASC
            """
        )).fetchall()
    finally:
        db.close()
    df = pd.DataFrame(rows, columns=[
        "match_id", "date", "league_id", "home_team_id", "away_team_id",
        "home_score", "away_score",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_ratings(
    matches: pd.DataFrame | None = None,
    tau: float = 0.5,
) -> tuple[dict[tuple[int, int, str], TeamRating], dict[int, TeamRating]]:
    """
    Iterate matches chronologically. For each match snapshot PRE-match rating
    of both teams, then update both ratings with the result.

    Returns:
        snapshots:    dict[(match_id, team_id, side)] = TeamRating  (PRE-match)
        current:      dict[team_id]                   = TeamRating  (after last)
    """
    if matches is None:
        matches = load_finished_matches()

    state: dict[int, TeamRating] = defaultdict(lambda: TeamRating())
    snapshots: dict[tuple[int, int, str], TeamRating] = {}

    for row in matches.itertuples(index=False):
        h_id, a_id = int(row.home_team_id), int(row.away_team_id)
        h_state = state[h_id]
        a_state = state[a_id]

        snapshots[(row.match_id, h_id, "home")] = h_state
        snapshots[(row.match_id, a_id, "away")] = a_state

        h_score = football_score(row.home_score, row.away_score, "home")
        a_score = football_score(row.home_score, row.away_score, "away")

        new_h = update_rating(h_state, [(a_state, h_score)], tau=tau)
        new_a = update_rating(a_state, [(h_state, a_score)], tau=tau)

        state[h_id] = new_h
        state[a_id] = new_a

    logger.info(
        f"Glicko-2 computed: {len(snapshots):,} (match,side) snapshots "
        f"over {len(state):,} teams"
    )
    return snapshots, dict(state)


def expected_home_win_prob(
    home: TeamRating, away: TeamRating, draw_pct: float = 0.25
) -> tuple[float, float, float]:
    """
    Convert two Glicko ratings → 3-way (H/D/A) probabilities.

    Glicko gives P(home beats away) ignoring draws. We split that into
    H/D/A using a fixed draw rate (default 25%, typical football).
    """
    from model.glicko.algorithm import expected_score
    p_home_vs_away = expected_score(home, away)  # P(home wins or draws-as-half)
    # Adjust: convert Glicko expected score to W/D/L assuming draw_pct draws
    # E_home = P(home_win) + 0.5 * P(draw)
    # P(home_win) + P(draw) + P(away_win) = 1
    # If P(draw) = draw_pct, then:
    #   P(home_win) = E_home - 0.5 * draw_pct
    #   P(away_win) = 1 - draw_pct - P(home_win)
    p_home = max(0.0, p_home_vs_away - 0.5 * draw_pct)
    p_away = max(0.0, 1.0 - draw_pct - p_home)
    p_draw = max(0.0, 1.0 - p_home - p_away)
    return p_home, p_draw, p_away
