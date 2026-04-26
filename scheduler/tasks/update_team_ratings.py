"""
scheduler/tasks/update_team_ratings.py

Recompute self-Glicko ratings for all teams from finished matches.

Runs:
  - Hourly (after live_tracker settles new results)
  - Persists current rating per team to `team_ratings` DB table
  - Future picks read directly from DB → no live SStats dependency

Performance: ~13k finished matches → ~3 sec full recompute.
"""
from datetime import datetime

from loguru import logger

from db.models import TeamRating
from db.session import SessionLocal
from model.glicko.compute import compute_ratings, load_finished_matches


def run_update_team_ratings() -> None:
    db = SessionLocal()
    try:
        logger.info("[Glicko] Loading finished matches …")
        matches = load_finished_matches()
        if matches.empty:
            logger.warning("[Glicko] No finished matches in DB")
            return

        logger.info(f"[Glicko] Computing ratings for {len(matches):,} matches …")
        _, current_ratings = compute_ratings(matches)

        # Per-team last match date + count
        team_last_date: dict[int, datetime] = {}
        team_count: dict[int, int] = {}
        for row in matches.itertuples(index=False):
            for tid in (int(row.home_team_id), int(row.away_team_id)):
                team_count[tid] = team_count.get(tid, 0) + 1
                team_last_date[tid] = row.date

        # Upsert
        upserted = 0
        for team_id, rating in current_ratings.items():
            existing = db.query(TeamRating).filter_by(team_id=team_id).first()
            if existing:
                existing.rating = rating.rating
                existing.rd = rating.rd
                existing.volatility = rating.volatility
                existing.matches_played = team_count.get(team_id, 0)
                existing.last_match_date = team_last_date.get(team_id)
                existing.updated_at = datetime.utcnow()
            else:
                db.add(TeamRating(
                    team_id=team_id,
                    rating=rating.rating,
                    rd=rating.rd,
                    volatility=rating.volatility,
                    matches_played=team_count.get(team_id, 0),
                    last_match_date=team_last_date.get(team_id),
                ))
            upserted += 1
        db.commit()
        logger.info(f"[Glicko] Upserted {upserted:,} team ratings")

        # Quick sanity: top-5 by rating
        top = (
            db.query(TeamRating)
            .order_by(TeamRating.rating.desc())
            .limit(5)
            .all()
        )
        from db.models import Team
        for t in top:
            team = db.query(Team).filter_by(id=t.team_id).first()
            logger.info(
                f"  Top: {team.name if team else f'#{t.team_id}'} | "
                f"rating={t.rating:.0f} rd={t.rd:.0f} matches={t.matches_played}"
            )

    finally:
        db.close()


if __name__ == "__main__":
    run_update_team_ratings()
