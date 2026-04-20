"""
Backfill match statistics from /Games/{id} endpoint.
Збирає: shots, possession, corners, passes, saves, cards, fouls, offsides.

Запуск: python -m data.backfill_match_stats
"""
import time
from loguru import logger
from data.api_client import SStatsClient
from db.session import SessionLocal
from db.models import Match, MatchStats

DELAY = 1.7        # ~35 req/хв — трохи нижче ліміту API
BATCH_SIZE = 50

# Ліги що реально використовуємо в моделі
ACTIVE_LEAGUE_API_IDS = {39, 140, 78, 135, 61, 94}  # EPL, La Liga, Bund, Serie A, Ligue 1, Primeira


def parse_stats(s: dict) -> dict:
    def iv(key):
        v = s.get(key)
        return int(v) if v is not None else None

    def fv(key):
        v = s.get(key)
        return float(v) if v is not None else None

    return {
        "home_shots":           iv("totalShotsHome"),
        "away_shots":           iv("totalShotsAway"),
        "home_shots_on_target": iv("shotsOnGoalHome"),
        "away_shots_on_target": iv("shotsOnGoalAway"),
        "home_shots_inside_box": iv("shotsInsideBoxHome"),
        "away_shots_inside_box": iv("shotsInsideBoxAway"),
        "home_possession":      fv("ballPossessionHome"),
        "away_possession":      fv("ballPossessionAway"),
        "home_corners":         iv("cornerKicksHome"),
        "away_corners":         iv("cornerKicksAway"),
        "home_passes_accurate": iv("passesAccurateHome"),
        "away_passes_accurate": iv("passesAccurateAway"),
        "home_passes_total":    iv("totalPassesHome"),
        "away_passes_total":    iv("totalPassesAway"),
        "home_gk_saves":        iv("goalkeeperSavesHome"),
        "away_gk_saves":        iv("goalkeeperSavesAway"),
        "home_yellow_cards":    iv("yellowCardsHome"),
        "away_yellow_cards":    iv("yellowCardsAway"),
        "home_red_cards":       iv("redCardsHome"),
        "away_red_cards":       iv("redCardsAway"),
        "home_fouls":           iv("foulsHome"),
        "away_fouls":           iv("foulsAway"),
        "home_offsides":        iv("offsidesHome"),
        "away_offsides":        iv("offsidesAway"),
    }


def run():
    db = SessionLocal()
    from db.models import League
    with SStatsClient() as client:
        try:
            # Тільки активні ліги + матчі де shots ще не заповнені
            active_league_ids = [
                l.id for l in db.query(League).all()
                if l.api_id in ACTIVE_LEAGUE_API_IDS
            ]
            matches = db.query(Match).join(MatchStats).filter(
                Match.status == "Finished",
                Match.league_id.in_(active_league_ids),
                MatchStats.home_shots.is_(None),
            ).all()

            logger.info(f"Матчів без статистики: {len(matches)}")

            done = 0
            failed = 0
            for match in matches:
                try:
                    data = client.get(f"/Games/{match.api_id}")
                    raw = (data.get("data") or {}).get("statistics") or {}
                    if not raw:
                        failed += 1
                        time.sleep(DELAY)
                        continue

                    parsed = parse_stats(raw)
                    # Оновлюємо існуючий MatchStats запис
                    ms = match.stats
                    for key, val in parsed.items():
                        setattr(ms, key, val)

                    done += 1
                    if done % BATCH_SIZE == 0:
                        db.commit()
                        logger.info(f"  {done}/{len(matches)} оброблено, {failed} без даних")
                    elif done % 10 == 0:
                        logger.info(f"  {done}/{len(matches)} ...")

                    time.sleep(DELAY)

                except Exception as e:
                    logger.warning(f"  Failed {match.api_id}: {e}")
                    failed += 1
                    time.sleep(DELAY)

            db.commit()
            logger.info(f"Backfill завершено: {done} оновлено, {failed} без даних")

        finally:
            db.close()


if __name__ == "__main__":
    run()
