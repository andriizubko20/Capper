"""
scheduler/tasks/update_monster_p_is.py

Щотижнево (вівторок 03:00 UTC) оновлює IS win rate для кожної ніші Monster.
Використовує ВСІ завершені матчі як IS дані — без cutoff.
Зберігає в таблицю monster_p_is.
"""
from datetime import datetime

from loguru import logger

from db.models import MonsterPIs
from db.session import SessionLocal
from model.monster.niches import MODELS, parse_niche
from model.monster.features import load_historical_data, compute_p_is


def run_update_monster_p_is() -> None:
    logger.info("[Monster p_is] Starting weekly p_is update...")

    hist_matches, hist_stats, hist_odds = load_historical_data()
    logger.info(f"[Monster p_is] Loaded {len(hist_matches)} historical matches")

    db = SessionLocal()
    updated = skipped = 0

    try:
        for league, niches in MODELS.items():
            for niche_str in niches:
                niche = parse_niche(niche_str)

                # Використовуємо далеке майбутнє як cutoff = всі дані
                p = compute_p_is(
                    hist_matches, hist_stats, hist_odds,
                    league, niche,
                    cutoff=datetime(2099, 1, 1),
                )

                if p is None:
                    logger.debug(f"[Monster p_is] {league} | {niche_str} — < 3 samples, skip")
                    skipped += 1
                    continue

                # Обчислюємо n_samples
                from model.monster.niches import match_niche
                from collections import defaultdict
                import numpy as np
                import pandas as pd

                # Швидший підрахунок n через той самий compute_p_is але рахуємо total
                n = _count_niche_samples(hist_matches, hist_stats, hist_odds, league, niche, niche_str)

                existing = db.query(MonsterPIs).filter_by(league=league, niche_str=niche_str).first()
                if existing:
                    existing.p_is = p
                    existing.n_samples = n
                    existing.computed_at = datetime.utcnow()
                else:
                    db.add(MonsterPIs(
                        league=league,
                        niche_str=niche_str,
                        p_is=p,
                        n_samples=n,
                        computed_at=datetime.utcnow(),
                    ))

                logger.info(f"[Monster p_is] {league:20} | {niche_str:45} p={p:.3f} n={n}")
                updated += 1

        db.commit()
        logger.info(f"[Monster p_is] Done: {updated} updated, {skipped} skipped (< 3 samples)")

    finally:
        db.close()


def _count_niche_samples(hist_matches, hist_stats, hist_odds, league, niche, niche_str) -> int:
    """Count total IS matches for a niche (all data)."""
    from model.monster.niches import match_niche
    from collections import defaultdict
    import numpy as np

    m = hist_matches.copy()
    m = m.merge(hist_stats[['match_id', 'home_xg', 'away_xg']], on='match_id', how='left')
    m = m.merge(hist_odds, on='match_id', how='left')
    m = m.sort_values('date').reset_index(drop=True)

    elos: dict[int, float] = {}
    team_pts: dict[int, list] = defaultdict(list)
    team_xgf: dict[int, list] = defaultdict(list)
    team_xga: dict[int, list] = defaultdict(list)

    ELO_DEFAULT = 1500.0
    ELO_K = 32

    def elo_exp(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    total = 0

    for _, row in m.iterrows():
        h_id, a_id = int(row.home_team_id), int(row.away_team_id)
        elo_h = elos.get(h_id, ELO_DEFAULT)
        elo_a = elos.get(a_id, ELO_DEFAULT)

        h_pts5 = np.mean(team_pts[h_id][-5:]) if team_pts[h_id] else np.nan
        a_pts5 = np.mean(team_pts[a_id][-5:]) if team_pts[a_id] else np.nan
        h_xgf5 = np.nanmean(team_xgf[h_id][-5:]) if team_xgf[h_id] else np.nan
        h_xga5 = np.nanmean(team_xga[h_id][-5:]) if team_xga[h_id] else np.nan
        a_xgf5 = np.nanmean(team_xgf[a_id][-5:]) if team_xgf[a_id] else np.nan
        a_xga5 = np.nanmean(team_xga[a_id][-5:]) if team_xga[a_id] else np.nan

        h_xg_ratio = h_xgf5 / max(h_xga5, 0.1) if not np.isnan(h_xgf5) else np.nan
        a_xg_ratio = a_xgf5 / max(a_xga5, 0.1) if not np.isnan(a_xgf5) else np.nan

        h_odds = row.get('home_odds')
        a_odds = row.get('away_odds')
        d_odds = row.get('draw_odds')
        mkt_h = mkt_a = None
        if h_odds and d_odds and a_odds and h_odds > 0 and d_odds > 0 and a_odds > 0:
            raw = 1/h_odds + 1/d_odds + 1/a_odds
            mkt_h = (1/h_odds) / raw
            mkt_a = (1/a_odds) / raw

        features = {
            'home_odds': h_odds, 'away_odds': a_odds,
            'elo_diff': elo_h - elo_a,
            'home_pts_5': h_pts5, 'away_pts_5': a_pts5,
            'xg_ratio_home_5': h_xg_ratio, 'xg_ratio_away_5': a_xg_ratio,
            'mkt_home_prob': mkt_h, 'mkt_away_prob': mkt_a,
        }

        if match_niche(features, niche, league, row.get('league_name', '')):
            total += 1

        exp_h = elo_exp(elo_h, elo_a)
        s_h = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        elos[h_id] = elo_h + ELO_K * (s_h - exp_h)
        elos[a_id] = elo_a + ELO_K * ((1 - s_h) - (1 - exp_h))

        pts_h = 3 if row.result == 'H' else (1 if row.result == 'D' else 0)
        pts_a = 3 if row.result == 'A' else (1 if row.result == 'D' else 0)
        team_pts[h_id].append(pts_h)
        team_pts[a_id].append(pts_a)

        hxg = row.get('home_xg')
        axg = row.get('away_xg')
        team_xgf[h_id].append(hxg if hxg else np.nan)
        team_xga[h_id].append(axg if axg else np.nan)
        team_xgf[a_id].append(axg if axg else np.nan)
        team_xga[a_id].append(hxg if hxg else np.nan)

    return total
