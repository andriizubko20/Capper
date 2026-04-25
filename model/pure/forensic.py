"""
model/pure/forensic.py

REVERSE ENGINEERING from a hand-curated list of "winning gem" matches.

Workflow:
  1. User provides match list (cyrillic team names + date + score).
  2. Parser → match_ids in DB.
  3. Feature extraction (same as Pure features.py).
  4. Comparative analysis:
       - User-set distribution per feature (mean, median, quartiles).
       - Population distribution (all candidate "fav at attractive odds" matches).
       - Anomaly score: where do user-set values fall vs population?
  5. Rule extraction: features whose user-set distribution shifts most from
     population are candidate threshold conditions.

Output: console report + CSVs for manual review.

Usage:
  python -m model.pure.forensic
"""
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from db.session import SessionLocal

REPORTS = Path(__file__).parent / "reports"


# ── Cyrillic → DB team name mapping (English used in DB) ─────────────────────
# Map each Cyrillic alias → canonical English DB name (or substring for fuzzy)
TEAM_ALIASES = {
    # Bundesliga
    "Лейпциг": "RB Leipzig",
    "Уніон": "Union Berlin",
    "Уніон Берлін": "Union Berlin",
    "Вольфсбург": "Wolfsburg",
    "Айнтрахт": "Eintracht Frankfurt",
    "Фрайбург": "Freiburg",
    "Хайденхайм": "Heidenheim",
    "Вердер": "Werder Bremen",
    # Premier League
    "Брайтон": "Brighton",
    "Челсі": "Chelsea",
    "Евертон": "Everton",
    "Ліверпуль": "Liverpool",
    "Манчестер Сіті": "Manchester City",
    "Манчестер Юнайтенд": "Manchester United",
    "Манчестер Юнайтед": "Manchester United",
    "Арсенал": "Arsenal",
    # Serie A
    "Верона": "Hellas Verona",
    "Мілан": "AC Milan",
    "Піза": "Pisa",
    "Дженоа": "Genoa",
    "Болонья": "Bologna",
    "Лечче": "Lecce",
    "Сассуоло": "Sassuolo",
    "Торіно": "Torino",
    "Рома": "AS Roma",
    "Аталанта": "Atalanta",
    "Кремонезе": "Cremonese",
    "Фіорентіна": "Fiorentina",
    "Кальярі": "Cagliari",
    "Наполі": "Napoli",
    "Комо": "Como",
    # La Liga
    "Жирона": "Girona",
    "Бетіс": "Real Betis",
    # Ligue 1
    "Мец": "Metz",
    "Париж": "Paris Saint Germain",
    # Ukrainian — likely not in DB
    "Олександрія": "Oleksandria",
    "Верес": "Veres Rivne",
    "Шахтар": "Shakhtar Donetsk",
    "Полісся": "Polissya",
    "Кривбас": "Kryvbas",
    "Рух": "Rukh Lviv",
}


# Hand-curated user list: (home_alias, away_alias, date, home_score, away_score)
USER_MATCHES = [
    ("Лейпциг", "Уніон Берлін", "2026-04-24", 3, 1),
    ("Брайтон", "Челсі", "2026-04-21", 3, 0),
    ("Жирона", "Бетіс", "2026-04-21", 2, 3),
    # ("Олександрія", "Верес", "2026-04-20", 0, 3),  # UPL, prob no data
    # ("Шахтар", "Полісся", "2026-04-20", 1, 0),
    ("Евертон", "Ліверпуль", "2026-04-19", 1, 2),
    ("Верона", "Мілан", "2026-04-19", 0, 1),
    ("Піза", "Дженоа", "2026-04-19", 1, 2),
    ("Фрайбург", "Хайденхайм", "2026-04-19", 2, 1),
    # ("Кривбас", "Рух", "2026-04-19", 3, 0),
    ("Мец", "Париж", "2026-04-19", 1, 3),
    ("Челсі", "Манчестер Юнайтенд", "2026-04-19", 0, 1),
    ("Уніон Берлін", "Вольфсбург", "2026-04-19", 1, 2),
    ("Айнтрахт", "Лейпциг", "2026-04-19", 1, 3),
    ("Болонья", "Лечче", "2026-04-12", 2, 0),
    ("Дженоа", "Сассуоло", "2026-04-12", 2, 1),
    ("Торіно", "Верона", "2026-04-11", 2, 1),
    ("Рома", "Піза", "2026-04-10", 3, 0),
    ("Лечче", "Аталанта", "2026-04-06", 0, 3),
    ("Піза", "Торіно", "2026-04-05", 0, 1),
    ("Кремонезе", "Болонья", "2026-04-05", 1, 2),
    ("Верона", "Фіорентіна", "2026-04-04", 0, 1),
    ("Вердер", "Лейпциг", "2026-04-04", 1, 2),
    ("Рома", "Лечче", "2026-03-22", 1, 0),
    ("Аталанта", "Верона", "2026-03-22", 1, 0),
    ("Комо", "Піза", "2026-03-22", 5, 0),
    ("Кальярі", "Наполі", "2026-03-20", 0, 1),
    ("Кремонезе", "Фіорентіна", "2026-03-16", 1, 4),
    ("Верона", "Дженоа", "2026-03-15", 0, 2),
]


def find_match_ids() -> pd.DataFrame:
    """Locate user matches in DB by team alias + date. Return found-only."""
    db = SessionLocal()
    rows = []
    try:
        for home_a, away_a, date_str, hs, as_ in USER_MATCHES:
            home_db = TEAM_ALIASES.get(home_a)
            away_db = TEAM_ALIASES.get(away_a)
            if not home_db or not away_db:
                logger.warning(f"  ❓ no alias for {home_a} or {away_a}")
                continue

            q = """
                SELECT m.id, m.date, l.name AS league, m.home_score, m.away_score,
                       th.name AS home_name, ta.name AS away_name
                FROM matches m
                JOIN teams th ON th.id = m.home_team_id
                JOIN teams ta ON ta.id = m.away_team_id
                JOIN leagues l ON l.id = m.league_id
                WHERE m.date::date = :d
                  AND (th.name ILIKE :hh OR th.name ILIKE :hh2)
                  AND (ta.name ILIKE :aa OR ta.name ILIKE :aa2)
                  AND m.home_score IS NOT NULL
                LIMIT 5
            """
            res = db.execute(
                text(q),
                {
                    "d": date_str,
                    "hh": f"%{home_db}%",  "hh2": f"%{home_db.split()[0]}%",
                    "aa": f"%{away_db}%",  "aa2": f"%{away_db.split()[0]}%",
                },
            ).fetchall()
            if not res:
                logger.warning(f"  ❌ NOT FOUND: {home_a} vs {away_a} ({date_str})")
                continue
            r = res[0]
            # Sanity: actual score should match user's
            if (r.home_score, r.away_score) != (hs, as_):
                logger.warning(
                    f"  ⚠️  {home_a} vs {away_a} ({date_str}): score mismatch "
                    f"(DB {r.home_score}-{r.away_score}, user {hs}-{as_})"
                )
            rows.append({
                "match_id": r.id,
                "date":     r.date,
                "league":   r.league,
                "home":     r.home_name,
                "away":     r.away_name,
                "home_score": r.home_score,
                "away_score": r.away_score,
                "result":   "H" if r.home_score > r.away_score else ("A" if r.away_score > r.home_score else "D"),
            })
    finally:
        db.close()

    df = pd.DataFrame(rows)
    logger.info(f"Resolved {len(df)} / {len(USER_MATCHES)} user matches in DB")
    return df


def extract_features(match_ids: list[int]) -> pd.DataFrame:
    """Pull match_factors.parquet rows for the matched ids."""
    factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    sub = factors[factors["match_id"].isin(match_ids)].copy()
    return sub


def perspective_features(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    For each match in user list, return features FROM THE WINNER'S perspective.

    Joins match_factors with the resolved match list to know which side won.
    Negates home-perspective columns when the winner was AWAY.
    """
    df = df.merge(results[["match_id", "result"]], on="match_id", how="inner", suffixes=("", "_user"))
    out = []
    for r in df.itertuples(index=False):
        side = "home" if r.result == "H" else ("away" if r.result == "A" else None)
        if side is None:
            continue
        winner_view = {
            "match_id":   r.match_id,
            "date":       r.date,
            "league":     r.league_name,
            "side":       side,
            "odds":       r.home_odds if side == "home" else r.away_odds,
            "market_prob": r.home_market_prob if side == "home" else r.away_market_prob,
            "glicko_prob": r.home_glicko_prob if side == "home" else r.away_glicko_prob,
            "glicko_gap":  r.glicko_gap if side == "home" else -r.glicko_gap,
            "xg_diff":     r.xg_diff_home if side == "home" else r.xg_diff_away,
            "attack_vs_def": r.attack_vs_def_home if side == "home" else r.attack_vs_def_away,
            "form_advantage": r.form_advantage if side == "home" else -r.form_advantage,
        }
        out.append(winner_view)
    return pd.DataFrame(out)


def population_features(leagues: list[str]) -> pd.DataFrame:
    """
    Build a population dataframe of all matches in the same leagues, oriented
    from the WINNING side's perspective (we want to compare apples to apples).
    """
    factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    factors = factors[factors["league_name"].isin(leagues)].copy()

    rows = []
    for r in factors.itertuples(index=False):
        if r.result == "D":
            continue
        side = "home" if r.result == "H" else "away"
        rows.append({
            "match_id":   r.match_id,
            "league":     r.league_name,
            "side":       side,
            "odds":       r.home_odds if side == "home" else r.away_odds,
            "market_prob": r.home_market_prob if side == "home" else r.away_market_prob,
            "glicko_prob": r.home_glicko_prob if side == "home" else r.away_glicko_prob,
            "glicko_gap":  r.glicko_gap if side == "home" else -r.glicko_gap,
            "xg_diff":     r.xg_diff_home if side == "home" else r.xg_diff_away,
            "attack_vs_def": r.attack_vs_def_home if side == "home" else r.attack_vs_def_away,
            "form_advantage": r.form_advantage if side == "home" else -r.form_advantage,
        })
    return pd.DataFrame(rows)


def report(user_view: pd.DataFrame, population: pd.DataFrame) -> None:
    if user_view.empty:
        print("No user matches resolved.")
        return

    cols = ["odds", "market_prob", "glicko_prob", "glicko_gap",
            "xg_diff", "attack_vs_def", "form_advantage"]

    print("\n" + "=" * 100)
    print(f"USER-CURATED WINNERS (n={len(user_view)}) vs POPULATION WINNERS (n={len(population):,})")
    print("=" * 100)
    print(f"{'feature':>20s} | {'user_q25':>9s} {'user_med':>9s} {'user_q75':>9s} | "
          f"{'pop_q25':>9s} {'pop_med':>9s} {'pop_q75':>9s} | {'shift_med':>10s}  pct_above_median_pop")
    print("-" * 130)
    for c in cols:
        u = user_view[c].dropna()
        p = population[c].dropna()
        if u.empty or p.empty:
            continue
        u_q = u.quantile([0.25, 0.5, 0.75]).tolist()
        p_q = p.quantile([0.25, 0.5, 0.75]).tolist()
        # Where does user's median sit in population CDF? (anomaly indicator)
        pct_above = (p < u_q[1]).mean() * 100  # % of pop below user's median
        shift = u_q[1] - p_q[1]
        print(
            f"{c:>20s} | "
            f"{u_q[0]:>9.3f} {u_q[1]:>9.3f} {u_q[2]:>9.3f} | "
            f"{p_q[0]:>9.3f} {p_q[1]:>9.3f} {p_q[2]:>9.3f} | "
            f"{shift:>+10.3f}  user_med is at p={pct_above:.0f} of pop CDF"
        )

    # Cohen's d (winner-perspective): user vs ALL pop winners
    print("\nEFFECT SIZE (Cohen's d) — user vs pop winners:")
    for c in cols:
        u = user_view[c].dropna().values
        p = population[c].dropna().values
        if len(u) < 5 or len(p) < 30:
            continue
        pooled_sd = np.sqrt(((len(u) - 1) * u.var(ddof=1) + (len(p) - 1) * p.var(ddof=1)) /
                            (len(u) + len(p) - 2))
        d = (u.mean() - p.mean()) / pooled_sd if pooled_sd > 1e-9 else 0
        marker = "🔥" if abs(d) > 0.5 else ("⚡" if abs(d) > 0.3 else "  ")
        print(f"  {marker} {c:>20s}: d = {d:+.3f}  (user {u.mean():+.3f} vs pop {p.mean():+.3f})")

    # Side breakdown of user list
    print("\nUser-list side breakdown:")
    print(user_view["side"].value_counts().to_string())

    user_view.to_csv(REPORTS / "forensic_user_winners.csv", index=False)
    population.to_csv(REPORTS / "forensic_population_winners.csv", index=False)


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logger.info("Resolving user matches in DB …")
    resolved = find_match_ids()
    if resolved.empty:
        logger.error("No matches resolved — fix team aliases.")
        return
    print("\nResolved matches:")
    for r in resolved.itertuples():
        print(f"  {r.match_id:>5d}  {r.date.date()}  {r.league:>22s}  {r.home:>22s} {r.home_score}-{r.away_score} {r.away}")

    factors = extract_features(resolved["match_id"].tolist())
    user_view = perspective_features(factors, resolved)

    leagues_in_user = list(resolved["league"].unique())
    logger.info(f"Computing population winners across leagues: {leagues_in_user}")
    pop = population_features(leagues_in_user)

    report(user_view, pop)


if __name__ == "__main__":
    run()
