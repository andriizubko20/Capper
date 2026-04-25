"""
model/gem/niches.py

Constants for Gem model: league clusters, target leagues, one-hot names.
"""

# Clusters define training groups and feature one-hot buckets.
TOP5_UCL = {
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
    "Champions League",
}

SECOND_TIER = {
    "Championship",
    "2. Bundesliga",
    "Eredivisie",
    "Jupiler Pro League",
    "Primeira Liga",
    "Süper Lig",
    "Eliteserien",
    "Allsvenskan",
}

TARGET_LEAGUES = TOP5_UCL | SECOND_TIER

# Deterministic order for one-hot encoding — keeps feature indices stable.
LEAGUE_NAMES_ORDERED: list[str] = sorted(TARGET_LEAGUES)


def league_cluster(name: str) -> str:
    if name in TOP5_UCL:
        return "top5_ucl"
    if name in SECOND_TIER:
        return "second_tier"
    return "other"


# Rolling window sizes
ROLLING_10 = 10
ROLLING_5 = 5

# Gem filter thresholds (v2)
# v1 had only 21 picks/24mo at P_bet>0.72. Smoke v2 with P_bet=0.68 + MIN_GEM=0.08
# gave 145 picks but WR 56.6% (filter let in noisy "10-20% gap" bucket).
# v2 final: keep tight P thresholds + require strong gem_score (top quartile on v1 = 85.7% WR).
MAX_DRAW_PROB = 0.28
MIN_BET_PROB = 0.70
MIN_GEM_SCORE = 0.12   # above the "bad" 10-20% bucket from v1 analysis
MIN_ODDS = 1.50
MAX_ODDS = 3.00

# Kelly/stake config
FLAT_STAKE_FRAC = 0.04  # 4% of current bankroll
