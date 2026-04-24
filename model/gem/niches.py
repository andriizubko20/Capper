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

# Gem filter thresholds (can be tuned after OOS backtest)
MAX_DRAW_PROB = 0.28
MIN_BET_PROB = 0.72
MIN_ODDS = 1.45
MAX_ODDS = 3.00

# Kelly/stake config
FLAT_STAKE_FRAC = 0.04  # 4% of current bankroll
