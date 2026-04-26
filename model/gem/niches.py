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

# Gem filter thresholds (v3 — relaxed for higher yield)
# v2 (P_bet>0.70, gem>0.12) gave only 28 picks/24mo (1.3/wk — too low for live use).
# v3: relax all 3 thresholds → expected 80-150 picks/24mo (3-6/wk),
# WR may drop 67%→62% but yield × ROI improves.
MAX_DRAW_PROB = 0.32
MIN_BET_PROB = 0.60
MIN_GEM_SCORE = 0.05   # was 0.12 — still requires positive edge over market
MIN_ODDS = 1.50
MAX_ODDS = 3.00

# Kelly/stake config
FLAT_STAKE_FRAC = 0.04  # 4% of current bankroll
