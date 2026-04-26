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

# Gem filter thresholds (v4 — sweep-optimal yield × ROI)
# v3 (P_bet>0.60, gem>0.05) was over-relaxed: 274 picks but ROI -4% on OOF.
# v4 from threshold sweep: 0.62/0.15 wins on annual_units (+4.54 u/yr):
#   60 picks/24mo (~22/yr, ~2/wk), WR 61.7%, lo95 49%, ROI +20.2%.
# Trade-off: lower WR than v2 (67%) but 2× yield × ROI.
MAX_DRAW_PROB = 0.32
MIN_BET_PROB = 0.62
MIN_GEM_SCORE = 0.15
MIN_ODDS = 1.50
MAX_ODDS = 3.00

# Kelly/stake config
FLAT_STAKE_FRAC = 0.04  # 4% of current bankroll
