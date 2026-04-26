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

# ── Movement filter (sharp-money signal) ─────────────────────────────────
# Toggle + thresholds for `model.gem.movement_filter`. See movement_filter.md
# for full design rationale. Filter is graceful: insufficient data → allow.
ENABLE_MOVEMENT_FILTER       = True
MOVEMENT_DRIFT_THRESHOLD     = 0.05   # 5% odds drift against us → skip
MOVEMENT_VELOCITY_THRESHOLD  = 0.03   # 3% shift in last 30 min against us → skip
MOVEMENT_DISPERSION_THRESHOLD = 0.10  # std/mean > 10% across books → skip
MOVEMENT_MIN_SNAPSHOTS       = 3      # need ≥3 snapshots to compute drift
MOVEMENT_MIN_BOOKMAKERS      = 2      # need ≥2 books for dispersion
