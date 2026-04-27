"""
model/gem/niches.py

Constants for Gem model: league clusters, target leagues, one-hot names.

League identity is a (name, country) tuple — both are needed because some
league names collide across countries (e.g. "Premier League" exists in
England and Ukraine). All single-string league IDs are produced via
`to_canonical(name, country)` → "Country: Name" (e.g. "England: Premier League").
"""

# Clusters define training groups and feature one-hot buckets.
# Each entry is (league_name, country).
TOP5_UCL: set[tuple[str, str]] = {
    ("Premier League",   "England"),
    ("La Liga",          "Spain"),
    ("Bundesliga",       "Germany"),
    ("Serie A",          "Italy"),
    ("Ligue 1",          "France"),
    ("Champions League", "Europe"),
}

SECOND_TIER: set[tuple[str, str]] = {
    ("Championship",        "England"),
    ("2. Bundesliga",       "Germany"),
    ("Eredivisie",          "Netherlands"),
    ("Jupiler Pro League",  "Belgium"),
    ("Primeira Liga",       "Portugal"),
    ("Süper Lig",           "Turkey"),
    ("Premier League",      "Ukraine"),
    ("Premiership",         "Scotland"),
    ("Ekstraklasa",         "Poland"),
    ("Serie B",             "Italy"),
    ("Eliteserien",         "Norway"),
    ("Allsvenskan",         "Sweden"),
}

TARGET_LEAGUES: set[tuple[str, str]] = TOP5_UCL | SECOND_TIER


def to_canonical(name: str, country: str | None) -> str:
    """Disambiguated league identifier — use everywhere a single string ID is needed.

    Format: "Country: Name". Uniform — no special cases.
    """
    return f"{country}: {name}"


# Deterministic order for one-hot encoding — keeps feature indices stable.
LEAGUE_NAMES_ORDERED: list[str] = sorted(to_canonical(n, c) for n, c in TARGET_LEAGUES)


def league_cluster(name: str, country: str | None = None) -> str:
    """Return cluster bucket for a league.

    Accepts either a (name, country) pair or a canonical "Country: Name" string
    via the `name` argument when country is None — the latter form is what gets
    serialised in artifacts.
    """
    if country is None:
        # `name` may already be in canonical "Country: Name" form
        if ": " in name:
            country, raw = name.split(": ", 1)
            key = (raw, country)
        else:
            key = (name, "")
    else:
        key = (name, country)
    if key in TOP5_UCL:
        return "top5_ucl"
    if key in SECOND_TIER:
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
