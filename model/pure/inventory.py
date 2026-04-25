"""
model/pure/inventory.py

Full inventory of variables / parameters we can use as filter/score signals.

Categories:
  ── ODDS / MARKET ───────── 1x2 odds, de-vigged probs, ratios
  ── STRENGTH ──────────────  Glicko ratings + probs, gaps
  ── ATTACK / DEFENSE (xG) ──  rolling 10-match xG splits + interactions
  ── FORM (PPG) ────────────  10-match avg points, last 5 form, advantages
  ── MOMENTUM ──────────────  xG trend, Glicko momentum, streaks
  ── STYLE ─────────────────  possession, SoT, pass accuracy
  ── PHYSICAL ──────────────  rest days, injuries flag
  ── HEAD-TO-HEAD ──────────  last 5 H2H WR + avg goals + last result
  ── LEAGUE CONTEXT ─────────  league baseline WR, cluster, league one-hot
  ── SAMPLE SIZE / META ─────  matches_played, date, league_name

For each variable we report:
  - source     : which file/function builds it (gem.team_state / data / etc.)
  - type       : continuous | ordinal | binary | categorical
  - direction  : positive_for_side | negative_for_side | symmetric | n/a
  - in_pure    : yes | no | derivable
  - range      : observed min / median / max from match_factors.parquet
  - n_nonnull  : how many rows have a value
  - notes      : anything quirky
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPORTS = Path(__file__).parent / "reports"


# ── Variable inventory (semantic metadata) ───────────────────────────────────
VARIABLES = [
    # ── ODDS / MARKET ──────────────────────────────────────────────────────
    {"name": "home_odds",            "category": "MARKET",   "type": "continuous", "direction": "low_means_home_favored",
     "in_pure": "yes", "source": "odds table (is_closing=false)",
     "notes": "Decimal 1x2 odds for HOME. Lower = market sees home as favorite. CAVEAT: 94% rows are post-match closing (leakage for training)."},
    {"name": "draw_odds",            "category": "MARKET",   "type": "continuous", "direction": "n/a",
     "in_pure": "no (used for de-vig)", "source": "odds table",
     "notes": "Decimal odds for draw. Used to remove bookmaker margin via 1/o normalization."},
    {"name": "away_odds",            "category": "MARKET",   "type": "continuous", "direction": "low_means_away_favored",
     "in_pure": "yes", "source": "odds table",
     "notes": "Decimal 1x2 odds for AWAY."},
    {"name": "home_market_prob",     "category": "MARKET",   "type": "continuous", "direction": "high_means_home_favored",
     "in_pure": "yes (max_market_prob filter)", "source": "de-vigged from odds (1/o / sum(1/o))",
     "notes": "Implied probability HOME wins, removed bookmaker margin. Used to find 'undervalued' (cap max → only low-implied = mispriced fav)."},
    {"name": "draw_market_prob",     "category": "MARKET",   "type": "continuous", "direction": "n/a",
     "in_pure": "no", "source": "de-vigged",
     "notes": "Implied draw probability."},
    {"name": "away_market_prob",     "category": "MARKET",   "type": "continuous", "direction": "high_means_away_favored",
     "in_pure": "yes", "source": "de-vigged",
     "notes": "Implied probability AWAY wins."},
    {"name": "odds_ratio",           "category": "MARKET",   "type": "continuous", "direction": "high_means_home_favored",
     "in_pure": "no (derivable)", "source": "away_odds / home_odds",
     "notes": "How much shorter the home odds are vs away. Quick intuition for 'how clear is the favorite'."},
    {"name": "vig",                  "category": "MARKET",   "type": "continuous", "direction": "n/a",
     "in_pure": "no (derivable)", "source": "(1/h + 1/d + 1/a) - 1",
     "notes": "Bookmaker margin. Typically 4-7%. Higher = thinner edge."},

    # ── STRENGTH (Glicko) ──────────────────────────────────────────────────
    {"name": "home_glicko",          "category": "STRENGTH", "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "yes (via glicko_gap)", "source": "match_stats.home_glicko (PRE-match)",
     "notes": "SStats Glicko-2 rating for home team. Default ~1500. Higher = stronger."},
    {"name": "away_glicko",          "category": "STRENGTH", "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "yes (via glicko_gap)", "source": "match_stats.away_glicko (PRE-match)",
     "notes": "Glicko rating for away team."},
    {"name": "glicko_gap",           "category": "STRENGTH", "type": "continuous", "direction": "positive_for_home",
     "in_pure": "YES (key)", "source": "home_glicko - away_glicko",
     "notes": "STRONGEST single discriminative feature in calibration. Top decile WR 78% if betting home."},
    {"name": "home_glicko_prob",     "category": "STRENGTH", "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "YES (key)", "source": "match_stats.home_win_prob",
     "notes": "Pre-match Glicko model's win probability for home. Best single calibrated factor (top decile 79% WR)."},
    {"name": "away_glicko_prob",     "category": "STRENGTH", "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "YES", "source": "match_stats.away_win_prob",
     "notes": "Glicko prob for away."},
    {"name": "glicko_prob_diff",     "category": "STRENGTH", "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no (derivable)", "source": "home_glicko_prob - home_market_prob (mispricing signal)",
     "notes": "Our Glicko model says home is N% likely; market says M%. Diff = the 'gem' edge."},

    # ── ATTACK / DEFENSE (xG) ─────────────────────────────────────────────
    {"name": "home_xg_for_10",       "category": "XG_ATTACK", "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "yes (via xg_diff_home)", "source": "rolling 10-match avg of HOME team's xG_for",
     "notes": "Last 10 matches xG scored. Range 0.5-3.0."},
    {"name": "home_xg_against_10",   "category": "XG_DEFENSE","type": "continuous", "direction": "low_means_home_strong",
     "in_pure": "yes (via xg_diff_home)", "source": "rolling 10-match avg of HOME's xG conceded",
     "notes": "Defense quality: lower better."},
    {"name": "away_xg_for_10",       "category": "XG_ATTACK", "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "yes", "source": "rolling 10-match avg, AWAY team",
     "notes": "Same as home_xg_for_10 but for away."},
    {"name": "away_xg_against_10",   "category": "XG_DEFENSE","type": "continuous", "direction": "low_means_away_strong",
     "in_pure": "yes", "source": "rolling 10-match",
     "notes": "Away team's defensive quality."},
    {"name": "xg_diff_home",         "category": "XG_NET",    "type": "continuous", "direction": "positive_for_home",
     "in_pure": "YES", "source": "home_xg_for_10 - home_xg_against_10",
     "notes": "Home team's net xG (attack minus defense). Positive = good team."},
    {"name": "xg_diff_away",         "category": "XG_NET",    "type": "continuous", "direction": "positive_for_away",
     "in_pure": "YES", "source": "away_xg_for_10 - away_xg_against_10",
     "notes": "Away team's net xG."},
    {"name": "attack_vs_def_home",   "category": "XG_MATCHUP","type": "continuous", "direction": "positive_for_home",
     "in_pure": "YES", "source": "home_xg_for_10 - away_xg_against_10",
     "notes": "Home attack vs away defense matchup. Positive = home outscores expected."},
    {"name": "attack_vs_def_away",   "category": "XG_MATCHUP","type": "continuous", "direction": "positive_for_away",
     "in_pure": "YES", "source": "away_xg_for_10 - home_xg_against_10",
     "notes": "Away attack vs home defense."},
    {"name": "xg_quality_gap",       "category": "XG_MATCHUP","type": "continuous", "direction": "positive_for_home",
     "in_pure": "YES (Gem v2)", "source": "(home.xg_for - away.xg_against) - (away.xg_for - home.xg_against)",
     "notes": "Net matchup advantage in xG terms. Captures both attack and defense imbalances."},
    {"name": "xg_for_home_10",       "category": "XG_HOME_SPLIT", "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "no (Gem only)", "source": "rolling 10 of HOME team xG_for ONLY when playing at HOME",
     "notes": "Home team's xG when playing AT HOME (different from overall). Captures home advantage in xG."},
    {"name": "xg_against_home_10",   "category": "XG_HOME_SPLIT", "type": "continuous", "direction": "low_means_home_strong",
     "in_pure": "no", "source": "rolling 10, only home matches",
     "notes": "Home team's defensive xG at home."},
    {"name": "xg_for_away_10",       "category": "XG_AWAY_SPLIT", "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "no", "source": "rolling 10, only away matches",
     "notes": "Away team's xG when playing on the road."},
    {"name": "xg_against_away_10",   "category": "XG_AWAY_SPLIT", "type": "continuous", "direction": "low_means_away_strong",
     "in_pure": "no", "source": "rolling 10, only away matches",
     "notes": "Away team's defensive xG on the road."},

    # ── FORM (PPG) ────────────────────────────────────────────────────────
    {"name": "home_ppg_10",          "category": "FORM",      "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "yes (via form_advantage)", "source": "rolling 10-match avg points (3 win/1 draw/0 loss)",
     "notes": "Home team's points-per-game over last 10. Range 0-3."},
    {"name": "away_ppg_10",          "category": "FORM",      "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "yes", "source": "rolling 10-match",
     "notes": "Away team's PPG."},
    {"name": "form_advantage",       "category": "FORM",      "type": "continuous", "direction": "positive_for_home",
     "in_pure": "YES (forensic key)", "source": "home_ppg_10 - away_ppg_10",
     "notes": "🔥 STRONGEST signal in user-curated forensic analysis (Cohen's d=+0.70)."},
    {"name": "home_form_5",          "category": "FORM",      "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "no (Gem)", "source": "rolling 5-match avg points",
     "notes": "Shorter window than ppg_10 — captures recent shape change."},
    {"name": "away_form_5",          "category": "FORM",      "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "no", "source": "rolling 5-match",
     "notes": "Same for away."},
    {"name": "form_advantage_5",     "category": "FORM",      "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no (derivable)", "source": "home_form_5 - away_form_5",
     "notes": "Short-term form gap (last 5 vs last 5). Possibly more predictive than 10-match."},
    {"name": "ppg_home_10",          "category": "FORM_SPLIT","type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "no", "source": "rolling 10, HOME matches only",
     "notes": "Home team's PPG at home — captures home advantage strength."},
    {"name": "ppg_away_10",          "category": "FORM_SPLIT","type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "no", "source": "rolling 10, AWAY matches only",
     "notes": "Away team's PPG on the road."},

    # ── MOMENTUM ──────────────────────────────────────────────────────────
    {"name": "home_xg_trend",        "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no", "source": "last_5_xg_for - rolling_10_xg_for (home team)",
     "notes": "Positive = home team's attack is accelerating recently."},
    {"name": "away_xg_trend",        "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_away",
     "in_pure": "no", "source": "last_5 - rolling_10 (away)",
     "notes": "Same for away."},
    {"name": "xg_trend_advantage",   "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no (derivable)", "source": "home_xg_trend - away_xg_trend",
     "notes": "Net momentum diff."},
    {"name": "home_glicko_momentum", "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no", "source": "home_glicko_now - home_glicko_5_matches_ago",
     "notes": "Glicko rating drift over last 5 matches. Proxy for squad health/rotation effect."},
    {"name": "away_glicko_momentum", "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_away",
     "in_pure": "no", "source": "Glicko drift over 5",
     "notes": ""},
    {"name": "glicko_momentum_diff", "category": "MOMENTUM",  "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no (derivable)", "source": "home_glicko_momentum - away_glicko_momentum",
     "notes": "Composite momentum gap."},
    {"name": "home_win_streak",      "category": "MOMENTUM",  "type": "ordinal",    "direction": "high_means_home_strong",
     "in_pure": "no", "source": "consecutive wins ending at last match",
     "notes": "0 if not on win streak."},
    {"name": "home_lose_streak",     "category": "MOMENTUM",  "type": "ordinal",    "direction": "low_means_home_strong",
     "in_pure": "no", "source": "consecutive losses",
     "notes": "0 if not on lose streak."},
    {"name": "away_win_streak",      "category": "MOMENTUM",  "type": "ordinal",    "direction": "high_means_away_strong",
     "in_pure": "no", "source": "consecutive wins (away team)",
     "notes": ""},
    {"name": "away_lose_streak",     "category": "MOMENTUM",  "type": "ordinal",    "direction": "low_means_away_strong",
     "in_pure": "no", "source": "consecutive losses (away team)",
     "notes": "Could be 'opponent weakness' signal: hot favorite vs cold opponent."},

    # ── STYLE ─────────────────────────────────────────────────────────────
    {"name": "home_possession_10",   "category": "STYLE",     "type": "continuous", "direction": "ambiguous",
     "in_pure": "no", "source": "rolling 10-match possession %",
     "notes": "0-100. Higher = more possession-based; not necessarily better."},
    {"name": "away_possession_10",   "category": "STYLE",     "type": "continuous", "direction": "ambiguous",
     "in_pure": "no", "source": "rolling 10",
     "notes": ""},
    {"name": "home_sot_10",          "category": "STYLE",     "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "no", "source": "shots on target rolling 10",
     "notes": "Higher = more attacking output."},
    {"name": "away_sot_10",          "category": "STYLE",     "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "no", "source": "rolling 10",
     "notes": ""},
    {"name": "home_pass_acc_10",     "category": "STYLE",     "type": "continuous", "direction": "high_means_home_strong",
     "in_pure": "no", "source": "pass accuracy rolling 10 (passes_accurate/passes_total)",
     "notes": "Quality of build-up. Top-tier teams ~85-90%."},
    {"name": "away_pass_acc_10",     "category": "STYLE",     "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "no", "source": "rolling 10",
     "notes": ""},

    # ── PHYSICAL ──────────────────────────────────────────────────────────
    {"name": "home_rest_days",       "category": "PHYSICAL",  "type": "ordinal",    "direction": "high_means_home_strong",
     "in_pure": "no", "source": "match_date - home.last_match_date",
     "notes": "Days since last match. 7+ = good rest, <4 = midweek-tired."},
    {"name": "away_rest_days",       "category": "PHYSICAL",  "type": "ordinal",    "direction": "high_means_away_strong",
     "in_pure": "no", "source": "Same for away",
     "notes": ""},
    {"name": "rest_advantage",       "category": "PHYSICAL",  "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no (derivable)", "source": "home_rest - away_rest",
     "notes": "Travel/fatigue advantage. Positive = home is fresher."},
    {"name": "home_has_injuries",    "category": "PHYSICAL",  "type": "binary",     "direction": "true_means_home_weak",
     "in_pure": "no", "source": "has any row in injury_reports for this match+team",
     "notes": "Crude flag — doesn't weight player importance."},
    {"name": "away_has_injuries",    "category": "PHYSICAL",  "type": "binary",     "direction": "true_means_away_weak",
     "in_pure": "no", "source": "Same for away",
     "notes": ""},

    # ── HEAD-TO-HEAD ──────────────────────────────────────────────────────
    {"name": "h2h_home_wr",          "category": "H2H",       "type": "continuous", "direction": "positive_for_home",
     "in_pure": "no", "source": "win rate of HOME team in last 5 H2H meetings (any venue)",
     "notes": "Range 0-1. Niches with strong dynastic dominance."},
    {"name": "h2h_avg_goals",        "category": "H2H",       "type": "continuous", "direction": "ambiguous",
     "in_pure": "no", "source": "avg total goals in last 5 H2H",
     "notes": "Style indicator; doesn't directly predict winner."},
    {"name": "h2h_home_last_result", "category": "H2H",       "type": "ordinal",    "direction": "positive_for_home",
     "in_pure": "no", "source": "1 if home won last H2H, -1 if lost, 0 if drew",
     "notes": "Recency: was last meeting a win? Some weight."},

    # ── LEAGUE CONTEXT ────────────────────────────────────────────────────
    {"name": "league_name",          "category": "CONTEXT",   "type": "categorical","direction": "n/a",
     "in_pure": "YES (one-hot + per-league models)", "source": "leagues.name",
     "notes": "Per-league niche tuning is essential — different leagues have different home advantage and form impacts."},
    {"name": "league_cluster_top5",  "category": "CONTEXT",   "type": "binary",     "direction": "n/a",
     "in_pure": "no (Gem only)", "source": "league in top-5 + UCL set",
     "notes": "Top-tier vs second-tier."},
    {"name": "league_home_wr",       "category": "CONTEXT",   "type": "continuous", "direction": "high_means_home_advantage",
     "in_pure": "no", "source": "all-time home win rate for this league",
     "notes": "Baseline. Range ~0.40-0.48. La Liga ~0.46, Serie A ~0.40."},
    {"name": "league_prior_home_wr", "category": "CONTEXT",   "type": "continuous", "direction": "high_means_home_advantage",
     "in_pure": "Gem only", "source": "Bayesian-smoothed train-only home WR (target encoding)",
     "notes": "Computed per fold to avoid leakage."},
    {"name": "league_prior_draw_rate","category": "CONTEXT",  "type": "continuous", "direction": "n/a",
     "in_pure": "Gem only", "source": "target encoding",
     "notes": "League-specific draw rate (varies 22-28%)."},
    {"name": "league_prior_away_wr", "category": "CONTEXT",   "type": "continuous", "direction": "high_means_away_strong",
     "in_pure": "Gem only", "source": "target encoding",
     "notes": ""},

    # ── SAMPLE SIZE / META ────────────────────────────────────────────────
    {"name": "matches_played_home",  "category": "META",      "type": "ordinal",    "direction": "n/a",
     "in_pure": "yes (min_matches_played)", "source": "team's prior matches in dataset",
     "notes": "Filter early-season noise. Typical: require >=5."},
    {"name": "matches_played_away",  "category": "META",      "type": "ordinal",    "direction": "n/a",
     "in_pure": "yes", "source": "Same for away",
     "notes": ""},
    {"name": "date",                 "category": "META",      "type": "ordinal",    "direction": "n/a",
     "in_pure": "yes (chronological splits)", "source": "matches.date",
     "notes": "Used for walk-forward CV and stability checks."},
    {"name": "last_match_date_home", "category": "META",      "type": "ordinal",    "direction": "n/a",
     "in_pure": "no (derivable for rest)", "source": "team's last match in dataset",
     "notes": "Used to compute rest_days."},
]


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df_inv = pd.DataFrame(VARIABLES)

    # Augment with empirical stats from match_factors.parquet where applicable
    factors = pd.read_parquet(REPORTS / "match_factors.parquet")
    stats_rows = []
    for v in VARIABLES:
        col = v["name"]
        if col in factors.columns and pd.api.types.is_numeric_dtype(factors[col]):
            s = factors[col].dropna()
            if len(s) > 0:
                stats_rows.append({
                    "name": col,
                    "n_nonnull": int(len(s)),
                    "min":      round(float(s.min()), 3),
                    "q25":      round(float(s.quantile(0.25)), 3),
                    "median":   round(float(s.quantile(0.50)), 3),
                    "q75":      round(float(s.quantile(0.75)), 3),
                    "max":      round(float(s.max()), 3),
                })
    stats_df = pd.DataFrame(stats_rows)
    df_inv = df_inv.merge(stats_df, on="name", how="left")

    out = REPORTS / "variable_inventory.csv"
    df_inv.to_csv(out, index=False)

    md_out = REPORTS / "variable_inventory.md"
    with open(md_out, "w") as f:
        f.write("# Pure / Gem Variable Inventory\n\n")
        f.write(f"Total variables: **{len(df_inv)}** across {df_inv['category'].nunique()} categories.\n\n")
        for cat in df_inv["category"].drop_duplicates().tolist():
            sub = df_inv[df_inv["category"] == cat]
            f.write(f"## {cat}  ({len(sub)} variables)\n\n")
            f.write("| name | type | direction | in_pure | range (q25–med–q75) | notes |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in sub.itertuples():
                rng = (
                    f"{r.q25}…{r.median}…{r.q75}"
                    if hasattr(r, "median") and pd.notna(getattr(r, "median", None))
                    else "—"
                )
                notes = (r.notes or "").replace("\n", " ")
                f.write(f"| `{r.name}` | {r.type} | {r.direction} | {r.in_pure} | {rng} | {notes} |\n")
            f.write("\n")

    print(f"\n✅ Inventory saved:\n  CSV: {out}\n  MD:  {md_out}\n")
    print("=" * 100)
    print(f"  Total variables: {len(df_inv)}")
    print(f"  By category: {df_inv['category'].value_counts().to_dict()}")
    print(f"  Currently in Pure: {(df_inv['in_pure'].str.contains('YES|yes', case=False, na=False)).sum()}")
    print(f"  Derivable but NOT in Pure: {(df_inv['in_pure'].str.contains('derivable', na=False)).sum()}")
    print(f"  In Gem only: {(df_inv['in_pure'].str.contains('Gem', case=False, na=False)).sum()}")
    print("=" * 100)


if __name__ == "__main__":
    run()
