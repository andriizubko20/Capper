"""
model/pure/features.py

Build a wide feature set per match (HOME perspective). All features are
PRE-match (use Gem's leakage-free team_state pipeline).

Two feature dicts produced per match — one from HOME perspective, one from
AWAY perspective — so we can fit per-side logistic regressions separately.

Usage:
  from model.pure.features import build_pure_features
  df = build_pure_features()  # → DataFrame with one row per (match, side)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.gem.data import load_historical
from model.gem.team_state import build_h2h, build_team_state

REPORTS_DIR = Path(__file__).parent / "reports"

# Wide candidate feature list. We let logistic regression assign weights — if
# something is noise, its coefficient shrinks toward 0 with L2 reg.
SIDE_FEATURES = [
    # Strength / quality (the "is favourite" signal)
    "glicko_gap",            # side_glicko - other_glicko
    "glicko_prob",           # SStats pre-match Glicko prob for side
    "glicko_prob_diff",      # glicko_prob - market_prob (mispricing signal)
    # Attack / defense (xG)
    "xg_for_10",
    "xg_against_10",
    "xg_diff",               # for - against, this side
    "xg_quality_gap",        # (side.xg_for - other.xg_against) - (other.xg_for - side.xg_against)
    "attack_vs_def",         # side.xg_for - other.xg_against
    # Form & momentum
    "ppg_10",
    "ppg_advantage",         # side.ppg_10 - other.ppg_10
    "form_5",
    "form_advantage",        # side.form_5 - other.form_5
    "xg_trend",              # last-5 xG_for - last-10 xG_for (this side)
    "xg_trend_advantage",    # side.xg_trend - other.xg_trend
    "glicko_momentum",       # this side's glicko 5-match drift
    "glicko_momentum_diff",  # side.glicko_momentum - other.glicko_momentum
    # Streaks
    "win_streak",
    "lose_streak",
    "win_streak_diff",       # side.win_streak - other.lose_streak
    # Style
    "possession_10",
    "sot_10",
    "pass_acc_10",
    # Physical
    "rest_days",
    "rest_advantage",        # side.rest - other.rest
    # H2H
    "h2h_wr",                # side win rate vs opponent (last 5)
    "h2h_avg_goals",
    # League prior (bookmaker-independent baseline)
    "league_home_wr",        # historical home WR for this league (computed in calibrate)
    # Market context (used as feature here — the model learns to discount it,
    # not to follow it; final filter excludes leakage by chronological split)
    "market_prob",
    "odds",
    "implied_prob_unvigged",
]


def build_pure_features() -> pd.DataFrame:
    """
    Build long-format DataFrame: 2 rows per match (one per side).

    Each row contains ALL SIDE_FEATURES from that side's perspective, plus
    metadata (match_id, date, league, side, won) for backtest scoring.
    """
    logger.info("Loading historical data …")
    data = load_historical()
    matches = data["matches"]
    stats = data["stats"]
    odds = data["odds"]

    logger.info("Building team state + H2H …")
    ts = build_team_state(matches, stats)
    h2h_dict = build_h2h(matches)

    stats_idx = stats.set_index("match_id")
    odds_idx = odds.set_index("match_id")

    rows: list[dict] = []
    for m in matches.itertuples(index=False):
        mid = m.match_id
        h = ts.get((mid, "home"))
        a = ts.get((mid, "away"))
        if h is None or a is None:
            continue

        # Need basic features
        if h["glicko_now"] is None or a["glicko_now"] is None:
            continue
        if h["xg_for_10"] is None or a["xg_for_10"] is None:
            continue
        if h["ppg_10"] is None or a["ppg_10"] is None:
            continue

        # Match-level shared
        s = None
        if mid in stats_idx.index:
            s = stats_idx.loc[mid]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
        home_glicko_prob = s["home_win_prob"] if s is not None else None
        away_glicko_prob = s["away_win_prob"] if s is not None else None

        if mid not in odds_idx.index:
            continue
        o = odds_idx.loc[mid]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        home_odds = o.get("home_odds")
        draw_odds = o.get("draw_odds")
        away_odds = o.get("away_odds")
        if any(v is None for v in [home_odds, draw_odds, away_odds]):
            continue

        # De-vigged market probs
        raw_sum = 1 / home_odds + 1 / draw_odds + 1 / away_odds
        home_market_prob = (1 / home_odds) / raw_sum
        away_market_prob = (1 / away_odds) / raw_sum

        # H2H from home perspective
        h2h = h2h_dict.get(mid, {})
        h2h_home_wr = h2h.get("h2h_home_wr")
        h2h_avg_goals = h2h.get("h2h_avg_goals")

        # Compute rest days (days since last match)
        match_date = pd.Timestamp(m.date)
        home_rest = (match_date - h["last_match_date"]).days if h.get("last_match_date") is not None else None
        away_rest = (match_date - a["last_match_date"]).days if a.get("last_match_date") is not None else None

        # ── HOME side row ──────────────────────────────────────────────
        rows.append({
            "match_id": mid,
            "date": m.date,
            "league_name": m.league_name,
            "side": "home",
            "result": m.result,
            "won": int(m.result == "H"),
            "odds": home_odds,
            # Strength
            "glicko_gap":   h["glicko_now"] - a["glicko_now"],
            "glicko_prob":  home_glicko_prob,
            "market_prob":  home_market_prob,
            "implied_prob_unvigged": home_market_prob,
            "glicko_prob_diff": (home_glicko_prob - home_market_prob) if home_glicko_prob is not None else None,
            # xG
            "xg_for_10":     h["xg_for_10"],
            "xg_against_10": h["xg_against_10"],
            "xg_diff":       h["xg_for_10"] - h["xg_against_10"],
            "xg_quality_gap": (h["xg_for_10"] - a["xg_against_10"]) - (a["xg_for_10"] - h["xg_against_10"]),
            "attack_vs_def":  h["xg_for_10"] - a["xg_against_10"],
            # Form
            "ppg_10":        h["ppg_10"],
            "ppg_advantage": h["ppg_10"] - a["ppg_10"],
            "form_5":        h.get("form_5"),
            "form_advantage": (h.get("form_5", 0) or 0) - (a.get("form_5", 0) or 0),
            "xg_trend":      h.get("xg_trend"),
            "xg_trend_advantage": (h.get("xg_trend") or 0) - (a.get("xg_trend") or 0),
            "glicko_momentum": h.get("glicko_momentum"),
            "glicko_momentum_diff": (h.get("glicko_momentum") or 0) - (a.get("glicko_momentum") or 0),
            # Streaks
            "win_streak":  h.get("win_streak", 0),
            "lose_streak": h.get("lose_streak", 0),
            "win_streak_diff": (h.get("win_streak") or 0) - (a.get("lose_streak") or 0),
            # Style
            "possession_10": h.get("possession_10"),
            "sot_10":        h.get("sot_10"),
            "pass_acc_10":   h.get("pass_acc_10"),
            # Physical
            "rest_days":      home_rest,
            "rest_advantage": (home_rest - away_rest) if home_rest is not None and away_rest is not None else None,
            # H2H
            "h2h_wr":         h2h_home_wr,
            "h2h_avg_goals":  h2h_avg_goals,
        })

        # ── AWAY side row ──────────────────────────────────────────────
        rows.append({
            "match_id": mid,
            "date": m.date,
            "league_name": m.league_name,
            "side": "away",
            "result": m.result,
            "won": int(m.result == "A"),
            "odds": away_odds,
            "glicko_gap":   a["glicko_now"] - h["glicko_now"],
            "glicko_prob":  away_glicko_prob,
            "market_prob":  away_market_prob,
            "implied_prob_unvigged": away_market_prob,
            "glicko_prob_diff": (away_glicko_prob - away_market_prob) if away_glicko_prob is not None else None,
            "xg_for_10":     a["xg_for_10"],
            "xg_against_10": a["xg_against_10"],
            "xg_diff":       a["xg_for_10"] - a["xg_against_10"],
            "xg_quality_gap": (a["xg_for_10"] - h["xg_against_10"]) - (h["xg_for_10"] - a["xg_against_10"]),
            "attack_vs_def":  a["xg_for_10"] - h["xg_against_10"],
            "ppg_10":        a["ppg_10"],
            "ppg_advantage": a["ppg_10"] - h["ppg_10"],
            "form_5":        a.get("form_5"),
            "form_advantage": (a.get("form_5", 0) or 0) - (h.get("form_5", 0) or 0),
            "xg_trend":      a.get("xg_trend"),
            "xg_trend_advantage": (a.get("xg_trend") or 0) - (h.get("xg_trend") or 0),
            "glicko_momentum": a.get("glicko_momentum"),
            "glicko_momentum_diff": (a.get("glicko_momentum") or 0) - (h.get("glicko_momentum") or 0),
            "win_streak":  a.get("win_streak", 0),
            "lose_streak": a.get("lose_streak", 0),
            "win_streak_diff": (a.get("win_streak") or 0) - (h.get("lose_streak") or 0),
            "possession_10": a.get("possession_10"),
            "sot_10":        a.get("sot_10"),
            "pass_acc_10":   a.get("pass_acc_10"),
            "rest_days":      away_rest,
            "rest_advantage": (away_rest - home_rest) if home_rest is not None and away_rest is not None else None,
            "h2h_wr":         (1 - h2h_home_wr) if h2h_home_wr is not None else None,
            "h2h_avg_goals":  h2h_avg_goals,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Per-league baseline home/away WR (used as a feature)
    league_baseline = matches.groupby("league_name").apply(
        lambda x: pd.Series({
            "league_home_wr": (x["result"] == "H").mean(),
            "league_away_wr": (x["result"] == "A").mean(),
        }), include_groups=False,
    ).reset_index()

    df = df.merge(league_baseline, on="league_name", how="left")
    df["league_home_wr"] = np.where(df["side"] == "home", df["league_home_wr"], df["league_away_wr"])
    df = df.drop(columns=["league_away_wr"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "pure_features.parquet"
    df.to_parquet(out)
    logger.info(f"Pure feature matrix: {len(df):,} rows ({df['side'].value_counts().to_dict()}) → {out}")
    return df


if __name__ == "__main__":
    build_pure_features()
