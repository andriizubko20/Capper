"""
model/pure/niches.py

Pure niche grammar — richer than Monster's. A niche is a discrete pattern
with hard threshold conditions; matching a niche means the pre-match factors
agree the side is a clear (often undervalued) favorite.

Grammar (dict form, no string parsing):
  {
    "side":             "home" | "away",
    "odds_range":       (lo, hi),                # the side's odds must lie in [lo, hi]
    "min_glicko_gap":   float | None,            # side_glicko - other_glicko ≥ this
    "min_xg_diff":      float | None,            # side.xg_for_10 - side.xg_against_10 ≥ this
    "min_attack_vs_def": float | None,           # side.xg_for_10 - other.xg_against_10 ≥ this
    "min_form_advantage": float | None,          # side.ppg_10 - other.ppg_10 ≥ this
    "min_glicko_prob":  float | None,            # SStats Glicko win prob ≥ this
    "max_market_prob":  float | None,            # de-vigged market prob ≤ this (mispricing)
  }

A match QUALIFIES under a niche iff every non-None condition is satisfied.
"""
from typing import Optional


# ── Grid space for niche generation ──────────────────────────────────────────
# We pick a coarse-but-meaningful grid; combinations explode quickly so each
# axis has only ~3-4 values plus None (= condition not used).
GRID = {
    "odds_range": [
        (1.40, 1.60), (1.55, 1.85), (1.70, 2.10), (1.85, 2.40), (2.10, 3.00),
    ],
    "min_glicko_gap":     [None, 60, 100, 150],
    "min_xg_diff":        [None, 0.20, 0.40, 0.60],
    "min_attack_vs_def":  [None, 0.30, 0.50, 0.70],
    "min_form_advantage": [None, 0.5, 1.0, 1.5],
    "min_glicko_prob":    [None, 0.55, 0.60, 0.65, 0.70],
    "max_market_prob":    [None, 0.55, 0.60, 0.65, 0.70],
}


def niche_id(niche: dict) -> str:
    """Compact, unique string id for a niche (used as key in storage)."""
    side = niche["side"]
    lo, hi = niche["odds_range"]
    parts = [f"{side}[{lo},{hi})"]
    for key, label in [
        ("min_glicko_gap",     "g"),
        ("min_xg_diff",        "xd"),
        ("min_attack_vs_def",  "ad"),
        ("min_form_advantage", "f"),
        ("min_glicko_prob",    "p"),
        ("max_market_prob",    "m"),
    ]:
        v = niche.get(key)
        if v is not None:
            if key == "max_market_prob":
                parts.append(f"{label}<={v}")
            else:
                parts.append(f"{label}>={v}")
    return " ".join(parts)


def matches_niche(features: dict, niche: dict, side: str) -> bool:
    """
    Check if a feature dict (from one side's perspective) matches a niche.

    `features` keys (numeric, may be None):
      odds, glicko_gap, xg_diff, attack_vs_def, form_advantage,
      glicko_prob, market_prob
    """
    if niche["side"] != side:
        return False
    odds = features.get("odds")
    if odds is None:
        return False
    lo, hi = niche["odds_range"]
    if odds < lo or odds > hi:
        return False

    def _ge(key: str, niche_key: str) -> bool:
        thr = niche.get(niche_key)
        if thr is None:
            return True
        v = features.get(key)
        return v is not None and v >= thr

    def _le(key: str, niche_key: str) -> bool:
        thr = niche.get(niche_key)
        if thr is None:
            return True
        v = features.get(key)
        return v is not None and v <= thr

    if not _ge("glicko_gap",      "min_glicko_gap"):       return False
    if not _ge("xg_diff",         "min_xg_diff"):          return False
    if not _ge("attack_vs_def",   "min_attack_vs_def"):    return False
    if not _ge("form_advantage",  "min_form_advantage"):   return False
    if not _ge("glicko_prob",     "min_glicko_prob"):      return False
    if not _le("market_prob",     "max_market_prob"):      return False
    return True


def generate_candidate_niches(side: str) -> list[dict]:
    """Cartesian product over GRID — each cell becomes a niche dict."""
    import itertools
    keys = ["odds_range", "min_glicko_gap", "min_xg_diff", "min_attack_vs_def",
            "min_form_advantage", "min_glicko_prob", "max_market_prob"]
    values = [GRID[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        niche = {"side": side}
        for k, v in zip(keys, combo):
            niche[k] = v
        out.append(niche)
    return out


def features_for_side(row: dict, side: str) -> dict:
    """Extract per-side features from a match-level row dict."""
    other = "away" if side == "home" else "home"
    home_glicko_prob = row.get("home_glicko_prob")
    away_glicko_prob = row.get("away_glicko_prob")
    home_market_prob = row.get("home_market_prob")
    away_market_prob = row.get("away_market_prob")
    if side == "home":
        return {
            "odds":            row.get("home_odds"),
            "glicko_gap":      row.get("glicko_gap"),
            "xg_diff":         row.get("xg_diff_home"),
            "attack_vs_def":   row.get("attack_vs_def_home"),
            "form_advantage":  row.get("form_advantage"),
            "glicko_prob":     home_glicko_prob,
            "market_prob":     home_market_prob,
        }
    return {
        "odds":            row.get("away_odds"),
        "glicko_gap":      -row["glicko_gap"] if row.get("glicko_gap") is not None else None,
        "xg_diff":         row.get("xg_diff_away"),
        "attack_vs_def":   row.get("attack_vs_def_away"),
        "form_advantage":  -row["form_advantage"] if row.get("form_advantage") is not None else None,
        "glicko_prob":     away_glicko_prob,
        "market_prob":     away_market_prob,
    }


SIDES = ("home", "away")
