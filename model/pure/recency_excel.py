"""
model/pure/recency_excel.py

Re-evaluate ALL niches from profitable_niches.xlsx on RECENT data
(default since 2025-08-01) and add columns:
  recent_n, recent_wr, recent_lo95, drift, recent_verdict

Result: profitable_niches_recent.xlsx with same per-league sheets +
the recency columns. Sorted by recent ROI.

Usage:
  python -m model.pure.recency_excel [--cutoff 2025-08-01]
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from model.pure.compute_pis import evaluate_niche

REPORTS = Path(__file__).parent / "reports"

# Map compact label → niche dict key
LABEL_MAP = {
    "g":   "min_glicko_gap",
    "p":   "min_glicko_prob",
    "xd":  "min_xg_diff",
    "xq":  "min_xg_quality_gap",
    "ad":  "min_attack_vs_def",
    "f":   "min_form_advantage",
    "ppg": "min_ppg",
    "xt":  "min_xg_trend",
    "gm":  "min_glicko_momentum",
    "ws":  "min_win_streak",
    "ols": "min_opp_lose_streak",
    "pos": "min_possession_10",
    "sot": "min_sot_10",
    "pa":  "min_pass_acc_10",
    "ra":  "min_rest_advantage",
    "h2h": "min_h2h_wr",
    "m":   "max_market_prob",
}

NICHE_RE = re.compile(r"^(home|away)\[([\d.]+),([\d.]+)\)(?:\s+(.+))?$")


def parse_niche_id(s: str, league: str) -> dict | None:
    s = s.strip()
    m = NICHE_RE.match(s)
    if not m:
        return None
    side, lo, hi, rest = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4) or ""
    out = {
        "side": side,
        "odds_range": (lo, hi),
        "_league": league,
        "niche_id": s,
    }
    for token in rest.split():
        mt = re.match(r"([a-z][a-z0-9]*)([<>]=)([\d.]+)", token)
        if not mt:
            continue
        label, op, val = mt.group(1), mt.group(2), float(mt.group(3))
        col = LABEL_MAP.get(label)
        if col is None:
            continue
        # Normalise pa, pos: scale 0-100 → 0-1 for storage
        if label in ("pa", "pos") and val > 1.5:
            val = val / 100.0
        out[col] = val
    return out


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def evaluate(df: pd.DataFrame, niche: dict) -> dict:
    return evaluate_niche(df, niche)


def verdict(full_p, recent_n, recent_p, recent_lo95, drift) -> str:
    if recent_n < 5 or recent_p is None:
        return "🟡 small n"
    if drift is None:
        return "?"
    if drift < -0.15 and recent_lo95 < 0.55:
        return "🔴 DROP — decayed"
    if drift < -0.05 and recent_lo95 < 0.55:
        return "🟠 watch — weakening"
    if recent_p >= (full_p or 0) - 0.03:
        return "🟢 stable"
    return "🟡 mild drift"


def run(cutoff: str = "2025-08-01") -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    src = REPORTS / "profitable_niches.xlsx"
    if not src.exists():
        logger.error(f"Source missing: {src}")
        return

    df_full   = pd.read_parquet(REPORTS / "match_factors.parquet")
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_recent = df_full[df_full["date"] >= cutoff].reset_index(drop=True)
    logger.info(f"Full {len(df_full):,} matches, recent {len(df_recent):,} matches")

    xl = pd.ExcelFile(src)
    out_sheets: dict[str, pd.DataFrame] = {}

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if df.empty or "niche_id" not in df.columns:
            out_sheets[sheet_name] = df
            continue

        # Compute league per row (Summary sheet has 'league' col already; per-league sheets use sheet name)
        rows = []
        for r in df.itertuples():
            league = getattr(r, "league", None) or sheet_name
            niche = parse_niche_id(r.niche_id, league)
            if niche is None:
                rows.append({"recent_n": 0, "recent_wr": None, "recent_lo95": 0,
                             "drift": None, "recent_verdict": "❌ parse"})
                continue
            full_p = getattr(r, "wr", None)
            recent = evaluate(df_recent, niche)
            recent_p = recent.get("p_is")
            recent_lo = recent.get("p_is_lower_95", 0.0)
            drift = (recent_p - full_p) if (full_p is not None and recent_p is not None) else None
            v = verdict(full_p, recent.get("n", 0), recent_p, recent_lo, drift)
            rows.append({
                "recent_n":      recent.get("n", 0),
                "recent_wr":     round(recent_p, 4) if recent_p is not None else None,
                "recent_lo95":   round(recent_lo, 4),
                "drift":         round(drift, 4) if drift is not None else None,
                "recent_verdict": v,
            })

        df = df.copy()
        for col in ["recent_n", "recent_wr", "recent_lo95", "drift", "recent_verdict"]:
            df[col] = [r[col] for r in rows]

        # Move new columns near 'wr'
        cols = list(df.columns)
        wr_idx = cols.index("wr") if "wr" in cols else 0
        new = ["recent_n", "recent_wr", "recent_lo95", "drift", "recent_verdict"]
        for c in new:
            cols.remove(c)
        cols[wr_idx + 1: wr_idx + 1] = new
        df = df[cols]
        out_sheets[sheet_name] = df

    # Write
    out_path = REPORTS / "profitable_niches_recent.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for name, d in out_sheets.items():
            d.to_excel(w, sheet_name=name, index=False)

    logger.info(f"Saved → {out_path}")

    # Summary across all sheets
    print("\n" + "=" * 60)
    print(f"OVERALL VERDICT DISTRIBUTION  (cutoff: {cutoff})")
    print("=" * 60)
    summary = (
        out_sheets["Summary_Top100"][["league", "side", "niche_id", "n", "wr", "roi",
                                       "recent_n", "recent_wr", "drift", "recent_verdict"]]
        if "Summary_Top100" in out_sheets else None
    )
    if summary is not None:
        print(summary["recent_verdict"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=str, default="2025-08-01")
    args = ap.parse_args()
    run(args.cutoff)
