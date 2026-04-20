"""
BETA/research.py

Empirical research: find filter combinations where real win rate >= 60%.
No ML — pure data analysis over historical matches.

We test:
  - Elo thresholds
  - xG ratios
  - Form filters
  - Odds ranges
  - Table position gaps
  - Combinations of the above

Output: ranked table of filters by win rate + sample size + ROI.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
matches, stats, odds, injuries = load_all()
df = build_feature_matrix(matches, stats, odds, injuries)

# Only rows with odds (we need them to simulate bets)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win'] = (df['result'] == 'A').astype(int)
df['draw']     = (df['result'] == 'D').astype(int)

print(f"Dataset: {len(df)} matches with odds")
print(f"Overall home win rate: {df['home_win'].mean():.1%}")
print(f"Overall away win rate: {df['away_win'].mean():.1%}")
print(f"Overall draw rate:     {df['draw'].mean():.1%}")
print()

# ── Helper ─────────────────────────────────────────────────────────────────────

def analyze(mask, label, side='home'):
    sub = df[mask]
    n = len(sub)
    if n < 30:
        return None
    if side == 'home':
        wr = sub['home_win'].mean()
        avg_odds = sub['home_odds_val'].mean()
        # Simple flat-stake ROI
        roi = (sub['home_win'] * (sub['home_odds_val'] - 1) - (1 - sub['home_win'])).mean()
    else:
        wr = sub['away_win'].mean()
        avg_odds = sub['away_odds_val'].mean()
        roi = (sub['away_win'] * (sub['away_odds_val'] - 1) - (1 - sub['away_win'])).mean()
    return {'label': label, 'side': side, 'n': n, 'win_rate': wr,
            'avg_odds': avg_odds, 'flat_roi': roi}


results = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline: by odds range only
# ─────────────────────────────────────────────────────────────────────────────
print("=== 1. ODDS RANGE ===")
for lo, hi in [(1.20,1.40),(1.40,1.55),(1.55,1.70),(1.70,1.85),(1.85,2.10),(2.10,2.50)]:
    mask = (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi)
    r = analyze(mask, f"home_odds [{lo},{hi})", 'home')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Elo gap filters (home advantage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. ELO GAP ===")
for threshold in [30, 50, 75, 100, 150]:
    mask = df['elo_diff'] >= threshold
    r = analyze(mask, f"elo_diff >= {threshold}", 'home')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

for threshold in [-50, -75, -100, -150]:
    mask = df['elo_diff'] <= threshold
    r = analyze(mask, f"elo_diff <= {threshold}", 'away')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. xG ratio filters
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. xG RATIO ===")
for col, side, thr in [
    ('xg_ratio_home_5', 'home', 1.3),
    ('xg_ratio_home_5', 'home', 1.5),
    ('xg_ratio_home_5', 'home', 1.8),
    ('xg_ratio_away_5', 'away', 1.3),
    ('xg_ratio_away_5', 'away', 1.5),
]:
    if col not in df.columns: continue
    mask = df[col] >= thr
    r = analyze(mask, f"{col} >= {thr}", side)
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Form filters (pts per game)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. FORM ===")
for col, side, thr in [
    ('home_pts_5', 'home', 2.0),
    ('home_pts_5', 'home', 2.3),
    ('away_pts_5', 'away', 2.0),
    ('home_pts_h5', 'home', 2.0),   # home form at home
    ('home_pts_h5', 'home', 2.3),
]:
    if col not in df.columns: continue
    mask = df[col] >= thr
    r = analyze(mask, f"{col} >= {thr}", side)
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Table position gap
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 5. TABLE POSITION ===")
for thr in [-5, -8, -10]:
    mask = df['table_pos_diff'] <= thr   # home higher (lower number = better)
    r = analyze(mask, f"table_pos_diff <= {thr} (home higher)", 'home')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

for thr in [5, 8, 10]:
    mask = df['table_pts_diff'] >= thr
    r = analyze(mask, f"table_pts_diff >= {thr} (home more pts)", 'home')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Market confirmation (market agrees with model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 6. MARKET PROB ===")
for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = df['mkt_home_prob'] >= thr
    r = analyze(mask, f"mkt_home_prob >= {thr}", 'home')
    if r: results.append(r); print(f"  {r['label']:35s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. COMBINATIONS — the good stuff
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 7. COMBINATIONS ===")

combos = [
    # (mask_expr, label, side)
    (
        (df['elo_diff'] >= 50) & (df['mkt_home_prob'] >= 0.55) & (df['home_odds_val'] >= 1.5),
        "elo>=50 + mkt_home>=0.55 + odds>=1.5", "home"
    ),
    (
        (df['elo_diff'] >= 75) & (df['mkt_home_prob'] >= 0.55),
        "elo>=75 + mkt_home>=0.55", "home"
    ),
    (
        (df['elo_diff'] >= 50) & (df['home_pts_5'].fillna(0) >= 1.8) & (df['away_pts_5'].fillna(3) <= 1.2),
        "elo>=50 + home_form>=1.8 + away_form<=1.2", "home"
    ),
    (
        (df['mkt_home_prob'] >= 0.60) & (df['home_odds_val'] >= 1.5) & (df['home_odds_val'] <= 1.9),
        "mkt_home>=0.60 + odds[1.5,1.9]", "home"
    ),
    (
        (df['mkt_home_prob'] >= 0.55)
        & (df['elo_diff'] >= 30)
        & (df['xg_ratio_home_5'].fillna(0) >= 1.2),
        "mkt>=0.55 + elo>=30 + xg_ratio>=1.2", "home"
    ),
    (
        (df['mkt_home_prob'] >= 0.60)
        & (df['elo_diff'] >= 50)
        & (df['home_pts_h5'].fillna(0) >= 1.5),
        "mkt>=0.60 + elo>=50 + home_home_form>=1.5", "home"
    ),
    (
        (df['mkt_home_prob'] >= 0.65)
        & (df['elo_diff'] >= 75),
        "mkt>=0.65 + elo>=75", "home"
    ),
    (
        (df['elo_diff'] >= 100)
        & (df['table_pts_diff'] >= 5),
        "elo>=100 + table_pts_diff>=5", "home"
    ),
    (
        (df['mkt_home_prob'] >= 0.55)
        & (df['home_pts_5'].fillna(0) >= 1.8)
        & (df['home_pts_h5'].fillna(0) >= 1.8)
        & (df['away_pts_5'].fillna(3) <= 1.5),
        "mkt>=0.55 + strong_home_form + poor_away", "home"
    ),
    (
        (df['elo_diff'] >= 50)
        & (df['xg_ratio_home_5'].fillna(0) >= 1.3)
        & (df['xg_ratio_away_5'].fillna(2) <= 0.9),
        "elo>=50 + xg_h>=1.3 + xg_a<=0.9", "home"
    ),
    # Away combos
    (
        (df['elo_diff'] <= -75) & (df['mkt_away_prob'] >= 0.40),
        "elo<=-75 + mkt_away>=0.40", "away"
    ),
    (
        (df['elo_diff'] <= -50)
        & (df['away_pts_5'].fillna(0) >= 2.0)
        & (df['home_pts_5'].fillna(3) <= 1.2),
        "elo<=-50 + away_form>=2.0 + home_poor", "away"
    ),
]

for mask, label, side in combos:
    r = analyze(mask, label, side)
    if r:
        results.append(r)
        marker = " ★" if r['win_rate'] >= 0.60 else ""
        print(f"  {label:50s} n={r['n']:4d} wr={r['win_rate']:.1%} roi={r['flat_roi']:+.3f}{marker}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary: top filters by win rate (min 50 samples)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP FILTERS (win_rate >= 55%, n >= 50)")
print("="*70)
top = [r for r in results if r and r['win_rate'] >= 0.55 and r['n'] >= 50]
top.sort(key=lambda x: -x['win_rate'])
print(f"{'Filter':55s} {'n':>5} {'WR':>6} {'Odds':>6} {'ROI':>7}")
print("-"*70)
for r in top:
    print(f"{r['label']:55s} {r['n']:>5} {r['win_rate']:>5.1%} {r['avg_odds']:>6.2f} {r['flat_roi']:>+6.3f}")

print("\n" + "="*70)
print("TOP BY FLAT ROI (n >= 50)")
print("="*70)
top_roi = [r for r in results if r and r['n'] >= 50]
top_roi.sort(key=lambda x: -x['flat_roi'])
for r in top_roi[:15]:
    print(f"{r['label']:55s} {r['n']:>5} {r['win_rate']:>5.1%} {r['avg_odds']:>6.2f} {r['flat_roi']:>+6.3f}")
