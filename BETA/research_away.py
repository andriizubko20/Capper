"""
BETA/research_away.py

Deep research: away favorites in odds range 1.7-2.5.
When a strong team plays away and is STILL a market favorite —
this is a special signal worth exploring.

Also tests: home underdogs with xG edge (odds 2.0-3.5).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix

print("Loading data...")
matches, stats, odds, injuries = load_all()
df = build_feature_matrix(matches, stats, odds, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win'] = (df['result'] == 'A').astype(int)
df['draw']     = (df['result'] == 'D').astype(int)

print(f"Dataset: {len(df)} matches | home WR={df['home_win'].mean():.1%} away WR={df['away_win'].mean():.1%}")

def analyze(mask, label, side='away'):
    sub = df[mask]
    n = len(sub)
    if n < 20:
        return None
    if side == 'away':
        wr      = sub['away_win'].mean()
        avg_o   = sub['away_odds_val'].mean()
        flat    = (sub['away_win'] * (sub['away_odds_val'] - 1) - (1 - sub['away_win'])).mean()
    elif side == 'home':
        wr      = sub['home_win'].mean()
        avg_o   = sub['home_odds_val'].mean()
        flat    = (sub['home_win'] * (sub['home_odds_val'] - 1) - (1 - sub['home_win'])).mean()
    else:  # draw
        wr      = sub['draw'].mean()
        avg_o   = sub['draw_odds_val'].mean()
        flat    = (sub['draw'] * (sub['draw_odds_val'] - 1) - (1 - sub['draw'])).mean()

    # Kelly ROI simulation (25% fractional, 4% cap)
    bankroll = 1000.0
    for _, row in sub.iterrows():
        if side == 'away':
            o   = row['away_odds_val']
            won = row['result'] == 'A'
            p   = wr  # use observed WR as proxy prob
        elif side == 'home':
            o   = row['home_odds_val']
            won = row['result'] == 'H'
            p   = wr
        else:
            o   = row['draw_odds_val']
            won = row['result'] == 'D'
            p   = wr
        b = o - 1
        f = max(0.0, (p * b - (1 - p)) / b) * 0.25
        stake = min(bankroll * f, bankroll * 0.04)
        bankroll += stake * b if won else -stake
        bankroll = max(bankroll, 1.0)

    return {
        'label': label, 'side': side, 'n': n,
        'win_rate': wr,
        'avg_odds': avg_o,
        'flat_roi': flat,
        'kelly_final': round(bankroll, 0),
        'kelly_x': round(bankroll / 1000.0, 2),
    }

def pr(r):
    if r is None: return
    star = ' ★★' if r['win_rate'] >= 0.60 and r['avg_odds'] >= 1.65 else \
           ' ★'  if r['win_rate'] >= 0.58 else ''
    print(f"  {r['label']:55s} n={r['n']:4d} wr={r['win_rate']:.1%} "
          f"odds={r['avg_odds']:.2f} flat={r['flat_roi']:+.3f} "
          f"kelly=${r['kelly_final']:>6.0f} ({r['kelly_x']:.2f}x){star}")

results = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE: away team odds ranges
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 1. AWAY ODDS BASELINE ===")
for lo, hi in [(1.3,1.5),(1.5,1.7),(1.7,2.0),(2.0,2.5),(2.5,3.0),(3.0,4.0)]:
    mask = (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi)
    r = analyze(mask, f"away_odds [{lo},{hi})", 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 2. AWAY FAVORITE: odds 1.7-2.5 + Elo filters
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. AWAY FAVORITE [1.7-2.5] + ELO ===")
base_away = (df['away_odds_val'] >= 1.7) & (df['away_odds_val'] <= 2.5)
for thr in [-50, -75, -100, -125, -150]:
    mask = base_away & (df['elo_diff'] <= thr)
    r = analyze(mask, f"away_odds[1.7-2.5] + elo<={thr}", 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 3. AWAY FAVORITE: odds 1.7-2.5 + xG
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. AWAY FAVORITE [1.7-2.5] + xG ===")
for col, thr in [('xg_ratio_away_5',1.2),('xg_ratio_away_5',1.4),
                 ('xg_ratio_away_5',1.6),('xg_ratio_away_10',1.3)]:
    if col not in df.columns: continue
    mask = base_away & (df[col].fillna(0) >= thr)
    r = analyze(mask, f"away_odds[1.7-2.5] + {col}>={thr}", 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 4. AWAY FAVORITE: odds 1.7-2.5 + form
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. AWAY FAVORITE [1.7-2.5] + FORM ===")
for col, thr in [('away_pts_5',1.8),('away_pts_5',2.0),('away_pts_5',2.2),
                 ('away_pts_a5',1.8),('away_pts_a5',2.0),
                 ('home_pts_5',1.2),('home_pts_5',1.0)]:
    if col not in df.columns: continue
    mask = base_away & (df[col].fillna(0) >= thr if 'away' in col
                         else df[col].fillna(3) <= thr)
    r = analyze(mask, f"away_odds[1.7-2.5] + {col} {'>' if 'away' in col else '<'}={thr}", 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 5. AWAY FAVORITE: odds 1.7-2.5 + market prob
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 5. AWAY FAVORITE [1.7-2.5] + MARKET ===")
for thr in [0.35, 0.40, 0.45, 0.50]:
    mask = base_away & (df['mkt_away_prob'] >= thr)
    r = analyze(mask, f"away_odds[1.7-2.5] + mkt_away>={thr}", 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 6. AWAY FAVORITE: COMBINATIONS [1.7-2.5]
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 6. AWAY FAVORITE COMBINATIONS [1.7-2.5] ===")
combos = [
    (base_away & (df['elo_diff'] <= -75) & (df['mkt_away_prob'] >= 0.40),
     "away[1.7-2.5] + elo<=-75 + mkt>=0.40"),
    (base_away & (df['elo_diff'] <= -75) & (df['xg_ratio_away_5'].fillna(0) >= 1.3),
     "away[1.7-2.5] + elo<=-75 + xg_away>=1.3"),
    (base_away & (df['elo_diff'] <= -50) & (df['away_pts_5'].fillna(0) >= 1.8),
     "away[1.7-2.5] + elo<=-50 + away_form>=1.8"),
    (base_away & (df['elo_diff'] <= -75) & (df['away_pts_5'].fillna(0) >= 1.8),
     "away[1.7-2.5] + elo<=-75 + away_form>=1.8"),
    (base_away & (df['elo_diff'] <= -50) & (df['home_pts_5'].fillna(3) <= 1.2),
     "away[1.7-2.5] + elo<=-50 + home_poor<=1.2"),
    (base_away & (df['elo_diff'] <= -75) & (df['home_pts_5'].fillna(3) <= 1.5),
     "away[1.7-2.5] + elo<=-75 + home_poor<=1.5"),
    (base_away & (df['mkt_away_prob'] >= 0.45) & (df['xg_ratio_away_5'].fillna(0) >= 1.3),
     "away[1.7-2.5] + mkt>=0.45 + xg>=1.3"),
    (base_away & (df['mkt_away_prob'] >= 0.45) & (df['elo_diff'] <= -50),
     "away[1.7-2.5] + mkt>=0.45 + elo<=-50"),
    (base_away & (df['elo_diff'] <= -75) & (df['away_pts_5'].fillna(0) >= 1.8)
               & (df['home_pts_5'].fillna(3) <= 1.5),
     "away[1.7-2.5] + elo<=-75 + good_away + poor_home"),
    (base_away & (df['elo_diff'] <= -100) & (df['mkt_away_prob'] >= 0.40),
     "away[1.7-2.5] + elo<=-100 + mkt>=0.40"),
    (base_away & (df['elo_diff'] <= -50) & (df['xg_ratio_away_5'].fillna(0) >= 1.3)
               & (df['mkt_away_prob'] >= 0.38),
     "away[1.7-2.5] + elo<=-50 + xg>=1.3 + mkt>=0.38"),
    (base_away & (df['table_pts_diff'] <= -5) & (df['mkt_away_prob'] >= 0.40),
     "away[1.7-2.5] + away_more_pts + mkt>=0.40"),
    (base_away & (df['table_pts_diff'] <= -10) & (df['elo_diff'] <= -50),
     "away[1.7-2.5] + table_pts>10 + elo<=-50"),
]
for mask, label in combos:
    r = analyze(mask, label, 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Wider odds range 1.7-3.0 — bigger payouts
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 7. WIDER AWAY RANGE [1.7-3.0] ===")
base_wide = (df['away_odds_val'] >= 1.7) & (df['away_odds_val'] <= 3.0)
combos_wide = [
    (base_wide & (df['elo_diff'] <= -100), "away[1.7-3.0] + elo<=-100"),
    (base_wide & (df['elo_diff'] <= -100) & (df['mkt_away_prob'] >= 0.35),
     "away[1.7-3.0] + elo<=-100 + mkt>=0.35"),
    (base_wide & (df['xg_ratio_away_5'].fillna(0) >= 1.4) & (df['elo_diff'] <= -50),
     "away[1.7-3.0] + xg>=1.4 + elo<=-50"),
    (base_wide & (df['away_pts_5'].fillna(0) >= 2.0) & (df['elo_diff'] <= -75),
     "away[1.7-3.0] + form>=2.0 + elo<=-75"),
]
for mask, label in combos_wide:
    r = analyze(mask, label, 'away')
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 8. HOME FAVORITE odds 1.7-2.5 — strong home team but market still gives value
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 8. HOME FAVORITE [1.7-2.5] — VALUE RANGE ===")
base_hval = (df['home_odds_val'] >= 1.7) & (df['home_odds_val'] <= 2.5)
combos_hval = [
    (base_hval & (df['elo_diff'] >= 50), "home[1.7-2.5] + elo>=50", 'home'),
    (base_hval & (df['elo_diff'] >= 75), "home[1.7-2.5] + elo>=75", 'home'),
    (base_hval & (df['elo_diff'] >= 100), "home[1.7-2.5] + elo>=100", 'home'),
    (base_hval & (df['xg_ratio_home_5'].fillna(0) >= 1.3),
     "home[1.7-2.5] + xg>=1.3", 'home'),
    (base_hval & (df['xg_ratio_home_5'].fillna(0) >= 1.5),
     "home[1.7-2.5] + xg>=1.5", 'home'),
    (base_hval & (df['mkt_home_prob'] >= 0.50),
     "home[1.7-2.5] + mkt>=0.50", 'home'),
    (base_hval & (df['mkt_home_prob'] >= 0.55),
     "home[1.7-2.5] + mkt>=0.55", 'home'),
    (base_hval & (df['elo_diff'] >= 50) & (df['xg_ratio_home_5'].fillna(0) >= 1.3),
     "home[1.7-2.5] + elo>=50 + xg>=1.3", 'home'),
    (base_hval & (df['elo_diff'] >= 75) & (df['mkt_home_prob'] >= 0.50),
     "home[1.7-2.5] + elo>=75 + mkt>=0.50", 'home'),
    (base_hval & (df['elo_diff'] >= 50) & (df['home_pts_5'].fillna(0) >= 1.8),
     "home[1.7-2.5] + elo>=50 + form>=1.8", 'home'),
    (base_hval & (df['elo_diff'] >= 75) & (df['home_pts_h5'].fillna(0) >= 1.8),
     "home[1.7-2.5] + elo>=75 + home_home_form>=1.8", 'home'),
    (base_hval & (df['mkt_home_prob'] >= 0.50) & (df['xg_ratio_home_5'].fillna(0) >= 1.3)
               & (df['elo_diff'] >= 50),
     "home[1.7-2.5] + mkt>=0.50 + xg>=1.3 + elo>=50", 'home'),
    (base_hval & (df['home_pts_h5'].fillna(0) >= 2.0) & (df['away_pts_5'].fillna(3) <= 1.3),
     "home[1.7-2.5] + strong_home_form + poor_away", 'home'),
]
for mask, label, side in combos_hval:
    r = analyze(mask, label, side)
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# 9. HOME UNDERDOG edge (2.0-3.5) — surprise wins with xG edge
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 9. HOME UNDERDOG [2.0-3.5] WITH xG EDGE ===")
base_hund = (df['home_odds_val'] >= 2.0) & (df['home_odds_val'] <= 3.5)
combos_hund = [
    (base_hund & (df['xg_ratio_home_5'].fillna(0) >= 1.3),
     "home[2.0-3.5] + xg_home>=1.3", 'home'),
    (base_hund & (df['xg_ratio_home_5'].fillna(0) >= 1.5),
     "home[2.0-3.5] + xg_home>=1.5", 'home'),
    (base_hund & (df['home_pts_h5'].fillna(0) >= 2.0) & (df['xg_ratio_home_5'].fillna(0) >= 1.2),
     "home[2.0-3.5] + home_form>=2.0 + xg>=1.2", 'home'),
    (base_hund & (df['elo_diff'] >= 0) & (df['xg_ratio_home_5'].fillna(0) >= 1.3),
     "home[2.0-3.5] + elo_diff>=0 + xg>=1.3", 'home'),
]
for mask, label, side in combos_hund:
    r = analyze(mask, label, side)
    if r: results.append(r); pr(r)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("TOP by KELLY FINAL (min 40 bets, min odds 1.60)")
print("="*80)
top = [r for r in results if r and r['n'] >= 40 and r['avg_odds'] >= 1.60]
top.sort(key=lambda x: -x['kelly_x'])
print(f"{'Filter':55s} {'n':>5} {'WR':>6} {'Odds':>6} {'Flat':>7} {'Kelly$':>8} {'x':>5}")
print("-"*80)
for r in top[:20]:
    star = ' ★★' if r['kelly_x'] >= 2.0 else ' ★' if r['kelly_x'] >= 1.5 else ''
    print(f"{r['label']:55s} {r['n']:>5} {r['win_rate']:>5.1%} "
          f"{r['avg_odds']:>6.2f} {r['flat_roi']:>+6.3f} "
          f"${r['kelly_final']:>7.0f} {r['kelly_x']:>5.2f}x{star}")

print("\n" + "="*80)
print("TOP by WIN RATE >= 58% with odds >= 1.65")
print("="*80)
top2 = [r for r in results if r and r['n'] >= 40
        and r['win_rate'] >= 0.58 and r['avg_odds'] >= 1.65]
top2.sort(key=lambda x: -(x['win_rate'] * x['avg_odds']))
for r in top2[:15]:
    print(f"{r['label']:55s} {r['n']:>5} {r['win_rate']:>5.1%} "
          f"{r['avg_odds']:>6.2f} {r['flat_roi']:>+6.3f} "
          f"${r['kelly_final']:>7.0f} {r['kelly_x']:>5.2f}x")
