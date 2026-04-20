"""
BETA/research_clusters.py

Systematic cluster research — знаходимо всі прибуткові ніші.

Методологія:
  1. Odds clusters: [1.30-1.55], [1.55-1.80], [1.80-2.20], [2.20-2.80], [2.80-4.00]
  2. Side: home / away
  3. Per cluster — тестуємо комбінації xG, Elo, Form, League фільтрів
  4. Kelly simulation для кожної ніші (compound bankroll)
  5. Output: рейтинг ніш за Kelly x-factor + WR + volume

Мета: знайти 3-5 незалежних прибуткових ніш для портфельної моделі.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
matches, stats, odds, injuries = load_all()
df = build_feature_matrix(matches, stats, odds, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win'] = (df['result'] == 'A').astype(int)
print(f"Dataset: {len(df)} matches | {df['date'].min().date()} – {df['date'].max().date()}")
print(f"Home WR: {df['home_win'].mean():.1%} | Away WR: {df['away_win'].mean():.1%}\n")

# ── Kelly simulation ──────────────────────────────────────────────────────────
def kelly_sim(sub, side, initial=1000.0, fractional=0.25, cap=0.06):
    """Simple flat Kelly simulation (no walk-forward, in-sample upper bound)."""
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    win_col  = 'home_win' if side == 'home' else 'away_win'
    wr = sub[win_col].mean()
    avg_odds = sub[odds_col].mean()
    b = avg_odds - 1
    q = 1 - wr
    f = max(0, (wr * b - q) / b) * fractional
    f = min(f, cap)
    # Geometric growth approximation
    x = (1 + f * b) ** wr * (1 - f) ** q
    n = len(sub)
    xfactor = x ** n
    return round(xfactor, 2), round(wr * 100, 1), round(avg_odds, 2), n

# ── Analyze helper ────────────────────────────────────────────────────────────
results = []

def analyze(mask, label, side, min_n=40):
    sub = df[mask]
    if len(sub) < min_n:
        return
    xf, wr, avg_odds, n = kelly_sim(sub, side)
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    win_col  = 'home_win' if side == 'home' else 'away_win'
    flat_roi = (sub[win_col] * (sub[odds_col] - 1) - (1 - sub[win_col])).mean()
    results.append({
        'label': label, 'side': side, 'n': n,
        'wr': wr, 'avg_odds': avg_odds,
        'flat_roi': round(flat_roi, 4),
        'kelly_xf': xf,
    })
    star = ' ★★' if wr >= 60 else (' ★' if wr >= 57 else '')
    print(f"  {label:60s} n={n:4d} wr={wr:5.1f}% odds={avg_odds:.2f} "
          f"roi={flat_roi:+.3f} kelly={xf:.2f}x{star}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. БАЗОВІ КЛАСТЕРИ — Odds range × Side (без додаткових фільтрів)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. БАЗОВІ КЛАСТЕРИ (odds range × side)")
print("=" * 80)

odds_clusters = [
    (1.30, 1.55, 'strong_fav'),
    (1.55, 1.80, 'mild_fav'),
    (1.80, 2.20, 'value_low'),
    (2.20, 2.80, 'value_high'),
    (2.80, 4.00, 'outsider'),
]

for lo, hi, name in odds_clusters:
    print(f"\n  [{lo}-{hi}] {name}")
    for side in ('home', 'away'):
        col = 'home_odds_val' if side == 'home' else 'away_odds_val'
        mask = (df[col] >= lo) & (df[col] < hi)
        analyze(mask, f"{side} odds[{lo},{hi})", side, min_n=30)

# ══════════════════════════════════════════════════════════════════════════════
# 2. xG ФІЛЬТРИ ВСЕРЕДИНІ КОЖНОГО КЛАСТЕРА
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. xG ФІЛЬТРИ × ODDS CLUSTER")
print("=" * 80)

xg_thresholds = [1.2, 1.5, 1.8]
for lo, hi, name in odds_clusters:
    print(f"\n  [{lo}-{hi}] {name}")
    for side in ('home', 'away'):
        col  = 'home_odds_val' if side == 'home' else 'away_odds_val'
        xcol = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
        for thr in xg_thresholds:
            if xcol not in df.columns:
                continue
            mask = (
                (df[col] >= lo) & (df[col] < hi) &
                (df[xcol].fillna(0) >= thr)
            )
            analyze(mask, f"{side}[{lo},{hi}) + xg>={thr}", side)

# ══════════════════════════════════════════════════════════════════════════════
# 3. ELO ФІЛЬТРИ × ODDS CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. ELO ФІЛЬТРИ × ODDS CLUSTER")
print("=" * 80)

for lo, hi, name in odds_clusters:
    print(f"\n  [{lo}-{hi}] {name}")
    # Home: elo_diff > 0 (home stronger)
    for elo_thr in [30, 75, 150]:
        col = 'home_odds_val'
        mask = (
            (df[col] >= lo) & (df[col] < hi) &
            (df['elo_diff'] >= elo_thr)
        )
        analyze(mask, f"home[{lo},{hi}) + elo>={elo_thr}", 'home')
    # Away: elo_diff < 0 (away stronger)
    for elo_thr in [-30, -75, -150]:
        col = 'away_odds_val'
        mask = (
            (df[col] >= lo) & (df[col] < hi) &
            (df['elo_diff'] <= elo_thr)
        )
        analyze(mask, f"away[{lo},{hi}) + elo<={elo_thr}", 'away')

# ══════════════════════════════════════════════════════════════════════════════
# 4. FORM ФІЛЬТРИ × ODDS CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. FORM ФІЛЬТРИ × ODDS CLUSTER")
print("=" * 80)

for lo, hi, name in odds_clusters:
    print(f"\n  [{lo}-{hi}] {name}")
    for form_thr in [1.8, 2.2, 2.5]:
        # Strong home form
        mask = (
            (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi) &
            (df['home_pts_5'].fillna(0) >= form_thr)
        )
        analyze(mask, f"home[{lo},{hi}) + home_pts>={form_thr}", 'home')
        # Strong away form
        mask = (
            (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi) &
            (df['away_pts_5'].fillna(0) >= form_thr)
        )
        analyze(mask, f"away[{lo},{hi}) + away_pts>={form_thr}", 'away')

# ══════════════════════════════════════════════════════════════════════════════
# 5. КОМБІНАЦІЇ (xG + Elo + Form) × ODDS CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. КОМБІНАЦІЇ × ODDS CLUSTER")
print("=" * 80)

combos = []
for lo, hi, name in odds_clusters:
    # Home combos
    combos += [
        (
            (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi) &
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['elo_diff'] >= 30),
            f"home[{lo},{hi}) xg>=1.5 elo>=30", 'home'
        ),
        (
            (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi) &
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['home_pts_5'].fillna(0) >= 1.8),
            f"home[{lo},{hi}) xg>=1.5 form>=1.8", 'home'
        ),
        (
            (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi) &
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['elo_diff'] >= 30) &
            (df['home_pts_5'].fillna(0) >= 1.8),
            f"home[{lo},{hi}) xg>=1.5 elo>=30 form>=1.8", 'home'
        ),
        (
            (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi) &
            (df['mkt_home_prob'] >= 0.50) &
            (df['xg_ratio_home_5'].fillna(0) >= 1.3),
            f"home[{lo},{hi}) mkt>=0.50 xg>=1.3", 'home'
        ),
    ]
    # Away combos
    combos += [
        (
            (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi) &
            (df['xg_ratio_away_5'].fillna(0) >= 1.2) &
            (df['elo_diff'] <= -30),
            f"away[{lo},{hi}) xg>=1.2 elo<=-30", 'away'
        ),
        (
            (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi) &
            (df['xg_ratio_away_5'].fillna(0) >= 1.2) &
            (df['away_pts_5'].fillna(0) >= 1.8),
            f"away[{lo},{hi}) xg>=1.2 form>=1.8", 'away'
        ),
        (
            (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi) &
            (df['xg_ratio_away_5'].fillna(0) >= 1.2) &
            (df['elo_diff'] <= -50) &
            (df['away_pts_5'].fillna(0) >= 1.6),
            f"away[{lo},{hi}) xg>=1.2 elo<=-50 form>=1.6", 'away'
        ),
        (
            (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi) &
            (df['mkt_away_prob'] >= 0.40) &
            (df['xg_ratio_away_5'].fillna(0) >= 1.2),
            f"away[{lo},{hi}) mkt>=0.40 xg>=1.2", 'away'
        ),
    ]

for mask, label, side in combos:
    analyze(mask, label, side)

# ══════════════════════════════════════════════════════════════════════════════
# 6. LEAGUE ANALYSIS — базовий WR/ROI по кожній лізі
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. АНАЛІЗ ПО ЛІГАХ")
print("=" * 80)

league_map = {
    39: 'EPL', 78: 'Bundesliga', 135: 'Serie A',
    140: 'La Liga', 61: 'Ligue 1', 94: 'Primeira Liga',
    136: 'Serie B', 88: 'Eredivisie', 144: 'Jupiler',
    2: 'UCL',
}

print(f"\n  {'League':15s} {'n':>5} {'H-WR':>6} {'A-WR':>6} {'H-ROI':>8} {'A-ROI':>8}")
print("  " + "-" * 55)
for lid, lname in sorted(league_map.items(), key=lambda x: x[1]):
    sub = df[df['league_id'] == lid]
    if len(sub) < 30:
        continue
    n = len(sub)
    hwr = sub['home_win'].mean() * 100
    awr = sub['away_win'].mean() * 100
    hroi = (sub['home_win'] * (sub['home_odds_val'] - 1) - (1 - sub['home_win'])).mean()
    aroi = (sub['away_win'] * (sub['away_odds_val'] - 1) - (1 - sub['away_win'])).mean()
    print(f"  {lname:15s} {n:>5} {hwr:>5.1f}% {awr:>5.1f}% {hroi:>+7.3f} {aroi:>+7.3f}")

# Best niche per league
print(f"\n  Найкращий xG фільтр по лігах [1.7-2.5]:")
print(f"  {'League':15s} {'side':5s} {'n':>5} {'wr':>6} {'roi':>8}")
print("  " + "-" * 45)
for lid, lname in sorted(league_map.items(), key=lambda x: x[1]):
    for side in ('home', 'away'):
        col  = 'home_odds_val' if side == 'home' else 'away_odds_val'
        xcol = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
        wcol = 'home_win' if side == 'home' else 'away_win'
        mask = (
            (df['league_id'] == lid) &
            (df[col] >= 1.70) & (df[col] <= 2.50) &
            (df[xcol].fillna(0) >= 1.3)
        )
        sub = df[mask]
        if len(sub) < 20:
            continue
        wr  = sub[wcol].mean() * 100
        roi = (sub[wcol] * (sub[col] - 1) - (1 - sub[wcol])).mean()
        star = ' ★★' if wr >= 62 else (' ★' if wr >= 57 else '')
        print(f"  {lname:15s} {side:5s} {len(sub):>5} {wr:>5.1f}% {roi:>+7.3f}{star}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY — Топ ніші за Kelly x-factor
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY — ТОП НІШІ (Kelly x-factor, WR >= 55%, n >= 40)")
print("=" * 80)

top = [r for r in results if r['wr'] >= 55.0 and r['n'] >= 40]
top.sort(key=lambda x: -x['kelly_xf'])

print(f"{'Label':65s} {'n':>5} {'WR':>6} {'Odds':>6} {'ROI':>8} {'Kelly':>8}")
print("-" * 105)
for r in top[:30]:
    star = ' ★★' if r['wr'] >= 60 else (' ★' if r['wr'] >= 57 else '')
    print(f"{r['label']:65s} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+7.3f} {r['kelly_xf']:>7.2f}x{star}")

print(f"\nТОП ПО WIN RATE (>= 58%, n >= 40):")
print("-" * 105)
top_wr = sorted([r for r in results if r['wr'] >= 58.0 and r['n'] >= 40],
                key=lambda x: -x['wr'])
for r in top_wr[:20]:
    star = ' ★★' if r['wr'] >= 63 else (' ★' if r['wr'] >= 60 else '')
    print(f"{r['label']:65s} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+7.3f} {r['kelly_xf']:>7.2f}x{star}")
