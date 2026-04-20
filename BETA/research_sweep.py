"""
BETA/research_sweep.py

Систематичний sweep по всіх комбінаціях факторів і odds рівнів.
Метрики: WR, flat ROI, Kelly x-factor, EV.

Логіка:
  - Для кожної комбінації (side × odds_range × filters) рахуємо всі 4 метрики
  - Відкидаємо якщо n < MIN_N
  - Сортуємо по flat ROI (головна метрика), виводимо топ
  - Окремо: heat-map xg_threshold × odds_range для home і away
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from itertools import product
from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
matches, stats, odds, injuries = load_all()
df = build_feature_matrix(matches, stats, odds, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win'] = (df['result'] == 'A').astype(int)
print(f"Dataset: {len(df)} matches | {df['date'].min().date()} – {df['date'].max().date()}")
print(f"Home WR: {df['home_win'].mean():.1%} | Away WR: {df['away_win'].mean():.1%}\n")

MIN_N = 40  # мінімум ставок для статистичної значущості

# ── Метрики ───────────────────────────────────────────────────────────────────
def metrics(sub, side, label):
    if len(sub) < MIN_N:
        return None
    win_col  = 'home_win' if side == 'home' else 'away_win'
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    n      = len(sub)
    wr     = sub[win_col].mean()
    avg_o  = sub[odds_col].mean()
    # Flat stake ROI
    flat_roi = (sub[win_col] * (sub[odds_col] - 1) - (1 - sub[win_col])).mean()
    # EV
    ev = wr * avg_o - 1
    # Kelly x-factor (geometric growth approx, 25% frac, 6% cap)
    b = avg_o - 1
    q = 1 - wr
    f = max(0, (wr * b - q) / b) * 0.25
    f = min(f, 0.06)
    if f > 0:
        xf = ((1 + f * b) ** wr) * ((1 - f) ** q)
        kelly_xf = round(xf ** n, 2)
    else:
        kelly_xf = 1.0
    return {
        'label': label, 'side': side, 'n': n,
        'wr': round(wr * 100, 1),
        'avg_odds': round(avg_o, 2),
        'flat_roi': round(flat_roi * 100, 2),  # у %
        'ev': round(ev * 100, 2),
        'kelly_xf': kelly_xf,
    }


# ── Параметри sweep ───────────────────────────────────────────────────────────
ODDS_RANGES = [
    (1.30, 1.50), (1.50, 1.70), (1.70, 1.90),
    (1.90, 2.20), (2.20, 2.60), (2.60, 3.50),
    # Ширші діапазони
    (1.30, 1.70), (1.50, 2.00), (1.70, 2.50), (2.00, 3.00),
]
XG_THRESHOLDS  = [0.0, 1.0, 1.2, 1.3, 1.5, 1.8]   # 0.0 = без фільтру
ELO_HOME_THRS  = [0, 30, 75, 150]                    # home elo_diff >=
ELO_AWAY_THRS  = [0, -30, -75, -150]                 # away elo_diff <=
FORM_HOME_THRS = [0.0, 1.5, 1.8, 2.2]               # home_pts_5 >=
FORM_AWAY_THRS = [0.0, 1.5, 1.8, 2.2]               # away_pts_5 >=
MKT_HOME_THRS  = [0.0, 0.45, 0.50, 0.55]            # mkt_home_prob >=
MKT_AWAY_THRS  = [0.0, 0.35, 0.40, 0.45]            # mkt_away_prob >=

results = []

# ══════════════════════════════════════════════════════════════════════════════
# 1. HEAT-MAP: xG threshold × odds range (home і away окремо)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. HEAT-MAP: xG × Odds (flat ROI %)")
print("=" * 80)

for side in ('home', 'away'):
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    xg_col   = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
    print(f"\n  {side.upper()}:")
    header = f"  {'xg\\odds':>10}" + ''.join(f"  [{lo:.2f},{hi:.2f})" for lo, hi in ODDS_RANGES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for xg_thr in XG_THRESHOLDS:
        row = f"  xg>={xg_thr:3.1f}    "
        for lo, hi in ODDS_RANGES:
            mask = (df[odds_col] >= lo) & (df[odds_col] < hi)
            if xg_thr > 0:
                mask &= (df[xg_col].fillna(0) >= xg_thr)
            r = metrics(df[mask], side, '')
            if r:
                roi_str = f"{r['flat_roi']:+5.1f}%"
                marker  = '★' if r['flat_roi'] > 5 else (' ' if r['flat_roi'] >= 0 else ' ')
                row += f"  {roi_str}{marker}"
            else:
                row += f"  {'n/a':>7} "
        print(row)

# ══════════════════════════════════════════════════════════════════════════════
# 2. HEAT-MAP: xG × Odds (WR %)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. HEAT-MAP: xG × Odds (WR %)")
print("=" * 80)

for side in ('home', 'away'):
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    xg_col   = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
    print(f"\n  {side.upper()}:")
    header = f"  {'xg\\odds':>10}" + ''.join(f"  [{lo:.2f},{hi:.2f})" for lo, hi in ODDS_RANGES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for xg_thr in XG_THRESHOLDS:
        row = f"  xg>={xg_thr:3.1f}    "
        for lo, hi in ODDS_RANGES:
            mask = (df[odds_col] >= lo) & (df[odds_col] < hi)
            if xg_thr > 0:
                mask &= (df[xg_col].fillna(0) >= xg_thr)
            r = metrics(df[mask], side, '')
            if r:
                wr_str = f"{r['wr']:5.1f}%"
                marker = '★' if r['wr'] >= 60 else ' '
                row += f"  {wr_str}{marker}"
            else:
                row += f"  {'n/a':>7} "
        print(row)

# ══════════════════════════════════════════════════════════════════════════════
# 3. FULL SWEEP: всі комбінації (xG + Elo/Form/Mkt) × odds × side
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. FULL SWEEP — всі фактори (топ по flat ROI)")
print("=" * 80)

SWEEP_ODDS = [
    (1.30, 1.55), (1.55, 1.80), (1.80, 2.20), (2.20, 2.80),
    (1.50, 2.00), (1.70, 2.50),
]

# HOME sweep
for lo, hi in SWEEP_ODDS:
    base_home = (df['home_odds_val'] >= lo) & (df['home_odds_val'] < hi)
    for xg_thr, elo_thr, form_thr, mkt_thr in product(
        XG_THRESHOLDS, ELO_HOME_THRS, FORM_HOME_THRS, MKT_HOME_THRS
    ):
        mask = base_home.copy()
        if xg_thr > 0:
            mask &= (df['xg_ratio_home_5'].fillna(0) >= xg_thr)
        if elo_thr > 0:
            mask &= (df['elo_diff'].fillna(0) >= elo_thr)
        if form_thr > 0:
            mask &= (df['home_pts_5'].fillna(0) >= form_thr)
        if mkt_thr > 0:
            mask &= (df['mkt_home_prob'].fillna(0) >= mkt_thr)

        parts = [f"home[{lo},{hi})"]
        if xg_thr > 0:  parts.append(f"xg>={xg_thr}")
        if elo_thr > 0:  parts.append(f"elo>={elo_thr}")
        if form_thr > 0: parts.append(f"form>={form_thr}")
        if mkt_thr > 0:  parts.append(f"mkt>={mkt_thr}")
        label = ' '.join(parts)

        r = metrics(df[mask], 'home', label)
        if r:
            results.append(r)

# AWAY sweep
for lo, hi in SWEEP_ODDS:
    base_away = (df['away_odds_val'] >= lo) & (df['away_odds_val'] < hi)
    for xg_thr, elo_thr, form_thr, mkt_thr in product(
        XG_THRESHOLDS, ELO_AWAY_THRS, FORM_AWAY_THRS, MKT_AWAY_THRS
    ):
        mask = base_away.copy()
        if xg_thr > 0:
            mask &= (df['xg_ratio_away_5'].fillna(0) >= xg_thr)
        if elo_thr < 0:
            mask &= (df['elo_diff'].fillna(0) <= elo_thr)
        if form_thr > 0:
            mask &= (df['away_pts_5'].fillna(0) >= form_thr)
        if mkt_thr > 0:
            mask &= (df['mkt_away_prob'].fillna(0) >= mkt_thr)

        parts = [f"away[{lo},{hi})"]
        if xg_thr > 0:  parts.append(f"xg>={xg_thr}")
        if elo_thr < 0:  parts.append(f"elo<={elo_thr}")
        if form_thr > 0: parts.append(f"form>={form_thr}")
        if mkt_thr > 0:  parts.append(f"mkt>={mkt_thr}")
        label = ' '.join(parts)

        r = metrics(df[mask], 'away', label)
        if r:
            results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ══════════════════════════════════════════════════════════════════════════════
df_r = pd.DataFrame(results)
df_r = df_r[df_r['flat_roi'] > 0].copy()  # тільки прибуткові

hdr = f"{'Label':<60} {'n':>5} {'WR':>6} {'Odds':>6} {'ROI%':>7} {'EV%':>6} {'Kelly':>7}"
sep = "-" * len(hdr)

# Топ по flat ROI
print(f"\nТОП-30 по flat ROI (прибуткові, n>={MIN_N}):")
print(hdr); print(sep)
for _, r in df_r.sort_values('flat_roi', ascending=False).head(30).iterrows():
    star = ' ★★' if r['flat_roi'] > 10 else (' ★' if r['flat_roi'] > 5 else '')
    print(f"{r['label']:<60} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf']:>6.2f}x{star}")

# Топ по Kelly x-factor
print(f"\nТОП-30 по Kelly x-factor (прибуткові, n>={MIN_N}):")
print(hdr); print(sep)
for _, r in df_r.sort_values('kelly_xf', ascending=False).head(30).iterrows():
    star = ' ★★' if r['kelly_xf'] > 2.0 else (' ★' if r['kelly_xf'] > 1.5 else '')
    print(f"{r['label']:<60} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf']:>6.2f}x{star}")

# Топ по WR >= 60% і ROI > 0 — золоті ніші
print(f"\nЗОЛОТІ НІШІ (WR>=60%, ROI>0%, n>={MIN_N}):")
print(hdr); print(sep)
gold = df_r[(df_r['wr'] >= 60) & (df_r['flat_roi'] > 0)].sort_values('flat_roi', ascending=False)
for _, r in gold.head(30).iterrows():
    print(f"{r['label']:<60} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf']:>6.2f}x")

# Окремо: кращі комбо з odds [1.70-2.50] (value range)
print(f"\nVALUE RANGE [1.70-2.50] — найкращі по ROI:")
print(hdr); print(sep)
value = df_r[df_r['label'].str.contains('1.7,2.5|1.50,2.00|1.70,1.90|1.90,2.20', regex=True)]
for _, r in value.sort_values('flat_roi', ascending=False).head(20).iterrows():
    star = ' ★★' if r['flat_roi'] > 8 else (' ★' if r['flat_roi'] > 4 else '')
    print(f"{r['label']:<60} {r['n']:>5} {r['wr']:>5.1f}% {r['avg_odds']:>6.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf']:>6.2f}x{star}")

print(f"\nВсього протестовано комбінацій: {len(results)}")
print(f"Прибуткових (ROI>0): {len(df_r)}")
print(f"Золотих (WR>=60%, ROI>0): {len(gold)}")
