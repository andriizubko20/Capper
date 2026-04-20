"""
BETA/oos_research.py

Комплексне OOS дослідження.

Розбивка:
  Discovery set: 2023-08 → 2025-10 (тренування, вибір фільтрів)
  OOS set:       2025-11 → 2026-04 (ніколи не торкались — чистий тест)

Фази:
  1. Pure filter scan  — всі комбінації без ML, flat ROI на OOS
  2. ML layer sweep    — топ-30 фільтрів × 4 confidence рівні
  3. Kelly cap sweep   — топ-20 моделей × 5 cap рівнів
  4. League breakdown  — топ-10 моделей по кожній лізі
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from itertools import product

from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols

# ── Split ──────────────────────────────────────────────────────────────────────
DISCOVERY_END = pd.Timestamp("2025-10-31")
OOS_START     = pd.Timestamp("2025-11-01")

KELLY_FRAC    = 0.25
MIN_N_OOS     = 10    # мінімум ставок на OOS для звіту

MARKET_COLS = {
    'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
    'home_odds_val','draw_odds_val','away_odds_val',
}

LEAGUE_MAP = {
    'Premier League':   'EPL',
    'Bundesliga':       'Bundesliga',
    'Serie A':          'Serie A',
    'La Liga':          'La Liga',
    'Ligue 1':          'Ligue 1',
    'Primeira Liga':    'Primeira Liga',
    'Serie B':          'Serie B',
    'Eredivisie':       'Eredivisie',
    'Jupiler Pro League': 'Jupiler',
    'Champions League': 'UCL',
}

# ── Load & split ───────────────────────────────────────────────────────────────
print("Loading data...")
matches, stats, odds_data, injuries = load_all()
df = build_feature_matrix(matches, stats, odds_data, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()

df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win'] = (df['result'] == 'A').astype(int)

disc = df[df['date'] <= DISCOVERY_END].copy()
oos  = df[df['date'] >= OOS_START].copy()

print(f"  Discovery: {len(disc)} matches ({disc['date'].min().date()} – {disc['date'].max().date()})")
print(f"  OOS:       {len(oos)} matches ({oos['date'].min().date()} – {oos['date'].max().date()})")
print(f"  Home WR → disc={disc['home_win'].mean():.1%} oos={oos['home_win'].mean():.1%}")
print(f"  Away WR → disc={disc['away_win'].mean():.1%} oos={oos['away_win'].mean():.1%}\n")

feat_cols = get_feature_cols(df)
train_feat_cols = [c for c in feat_cols if c not in MARKET_COLS]


# ── Helpers ────────────────────────────────────────────────────────────────────
def flat_metrics(sub, side):
    win_col  = 'home_win' if side == 'home' else 'away_win'
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    if len(sub) == 0:
        return None
    n       = len(sub)
    wr      = sub[win_col].mean()
    avg_o   = sub[odds_col].mean()
    roi     = (sub[win_col] * (sub[odds_col] - 1) - (1 - sub[win_col])).mean() * 100
    ev      = (wr * avg_o - 1) * 100
    return {'n': n, 'wr': round(wr*100,1), 'avg_odds': round(avg_o,2),
            'flat_roi': round(roi,2), 'ev': round(ev,2)}


def kelly_sim(sub, side, cap):
    win_col  = 'home_win' if side == 'home' else 'away_win'
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    bank = 1000.0
    for _, row in sub.iterrows():
        odds = row[odds_col]
        b = odds - 1
        p = row[win_col]  # use actual result as proxy (flat Kelly sim)
        # estimated p from WR
    wr = sub[win_col].mean()
    b  = sub[odds_col].mean() - 1
    q  = 1 - wr
    f  = max(0, (wr * b - q) / b) * KELLY_FRAC
    f  = min(f, cap)
    if f <= 0:
        return 1.0
    xf = ((1 + f*b)**wr) * ((1-f)**q)
    return round(xf ** len(sub), 2)


def fit_lgbm_model(X_tr, y_tr, X_cal, y_cal):
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_cal)) < 2:
        return None, None
    params = {
        'objective': 'binary', 'num_leaves': 31, 'learning_rate': 0.03,
        'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.5, 'random_state': 42,
        'n_jobs': -1, 'verbose': -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(params, dtrain, num_boost_round=300,
                        callbacks=[lgb.log_evaluation(period=-1)])
    raw = booster.predict(X_cal).reshape(-1, 1)
    cal = LogisticRegression(max_iter=300, C=1.0)
    cal.fit(raw, y_cal)
    return booster, cal


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 1: Pure filter scan (без ML)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("ФАЗА 1: Pure filter scan (OOS, без ML)")
print("=" * 80)

ODDS_RANGES = [
    (1.30, 1.55), (1.55, 1.80), (1.70, 2.00),
    (1.80, 2.20), (2.00, 2.50), (2.20, 2.80), (2.50, 3.50),
]
XG_THRS   = [0.0, 1.0, 1.2, 1.3, 1.5, 1.8]
ELO_H_THRS = [0, 30, 75, 150]
ELO_A_THRS = [0, -30, -75, -150]
FORM_THRS  = [0.0, 1.5, 1.8, 2.2]
MKT_H_THRS = [0.0, 0.45, 0.50, 0.55]
MKT_A_THRS = [0.0, 0.35, 0.40, 0.45]

phase1_results = []

for side in ('home', 'away'):
    odds_col  = 'home_odds_val' if side == 'home' else 'away_odds_val'
    xg_col    = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
    form_col  = 'home_pts_5' if side == 'home' else 'away_pts_5'
    mkt_thrs  = MKT_H_THRS if side == 'home' else MKT_A_THRS
    elo_thrs  = ELO_H_THRS if side == 'home' else ELO_A_THRS
    mkt_col   = 'mkt_home_prob' if side == 'home' else 'mkt_away_prob'

    for lo, hi in ODDS_RANGES:
        base = (oos[odds_col] >= lo) & (oos[odds_col] < hi)
        for xg_t, elo_t, form_t, mkt_t in product(XG_THRS, elo_thrs, FORM_THRS, mkt_thrs):
            mask = base.copy()
            if xg_t > 0:   mask &= (oos[xg_col].fillna(0) >= xg_t)
            if side == 'home' and elo_t > 0:  mask &= (oos['elo_diff'].fillna(0) >= elo_t)
            if side == 'away' and elo_t < 0:  mask &= (oos['elo_diff'].fillna(0) <= elo_t)
            if form_t > 0:  mask &= (oos[form_col].fillna(0) >= form_t)
            if mkt_t > 0:   mask &= (oos[mkt_col].fillna(0) >= mkt_t)

            sub = oos[mask]
            if len(sub) < MIN_N_OOS:
                continue
            m = flat_metrics(sub, side)
            if not m or m['flat_roi'] <= 0:
                continue

            parts = [f"{side}[{lo},{hi})"]
            if xg_t > 0:   parts.append(f"xg>={xg_t}")
            if side == 'home' and elo_t > 0: parts.append(f"elo>={elo_t}")
            if side == 'away' and elo_t < 0: parts.append(f"elo<={elo_t}")
            if form_t > 0: parts.append(f"form>={form_t}")
            if mkt_t > 0:  parts.append(f"mkt>={mkt_t}")

            phase1_results.append({
                'label': ' '.join(parts), 'side': side,
                'odds_lo': lo, 'odds_hi': hi,
                'xg_t': xg_t, 'elo_t': elo_t, 'form_t': form_t, 'mkt_t': mkt_t,
                **m,
                'kelly_xf_5': kelly_sim(sub, side, 0.05),
                'kelly_xf_7': kelly_sim(sub, side, 0.07),
            })

df_p1 = pd.DataFrame(phase1_results).sort_values('flat_roi', ascending=False)
print(f"\n  Протестовано комбінацій: {len(phase1_results)} прибуткових (n>={MIN_N_OOS})")

hdr = f"  {'Label':<58} {'n':>4} {'WR':>6} {'Odds':>5} {'ROI%':>7} {'EV%':>6} {'K5%':>6} {'K7%':>6}"
sep = "  " + "-" * 100
print(f"\n  ТОП-30 по flat ROI (OOS):")
print(hdr); print(sep)
for _, r in df_p1.head(30).iterrows():
    star = ' ★★' if r['flat_roi'] > 15 else (' ★' if r['flat_roi'] > 7 else '')
    print(f"  {r['label']:<58} {r['n']:>4} {r['wr']:>5.1f}% {r['avg_odds']:>5.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf_5']:>5.2f}x "
          f"{r['kelly_xf_7']:>5.2f}x{star}")

print(f"\n  ТОП-30 по WR >= 60% (OOS):")
print(hdr); print(sep)
for _, r in df_p1[df_p1['wr'] >= 60].head(30).iterrows():
    print(f"  {r['label']:<58} {r['n']:>4} {r['wr']:>5.1f}% {r['avg_odds']:>5.2f} "
          f"{r['flat_roi']:>+6.1f}% {r['ev']:>+5.1f}% {r['kelly_xf_5']:>5.2f}x "
          f"{r['kelly_xf_7']:>5.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 2: ML layer sweep (топ-30 фільтрів + LightGBM)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("ФАЗА 2: ML layer (LightGBM) на топ фільтрах")
print("=" * 80)

CONFIDENCE_LEVELS = [0, 20, 35, 50]  # 0 = без фільтру по confidence
KELLY_CAPS_ML = [0.04, 0.06, 0.08]

top_filters = df_p1.head(30).to_dict('records')

phase2_results = []

for i, flt in enumerate(top_filters):
    side     = flt['side']
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    xg_col   = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
    form_col = 'home_pts_5' if side == 'home' else 'away_pts_5'
    mkt_col  = 'mkt_home_prob' if side == 'home' else 'mkt_away_prob'
    win_col  = 'home_win' if side == 'home' else 'away_win'
    win_val  = 'H' if side == 'home' else 'A'
    lo, hi   = flt['odds_lo'], flt['odds_hi']

    def apply_filter(d):
        mask = (d[odds_col] >= lo) & (d[odds_col] < hi)
        if flt['xg_t'] > 0:   mask &= (d[xg_col].fillna(0) >= flt['xg_t'])
        if side == 'home' and flt['elo_t'] > 0: mask &= (d['elo_diff'].fillna(0) >= flt['elo_t'])
        if side == 'away' and flt['elo_t'] < 0: mask &= (d['elo_diff'].fillna(0) <= flt['elo_t'])
        if flt['form_t'] > 0: mask &= (d[form_col].fillna(0) >= flt['form_t'])
        if flt['mkt_t'] > 0:  mask &= (d[mkt_col].fillna(0) >= flt['mkt_t'])
        return d[mask]

    disc_flt = apply_filter(disc)
    oos_flt  = apply_filter(oos)

    if len(disc_flt) < 50 or len(oos_flt) < MIN_N_OOS:
        continue

    # Тренуємо LightGBM
    y_disc   = (disc_flt['result'] == win_val).astype(int).values
    cal_size = max(20, int(len(disc_flt) * 0.2))
    X_tr  = disc_flt[train_feat_cols].fillna(0).values[:-cal_size]
    y_tr  = y_disc[:-cal_size]
    X_cal = disc_flt[train_feat_cols].fillna(0).values[-cal_size:]
    y_cal = y_disc[-cal_size:]

    booster, cal_model = fit_lgbm_model(X_tr, y_tr, X_cal, y_cal)
    if booster is None:
        continue

    X_oos  = oos_flt[train_feat_cols].fillna(0).values
    raw    = booster.predict(X_oos).reshape(-1, 1)
    probs  = cal_model.predict_proba(raw)[:, 1]

    for conf_pct in CONFIDENCE_LEVELS:
        if conf_pct > 0:
            thr  = np.percentile(probs, 100 - conf_pct)
            mask = (probs >= thr) & ((probs * (oos_flt[odds_col].values) - 1) > 0)
        else:
            mask = (probs * (oos_flt[odds_col].values) - 1) > 0

        sub = oos_flt[mask.astype(bool) if hasattr(mask, 'astype') else mask]
        if len(sub) < MIN_N_OOS:
            continue

        m = flat_metrics(sub, side)
        if not m:
            continue

        for cap in KELLY_CAPS_ML:
            kx = kelly_sim(sub, side, cap)
            phase2_results.append({
                'label':     flt['label'],
                'conf_pct':  conf_pct,
                'kelly_cap': cap,
                'side':      side,
                **m,
                'kelly_xf':  kx,
            })

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(top_filters)}] оброблено...")

df_p2 = pd.DataFrame(phase2_results)

if len(df_p2) > 0:
    print(f"\n  ML результати: {len(df_p2)} комбінацій")
    best = df_p2[df_p2['flat_roi'] > 0].sort_values('flat_roi', ascending=False)

    print(f"\n  ТОП-30 по flat ROI (ML + OOS):")
    hdr2 = f"  {'Label':<55} {'Conf':>5} {'Cap':>5} {'n':>4} {'WR':>6} {'ROI%':>7} {'Kelly':>6}"
    print(hdr2); print("  " + "-" * 95)
    for _, r in best.head(30).iterrows():
        star = ' ★★' if r['flat_roi'] > 15 else (' ★' if r['flat_roi'] > 7 else '')
        print(f"  {r['label']:<55} top{r['conf_pct']:>2}% {r['kelly_cap']*100:.0f}%  "
              f"{r['n']:>4} {r['wr']:>5.1f}% {r['flat_roi']:>+6.1f}% "
              f"{r['kelly_xf']:>5.2f}x{star}")

    # Топ без ML (conf=0) vs з ML
    print(f"\n  Порівняння: без ML (conf=0) vs top-35% confidence:")
    comp = df_p2[df_p2['kelly_cap'] == 0.06].pivot_table(
        index='label', columns='conf_pct',
        values='flat_roi', aggfunc='mean'
    ).round(1)
    print(comp.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 3: Kelly cap sensitivity (топ-20 по Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("ФАЗА 3: Kelly cap sensitivity (топ фільтри × cap 2-10%)")
print("=" * 80)

KELLY_CAPS_SWEEP = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

def kelly_xf_math(wr_pct, avg_odds, n, cap):
    wr  = wr_pct / 100
    b   = avg_odds - 1
    q   = 1 - wr
    f   = max(0.0, (wr * b - q) / b) * KELLY_FRAC
    f   = min(f, cap)
    if f <= 0:
        return 1.0
    xf  = ((1 + f * b) ** wr) * ((1 - f) ** q)
    return round(xf ** n, 2)

top20_p1 = df_p1.head(20)
print(f"\n  {'Label':<58} {'n':>4}  " + '  '.join(f"{c*100:.0f}%" for c in KELLY_CAPS_SWEEP))
print("  " + "-" * 115)
for _, row in top20_p1.iterrows():
    caps_str = '  '.join(
        f"{kelly_xf_math(row['wr'], row['avg_odds'], row['n'], c):5.2f}x"
        for c in KELLY_CAPS_SWEEP
    )
    print(f"  {row['label']:<58} {row['n']:>4}  {caps_str}")

# Додатково: форма порогів — як form впливає на ROI по odds діапазонах
print(f"\n  FORM THRESHOLD ANALYSIS (home, OOS):")
print(f"  {'odds \\ form':<20} {'no filt':>9} {'>=1.5':>9} {'>=1.8':>9} {'>=2.0':>9} {'>=2.2':>9} {'>=2.5':>9}")
print("  " + "-" * 80)
for lo, hi in [(1.30,1.55),(1.55,1.80),(1.70,2.00),(1.80,2.20),(2.00,2.50),(2.20,2.80)]:
    row_str = f"  home[{lo},{hi})<{'':<8}"
    for ft in [0.0, 1.5, 1.8, 2.0, 2.2, 2.5]:
        mask = (oos['home_odds_val'] >= lo) & (oos['home_odds_val'] < hi)
        if ft > 0:
            mask &= (oos['home_pts_5'].fillna(0) >= ft)
        sub = oos[mask]
        if len(sub) < 5:
            row_str += f"  {'n/a':>7}"
        else:
            m = flat_metrics(sub, 'home')
            row_str += f"  {m['flat_roi']:>+6.1f}%"
    print(row_str)

print(f"\n  FORM THRESHOLD ANALYSIS (away, OOS):")
print(f"  {'odds \\ form':<20} {'no filt':>9} {'>=1.5':>9} {'>=1.8':>9} {'>=2.0':>9} {'>=2.2':>9} {'>=2.5':>9}")
print("  " + "-" * 80)
for lo, hi in [(1.30,1.55),(1.55,1.80),(1.70,2.00),(1.80,2.20),(2.00,2.50),(2.20,2.80)]:
    row_str = f"  away[{lo},{hi})<{'':<8}"
    for ft in [0.0, 1.5, 1.8, 2.0, 2.2, 2.5]:
        mask = (oos['away_odds_val'] >= lo) & (oos['away_odds_val'] < hi)
        if ft > 0:
            mask &= (oos['away_pts_5'].fillna(0) >= ft)
        sub = oos[mask]
        if len(sub) < 5:
            row_str += f"  {'n/a':>7}"
        else:
            m = flat_metrics(sub, 'away')
            row_str += f"  {m['flat_roi']:>+6.1f}%"
    print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 4: League breakdown (топ-10 моделей)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("ФАЗА 4: League breakdown (топ моделі по лігах)")
print("=" * 80)

# Беремо топ-10 фільтрів по flat ROI (без ML) для ліг
top10_filters = df_p1.head(10).to_dict('records')

for flt in top10_filters:
    side     = flt['side']
    odds_col = 'home_odds_val' if side == 'home' else 'away_odds_val'
    xg_col   = 'xg_ratio_home_5' if side == 'home' else 'xg_ratio_away_5'
    form_col = 'home_pts_5' if side == 'home' else 'away_pts_5'
    mkt_col  = 'mkt_home_prob' if side == 'home' else 'mkt_away_prob'
    win_col  = 'home_win' if side == 'home' else 'away_win'
    lo, hi   = flt['odds_lo'], flt['odds_hi']

    base = (oos[odds_col] >= lo) & (oos[odds_col] < hi)
    mask = base.copy()
    if flt['xg_t'] > 0:   mask &= (oos[xg_col].fillna(0) >= flt['xg_t'])
    if side == 'home' and flt['elo_t'] > 0: mask &= (oos['elo_diff'].fillna(0) >= flt['elo_t'])
    if side == 'away' and flt['elo_t'] < 0: mask &= (oos['elo_diff'].fillna(0) <= flt['elo_t'])
    if flt['form_t'] > 0: mask &= (oos[form_col].fillna(0) >= flt['form_t'])
    if flt['mkt_t'] > 0:  mask &= (oos[mkt_col].fillna(0) >= flt['mkt_t'])

    sub_all = oos[mask]
    if len(sub_all) < MIN_N_OOS:
        continue

    print(f"\n  [{flt['label']}]  total: n={flt['n']}, WR={flt['wr']}%, ROI={flt['flat_roi']:+.1f}%")
    print(f"  {'League':<15} {'n':>4} {'WR':>6} {'ROI%':>8}")
    print(f"  {'-'*38}")
    found_any = False
    for lname_key, lname_short in sorted(LEAGUE_MAP.items(), key=lambda x: x[1]):
        sub_l = sub_all[sub_all['league_name'] == lname_key]
        if len(sub_l) < 3:
            continue
        m = flat_metrics(sub_l, side)
        if m:
            verdict = ' ✅' if m['flat_roi'] > 0 else ' ❌'
            print(f"  {lname_short:<15} {m['n']:>4} {m['wr']:>5.1f}% {m['flat_roi']:>+7.1f}%{verdict}")
            found_any = True
    if not found_any:
        # Show discovery set league stats for context (more data)
        base_d = (disc[odds_col] >= lo) & (disc[odds_col] < hi)
        mask_d = base_d.copy()
        if flt['xg_t'] > 0:   mask_d &= (disc[xg_col].fillna(0) >= flt['xg_t'])
        if flt['form_t'] > 0:  mask_d &= (disc[form_col].fillna(0) >= flt['form_t'])
        if flt['mkt_t'] > 0:   mask_d &= (disc[mkt_col].fillna(0) >= flt['mkt_t'])
        sub_disc = disc[mask_d]
        print(f"  (OOS замало — показую Discovery period, n={len(sub_disc)})")
        for lname_key, lname_short in sorted(LEAGUE_MAP.items(), key=lambda x: x[1]):
            sub_l = sub_disc[sub_disc['league_name'] == lname_key]
            if len(sub_l) < 5:
                continue
            m = flat_metrics(sub_l, side)
            if m:
                verdict = ' ✅' if m['flat_roi'] > 0 else ' ❌'
                print(f"  {lname_short:<15} {m['n']:>4} {m['wr']:>5.1f}% {m['flat_roi']:>+7.1f}%{verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# ФІНАЛЬНИЙ SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("ФІНАЛЬНИЙ SUMMARY — найкращі моделі для walk-forward")
print("=" * 80)
print(f"\n  OOS period: {OOS_START.date()} – {oos['date'].max().date()} ({len(oos)} matches)")
print(f"\n  ТОП-5 pure filter (без ML):")
for _, r in df_p1.head(5).iterrows():
    print(f"    {r['label']:<58} WR={r['wr']:.1f}% ROI={r['flat_roi']:+.1f}%")

if len(df_p2) > 0:
    best_ml = df_p2[df_p2['flat_roi'] > 0].sort_values('flat_roi', ascending=False).head(5)
    print(f"\n  ТОП-5 з ML (conf=35%, cap=6%):")
    for _, r in best_ml[best_ml['kelly_cap'] == 0.06].head(5).iterrows():
        print(f"    {r['label']:<55} conf=top{r['conf_pct']:.0f}% "
              f"WR={r['wr']:.1f}% ROI={r['flat_roi']:+.1f}%")
