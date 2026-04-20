"""
BETA/check_oos_niches.py
Знаходить ніші з WF avg_roi>=10%, win%>=60%, n_windows>=4
та n>=15 в OOS з позитивним ROI.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from itertools import product

from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix

WF_END    = pd.Timestamp('2025-10-31')
OOS_START = pd.Timestamp('2025-11-01')
MIN_OOS_N = 15

print("Loading data...")
matches, stats, odds_data, injuries = load_all()
df = build_feature_matrix(matches, stats, odds_data, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win']  = (df['result'] == 'A').astype(int)

disc_df = df[df['date'] <= WF_END].copy()
oos_df  = df[df['date'] >= OOS_START].copy()

WINDOWS = []
cur = pd.Timestamp('2023-08-01')
while cur <= WF_END:
    w_end = cur + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    if w_end > WF_END: w_end = WF_END
    WINDOWS.append((cur.strftime('%Y-%m'), cur, w_end))
    cur += pd.DateOffset(months=3)
WIN_LABELS = [w[0] for w in WINDOWS]

LEAGUES = [
    'Premier League','Bundesliga','Serie A','La Liga','Ligue 1',
    'Primeira Liga','Serie B','Eredivisie','Jupiler Pro League','Champions League'
]
LSHORT = {
    'Premier League':'EPL','Bundesliga':'Bundesliga','Serie A':'Serie A',
    'La Liga':'La Liga','Ligue 1':'Ligue 1','Primeira Liga':'Portugal',
    'Serie B':'Serie B','Eredivisie':'Eredivisie',
    'Jupiler Pro League':'Jupiler','Champions League':'UCL'
}

ODDS_RANGES = [(1.30,1.55),(1.55,1.80),(1.70,2.00),(1.80,2.20),(2.00,2.50),(2.20,2.80),(2.50,3.50)]
XG    = [0.0,1.0,1.2,1.5,1.8]
EH    = [0,30,75,150];   EA = [0,-30,-75,-150]
FT    = [0.0,1.5,1.8,2.2]
MH    = [0.0,0.45,0.50,0.55]; MA = [0.0,0.35,0.40,0.45]

win_slices = {wlbl: disc_df[(disc_df['date'] >= ws) & (disc_df['date'] <= we)]
              for wlbl, ws, we in WINDOWS}

def apply_mask(d, side, lo, hi, xg_t, elo_t, form_t, mkt_t):
    oc  = 'home_odds_val'   if side=='home' else 'away_odds_val'
    xgc = 'xg_ratio_home_5' if side=='home' else 'xg_ratio_away_5'
    fc  = 'home_pts_5'      if side=='home' else 'away_pts_5'
    mc  = 'mkt_home_prob'   if side=='home' else 'mkt_away_prob'
    m = (d[oc] >= lo) & (d[oc] < hi)
    if xg_t > 0:                         m &= (d[xgc].fillna(0) >= xg_t)
    if side=='home' and elo_t > 0:       m &= (d['elo_diff'].fillna(0) >= elo_t)
    if side=='away' and elo_t < 0:       m &= (d['elo_diff'].fillna(0) <= elo_t)
    if form_t > 0:                       m &= (d[fc].fillna(0) >= form_t)
    if mkt_t > 0:                        m &= (d[mc].fillna(0) >= mkt_t)
    return d[m]

print("Sweeping filters...")
results = []

for side in ('home', 'away'):
    et = EH if side=='home' else EA
    mt = MH if side=='home' else MA
    wc = 'home_win'      if side=='home' else 'away_win'
    oc = 'home_odds_val' if side=='home' else 'away_odds_val'

    for lo, hi in ODDS_RANGES:
        for xg_t, elo_t, form_t, mkt_t in product(XG, et, FT, mt):
            parts = [f'{side}[{lo},{hi})']
            if xg_t > 0:                     parts.append(f'xg>={xg_t}')
            if side=='home' and elo_t > 0:   parts.append(f'elo>={elo_t}')
            if side=='away' and elo_t < 0:   parts.append(f'elo<={elo_t}')
            if form_t > 0:                   parts.append(f'form>={form_t}')
            if mkt_t > 0:                    parts.append(f'mkt>={mkt_t}')
            label = ' '.join(parts)

            # Pre-filter OOS once per combination
            oos_sub = apply_mask(oos_df, side, lo, hi, xg_t, elo_t, form_t, mkt_t)

            for lg in LEAGUES:
                # WF check
                wf_rois = []
                for wlbl, _, _ in WINDOWS:
                    sl = apply_mask(win_slices[wlbl], side, lo, hi, xg_t, elo_t, form_t, mkt_t)
                    sl = sl[sl['league_name'] == lg]
                    if len(sl) < 3: continue
                    wf_rois.append((sl[wc]*(sl[oc]-1)-(1-sl[wc])).mean()*100)
                if len(wf_rois) < 4: continue
                avg_roi = np.mean(wf_rois)
                win_pct = sum(1 for r in wf_rois if r > 0) / len(wf_rois) * 100
                if avg_roi < 10 or win_pct < 60: continue

                # OOS check
                sl_oos = oos_sub[oos_sub['league_name'] == lg]
                if len(sl_oos) < MIN_OOS_N: continue
                oos_roi = (sl_oos[wc]*(sl_oos[oc]-1)-(1-sl_oos[wc])).mean()*100
                oos_wr  = sl_oos[wc].mean()*100

                results.append({
                    'lg': LSHORT[lg], 'label': label,
                    'wf_roi': round(avg_roi, 1), 'wf_wp': round(win_pct, 1),
                    'wf_n_win': f"{sum(1 for r in wf_rois if r>0)}/{len(wf_rois)}",
                    'oos_n': len(sl_oos),
                    'oos_wr': round(oos_wr, 1),
                    'oos_roi': round(oos_roi, 1),
                })

pos = sorted([r for r in results if r['oos_roi'] > 0], key=lambda x: -x['oos_roi'])
neg = [r for r in results if r['oos_roi'] <= 0]

print(f"\n{'='*70}")
print(f"n>={MIN_OOS_N} в OOS: всього {len(results)} ніш  |  профітних: {len(pos)}  |  збиткових: {len(neg)}")
print(f"{'='*70}\n")

by_lg = {}
for p in pos:
    by_lg.setdefault(p['lg'], []).append(p)

for lg, items in sorted(by_lg.items(), key=lambda x: -sum(i['oos_roi'] for i in x[1])):
    avg = np.mean([i['oos_roi'] for i in items])
    print(f"{'─'*60}")
    print(f"{lg}: {len(items)} профітних ніш  (avg OOS ROI = {avg:+.1f}%)")
    print(f"{'─'*60}")
    for p in items:
        print(f"  OOS: {p['oos_roi']:+5.1f}% ROI  WR={p['oos_wr']:.1f}%  n={p['oos_n']:>3}"
              f"  | WF: {p['wf_roi']:+.1f}% ({p['wf_wp']:.0f}% вікон, {p['wf_n_win']})")
        print(f"    {p['label']}")
    print()

# Summary table
print(f"\n{'='*70}")
print("SUMMARY по лігах:")
print(f"{'Ліга':<14} {'Ніш+':<6} {'Avg OOS ROI':<14} {'Топ OOS ROI':<14} {'Avg OOS WR'}")
print(f"{'─'*60}")
for lg, items in sorted(by_lg.items(), key=lambda x: -len(x[1])):
    avg_roi = np.mean([i['oos_roi'] for i in items])
    top_roi = max(i['oos_roi'] for i in items)
    avg_wr  = np.mean([i['oos_wr'] for i in items])
    print(f"{lg:<14} {len(items):<6} {avg_roi:+.1f}%{'':<9} {top_roi:+.1f}%{'':<9} {avg_wr:.1f}%")
