"""
BETA/walkforward.py

Walk-forward validation per league.
11 quarterly windows (2023-08 → 2026-04), 3m each.
For every filter combination × league → ROI per window.

Output:
  BETA/walkforward_league_full.txt  — повна таблиця
  Console                           — condensed summary
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
matches, stats, odds_data, injuries = load_all()
df = build_feature_matrix(matches, stats, odds_data, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result'] == 'H').astype(int)
df['away_win']  = (df['result'] == 'A').astype(int)
print(f"Total: {len(df)} | {df['date'].min().date()} – {df['date'].max().date()}")

LEAGUES = [
    'Premier League','Bundesliga','Serie A','La Liga','Ligue 1',
    'Primeira Liga','Serie B','Eredivisie','Jupiler Pro League','Champions League'
]
LEAGUE_SHORT = {
    'Premier League':'EPL','Bundesliga':'Bundesliga','Serie A':'Serie A',
    'La Liga':'La Liga','Ligue 1':'Ligue 1','Primeira Liga':'Portugal',
    'Serie B':'Serie B','Eredivisie':'Eredivisie',
    'Jupiler Pro League':'Jupiler','Champions League':'UCL'
}

# ── Quarterly windows ─────────────────────────────────────────────────────────
WINDOWS = []
cur = pd.Timestamp('2023-08-01')
end = pd.Timestamp('2026-04-30')
while cur < end:
    w_end = cur + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    if w_end > end: w_end = end
    WINDOWS.append((cur.strftime('%Y-%m'), cur, w_end))
    cur += pd.DateOffset(months=3)
WIN_LABELS = [w[0] for w in WINDOWS]
print(f"Windows: {len(WINDOWS)} quarters ({WIN_LABELS[0]} → {WIN_LABELS[-1]})")

# ── Filter space ──────────────────────────────────────────────────────────────
ODDS_RANGES = [
    (1.30,1.55),(1.55,1.80),(1.70,2.00),
    (1.80,2.20),(2.00,2.50),(2.20,2.80),(2.50,3.50),
]
XG_THRS    = [0.0,1.0,1.2,1.5,1.8]
ELO_H_THRS = [0,30,75,150]
ELO_A_THRS = [0,-30,-75,-150]
FORM_THRS  = [0.0,1.5,1.8,2.2]
MKT_H_THRS = [0.0,0.45,0.50,0.55]
MKT_A_THRS = [0.0,0.35,0.40,0.45]
MIN_N_WIN  = 3

# Pre-slice by window
win_slices = {}
for wlbl, ws, we in WINDOWS:
    win_slices[wlbl] = df[(df['date'] >= ws) & (df['date'] <= we)]

# ── Main loop ─────────────────────────────────────────────────────────────────
# results[league][label] = dict with window keys + metadata
results = {lg: {} for lg in LEAGUES}

print("Running sweep...")
total_combos = 0
for side in ('home','away'):
    oc  = 'home_odds_val'   if side=='home' else 'away_odds_val'
    xgc = 'xg_ratio_home_5' if side=='home' else 'xg_ratio_away_5'
    fc  = 'home_pts_5'      if side=='home' else 'away_pts_5'
    mc  = 'mkt_home_prob'   if side=='home' else 'mkt_away_prob'
    wc  = 'home_win'        if side=='home' else 'away_win'
    mt  = MKT_H_THRS if side=='home' else MKT_A_THRS
    et  = ELO_H_THRS if side=='home' else ELO_A_THRS

    for lo, hi in ODDS_RANGES:
        for xg_t, elo_t, form_t, mkt_t in product(XG_THRS, et, FORM_THRS, mt):
            parts = [f'{side}[{lo},{hi})']
            if xg_t  > 0: parts.append(f'xg>={xg_t}')
            if side=='home' and elo_t > 0: parts.append(f'elo>={elo_t}')
            if side=='away' and elo_t < 0: parts.append(f'elo<={elo_t}')
            if form_t > 0: parts.append(f'form>={form_t}')
            if mkt_t  > 0: parts.append(f'mkt>={mkt_t}')
            label = ' '.join(parts)
            total_combos += 1

            for wlbl, _, _ in WINDOWS:
                wdf  = win_slices[wlbl]
                mask = (wdf[oc] >= lo) & (wdf[oc] < hi)
                if xg_t  > 0: mask &= (wdf[xgc].fillna(0) >= xg_t)
                if side=='home' and elo_t > 0: mask &= (wdf['elo_diff'].fillna(0) >= elo_t)
                if side=='away' and elo_t < 0: mask &= (wdf['elo_diff'].fillna(0) <= elo_t)
                if form_t > 0: mask &= (wdf[fc].fillna(0)  >= form_t)
                if mkt_t  > 0: mask &= (wdf[mc].fillna(0)  >= mkt_t)
                sub = wdf[mask]

                for lg in LEAGUES:
                    sub_l = sub[sub['league_name'] == lg]
                    n = len(sub_l)
                    if n < MIN_N_WIN: continue
                    wr  = sub_l[wc].mean()
                    avg_o = sub_l[oc].mean()
                    roi = (sub_l[wc]*(sub_l[oc]-1) - (1-sub_l[wc])).mean()*100
                    ev  = (wr * avg_o - 1) * 100
                    if label not in results[lg]:
                        results[lg][label] = {'side':side,'lo':lo,'hi':hi,
                                               'xg_t':xg_t,'elo_t':elo_t,
                                               'form_t':form_t,'mkt_t':mkt_t}
                    results[lg][label][wlbl] = {
                        'roi': round(roi,1), 'n': n,
                        'wr':  round(wr*100,1), 'ev': round(ev,1)
                    }

print(f"Sweep done. {total_combos} combinations.")

# ── Helper: aggregate stats for a filter across windows ───────────────────────
def agg(data):
    wins = [(wl, data[wl]) for wl in WIN_LABELS if wl in data]
    if not wins: return None
    n_tot      = sum(w['n']   for _, w in wins)
    avg_roi    = round(np.mean([w['roi'] for _, w in wins]), 1)
    avg_wr     = round(np.mean([w['wr']  for _, w in wins]), 1)
    avg_ev     = round(np.mean([w['ev']  for _, w in wins]), 1)
    n_pos      = sum(1 for _, w in wins if w['roi'] > 0)
    win_pct    = round(n_pos / len(wins) * 100, 0)
    n_windows  = len(wins)
    return {'n_tot':n_tot,'avg_roi':avg_roi,'avg_wr':avg_wr,'avg_ev':avg_ev,
            'win_pct':win_pct,'n_windows':n_windows,'n_pos':n_pos}

# ── Build full output file ─────────────────────────────────────────────────────
print("Building output file...")
out_lines = []

for lg in LEAGUES:
    lname  = LEAGUE_SHORT[lg]
    sub_lg = df[df['league_name'] == lg]
    out_lines.append(f'\n{"="*160}')
    out_lines.append(f'  {lname} ({lg})')
    out_lines.append(f'  Total matches: {len(sub_lg)}   Date: {sub_lg["date"].min().date()} – {sub_lg["date"].max().date()}')
    out_lines.append(f'{"="*160}')

    # Column header
    win_cols = '  '.join(f'{wl:<12}' for wl in WIN_LABELS)
    hdr = f'  {"Filter":<54}  {"n_tot":>5}  {"Avg_ROI%":>8}  {"Avg_WR%":>7}  {"Win%":>5}  {"Windows":>7}  {win_cols}'
    sep = '  ' + '-'*155

    # Gather all rows with aggregated stats
    all_rows = []
    for label, data in results[lg].items():
        a = agg(data)
        if not a or a['n_tot'] < 10 or a['n_windows'] < 4: continue
        all_rows.append({'label':label, **a, 'data':data})

    all_rows.sort(key=lambda x: x['avg_roi'], reverse=True)

    sections = [
        ('СТАБІЛЬНО ПРИБУТКОВІ',  lambda r: r['avg_roi']>0  and r['win_pct']>=60),
        ('ПОМІРНО ПРИБУТКОВІ',    lambda r: r['avg_roi']>0  and 40<=r['win_pct']<60),
        ('ЗБИТКОВІ (avg_roi<=0)', lambda r: r['avg_roi']<=0),
    ]

    for title, filt_fn in sections:
        sect = [r for r in all_rows if filt_fn(r)]
        out_lines.append(f'\n  ── {title}  ({len(sect)} фільтрів) ──')
        out_lines.append(hdr)
        out_lines.append(sep)
        for r in sect:
            # per-window cells: ROI%/n
            cells = []
            for wl in WIN_LABELS:
                if wl in r['data']:
                    d = r['data'][wl]
                    sign = '+' if d['roi'] >= 0 else ''
                    cells.append(f'{sign}{d["roi"]:.0f}%/n{d["n"]}')
                else:
                    cells.append('—')
            cells_str = '  '.join(f'{c:<12}' for c in cells)
            star = ' ★★' if r['avg_roi']>15 else (' ★' if r['avg_roi']>7 else '')
            out_lines.append(
                f'  {r["label"]:<54}  {r["n_tot"]:>5}  {r["avg_roi"]:>+7.1f}%  '
                f'{r["avg_wr"]:>6.1f}%  {r["win_pct"]:>4.0f}%  '
                f'{r["n_pos"]:>2}/{r["n_windows"]:>2}win  '
                f'{cells_str}{star}'
            )

outpath = os.path.join(os.path.dirname(__file__), 'walkforward_league_full.txt')
with open(outpath, 'w') as f:
    f.write('\n'.join(out_lines))
print(f"Saved: {outpath}  ({os.path.getsize(outpath)//1024}KB, {len(out_lines)} lines)")

# ── Console: condensed summary ─────────────────────────────────────────────────
print()
print('='*130)
print('WALK-FORWARD SUMMARY — топ-5 стабільних фільтрів per ліга (avg_roi>0, win%>=60%)')
print('='*130)
print(f'{"Ліга":<11}  {"Filter":<52}  {"n":>5}  {"Avg%":>6}  {"WR%":>5}  {"Win%":>5}  ' + '  '.join(WIN_LABELS))
print('-'*130)

for lg in LEAGUES:
    lname = LEAGUE_SHORT[lg]
    stable = []
    for label, data in results[lg].items():
        a = agg(data)
        if not a: continue
        if a['n_tot']<10 or a['n_windows']<4: continue
        if a['avg_roi']>0 and a['win_pct']>=60:
            stable.append({'label':label,'data':data,**a})
    stable.sort(key=lambda x: x['avg_roi'], reverse=True)

    if stable:
        for i, r in enumerate(stable[:5]):
            pref = f'{lname:<11}' if i==0 else f'{"":11}'
            cells = []
            for wl in WIN_LABELS:
                if wl in r['data']:
                    rv = r['data'][wl]['roi']
                    s = '+' if rv>=0 else ''
                    cells.append(f'{s}{rv:.0f}%')
                else:
                    cells.append('—')
            cells_str = '  '.join(f'{c:<6}' for c in cells)
            print(f'{pref}  {r["label"]:<52}  {r["n_tot"]:>5}  {r["avg_roi"]:>+5.1f}%  '
                  f'{r["avg_wr"]:>4.1f}%  {r["win_pct"]:>4.0f}%  {cells_str}')
        if len(stable) > 5:
            print(f'{"":11}  ... ще {len(stable)-5} стабільних фільтрів')
        print()
    else:
        print(f'{lname:<11}  — немає фільтрів (avg_roi>0, win%>=60%, n>=10, windows>=4)\n')
