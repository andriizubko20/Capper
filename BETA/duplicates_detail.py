import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re, sys
sys.path.insert(0,'.')
from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix

FLAT_STAKE = 50.0
OOS_START  = pd.Timestamp('2025-08-01')

MODELS = {
    'Premier League': ['home[2.5,3.5) xg>=1.0','home[2.2,2.8) xg>=1.2 form>=1.5','home[1.8,2.2) form>=2.2','home[1.7,2.0) xg>=1.8','away[1.55,1.8) xg>=1.2 elo<=-75 form>=2.2'],
    'Bundesliga': ['home[2.2,2.8) xg>=1.2 form>=1.5','home[1.8,2.2) xg>=1.2 elo>=30 mkt>=0.5','home[1.8,2.2) elo>=75 form>=1.5 mkt>=0.5','home[1.55,1.8) elo>=150 form>=1.5','away[2.5,3.5) xg>=1.0 elo<=-30 form>=1.5','away[2.5,3.5) form>=1.5 mkt>=0.35','away[2.5,3.5) elo<=-75 mkt>=0.35','away[2.5,3.5) elo<=-30 form>=1.5 mkt>=0.35','away[2.2,2.8) xg>=1.5 mkt>=0.4','away[2.2,2.8) mkt>=0.4','away[2.2,2.8) elo<=-75 mkt>=0.35','away[2.0,2.5) mkt>=0.4','away[1.7,2.0) xg>=1.0 form>=2.2'],
    'Serie A': ['home[2.5,3.5) elo>=30 form>=1.5','home[2.0,2.5) xg>=1.0 mkt>=0.45','home[2.0,2.5) form>=2.2 mkt>=0.45','home[1.8,2.2) xg>=1.5 form>=1.5','home[1.3,1.55) xg>=1.5 elo>=150','away[2.5,3.5) xg>=1.0 form>=1.8 mkt>=0.35','away[2.2,2.8) xg>=1.2 form>=1.8 mkt>=0.35','away[2.2,2.8) xg>=1.0 elo<=-150','away[1.8,2.2) xg>=1.5 form>=2.2'],
    'La Liga': ['home[2.2,2.8) xg>=1.0 form>=1.5','home[2.0,2.5) xg>=1.0 form>=1.5','home[1.8,2.2) xg>=1.2 form>=1.5','home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5','home[1.8,2.2) xg>=1.0 elo>=30 mkt>=0.5','home[1.55,1.8) xg>=1.8','away[2.5,3.5) xg>=1.8 form>=1.5','away[2.5,3.5) elo<=-75 form>=2.2','away[2.2,2.8) xg>=1.2 mkt>=0.4','away[1.8,2.2) xg>=1.8 elo<=-75 mkt>=0.45'],
    'Ligue 1': ['home[2.5,3.5) xg>=1.5','home[2.2,2.8) xg>=1.2','home[2.0,2.5) elo>=75','home[1.7,2.0) xg>=1.8','home[1.7,2.0) xg>=1.5 mkt>=0.5','home[1.7,2.0) xg>=1.5 elo>=75','home[1.7,2.0) xg>=1.5 elo>=30','home[1.55,1.8) xg>=1.2 elo>=75','away[2.5,3.5) xg>=1.0 form>=2.2','away[2.2,2.8) form>=1.5 mkt>=0.4','away[2.2,2.8) elo<=-30 form>=1.5 mkt>=0.4'],
    'Primeira Liga': ['away[2.2,2.8) xg>=1.2 mkt>=0.4','home[2.0,2.5) form>=1.5 mkt>=0.45','home[1.55,1.8) elo>=150','home[1.8,2.2) form>=2.2'],
    'Serie B': ['home[2.2,2.8) form>=2.2','home[2.0,2.5) elo>=75 form>=2.2','home[1.7,2.0) xg>=1.0 form>=2.2','home[1.55,1.8) xg>=1.8','home[1.55,1.8) xg>=1.5 elo>=75 form>=2.2','home[1.55,1.8) xg>=1.2 elo>=150','home[1.55,1.8) xg>=1.5','away[2.0,2.5) xg>=1.5 elo<=-30'],
    'Eredivisie': ['home[2.5,3.5) xg>=1.0','home[2.0,2.5) elo>=75','away[2.5,3.5) elo<=-75 mkt>=0.35','away[1.8,2.2) xg>=1.2 form>=2.2','away[1.7,2.0) xg>=1.2 form>=2.2','away[1.55,1.8) elo<=-150'],
    'Jupiler Pro League': ['home[2.5,3.5) form>=2.2','home[2.0,2.5) xg>=1.0 mkt>=0.45','home[1.8,2.2) xg>=1.2 mkt>=0.5','home[1.8,2.2) xg>=1.0 form>=1.5 mkt>=0.5','home[1.7,2.0) xg>=1.0 form>=1.5 mkt>=0.5','home[1.55,1.8) xg>=1.8','home[1.55,1.8) form>=2.2','home[1.3,1.55) xg>=1.5 elo>=150 form>=1.5','away[2.2,2.8) xg>=1.2 form>=2.2 mkt>=0.35','away[2.0,2.5) xg>=1.5 form>=2.2'],
    'Champions League': ['home[2.2,2.8) xg>=1.8 form>=1.5','home[2.0,2.5) form>=1.5 mkt>=0.45','home[1.8,2.2) mkt>=0.5','home[1.7,2.0) xg>=1.5 mkt>=0.55','home[1.7,2.0) xg>=1.5 elo>=75 mkt>=0.5','home[1.55,1.8) elo>=30 form>=2.2','home[1.55,1.8) elo>=150','home[1.3,1.55) xg>=1.8 form>=1.8','away[2.2,2.8) xg>=1.8 mkt>=0.4','away[2.2,2.8) xg>=1.5 form>=1.8 mkt>=0.4'],
}

def parse_niche(s):
    m=re.match(r'(home|away)\[(\d+\.?\d*),(\d+\.?\d*)\)',s)
    side=m.group(1); lo=float(m.group(2)); hi=float(m.group(3))
    xg=0.0; elo=0; fm=0.0; mk=0.0
    mx=re.search(r'xg>=([\d.]+)',s);   xg=float(mx.group(1)) if mx else 0.0
    mx=re.search(r'elo>=([-\d]+)',s);  elo=int(mx.group(1)) if mx else elo
    mx=re.search(r'elo<=([-\d]+)',s);  elo=int(mx.group(1)) if mx else elo
    mx=re.search(r'form>=([\d.]+)',s); fm=float(mx.group(1)) if mx else 0.0
    mx=re.search(r'mkt>=([\d.]+)',s);  mk=float(mx.group(1)) if mx else 0.0
    return side,lo,hi,xg,elo,fm,mk

def apply_mask(data,side,lo,hi,xg,elo,fm,mk,league):
    oc='home_odds_val' if side=='home' else 'away_odds_val'
    m=(data['league_name']==league)&(data[oc]>=lo)&(data[oc]<hi)
    if xg>0:  m&=data['xg_ratio_home_5' if side=='home' else 'xg_ratio_away_5'].fillna(0)>=xg
    if side=='home' and elo>0: m&=data['elo_diff'].fillna(0)>=elo
    if side=='away' and elo<0: m&=data['elo_diff'].fillna(0)<=elo
    if fm>0:  m&=data['home_pts_5' if side=='home' else 'away_pts_5'].fillna(0)>=fm
    if mk>0:  m&=data['mkt_home_prob' if side=='home' else 'mkt_away_prob'].fillna(0)>=mk
    return data[m]

def calc_roi(bets, wc, oc):
    if len(bets)==0: return None, 0
    pl=(bets[wc]*(bets[oc]-1)-(1-bets[wc])).sum()*FLAT_STAKE
    n=len(bets)
    return pl/(n*FLAT_STAKE)*100, n

print('Loading data...')
matches, stats, odds_data, injuries = load_all()
df = build_feature_matrix(matches, stats, odds_data, injuries)
df = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
df['home_win'] = (df['result']=='H').astype(int)
df['away_win']  = (df['result']=='A').astype(int)
df['ym'] = df['date'].dt.to_period('M')
oos_p = OOS_START.to_period('M')

print()
print('='*90)

for league, niches in MODELS.items():
    # Build model info
    model_info = []
    for niche_str in niches:
        side,lo,hi,xg,elo,fm,mk = parse_niche(niche_str)
        wc='home_win' if side=='home' else 'away_win'
        oc='home_odds_val' if side=='home' else 'away_odds_val'
        bets = apply_mask(df,side,lo,hi,xg,elo,fm,mk,league)
        ids = set(bets.index.tolist())

        oos_bets = bets[bets['ym']>=oos_p]
        full_roi, full_n = calc_roi(bets, wc, oc)
        oos_roi, oos_n   = calc_roi(oos_bets, wc, oc)

        model_info.append({
            'niche': niche_str, 'wc': wc, 'oc': oc,
            'bets': bets, 'ids': ids,
            'full_n': full_n, 'full_roi': full_roi,
            'oos_n': oos_n, 'oos_roi': oos_roi,
        })

    # Map each match_id → list of models covering it
    idx_to_models = {}
    for mi in model_info:
        for idx in mi['ids']:
            idx_to_models.setdefault(idx, []).append(mi['niche'])

    # Find pure duplicates (u_n == 0)
    dups = []
    for mi in model_info:
        unique_ids = {idx for idx in mi['ids'] if len(idx_to_models[idx])==1}
        if len(unique_ids) == 0 and mi['full_n'] > 0:
            # Find which models cover its matches
            covering = {}
            for idx in mi['ids']:
                for other_niche in idx_to_models[idx]:
                    if other_niche != mi['niche']:
                        covering[other_niche] = covering.get(other_niche, 0) + 1
            # Sort by coverage %
            total = mi['full_n']
            covering_sorted = sorted(covering.items(), key=lambda x:-x[1])
            dups.append({
                'mi': mi,
                'covering': covering_sorted,
                'total': total,
            })

    if not dups:
        continue

    print(f'\n── {league} ── {len(dups)} дублів ──')
    print(f"{'Модель':<42} {'n':>4} {'OOS ROI':>8}  Покривається моделями (n матчів)")
    print('-'*90)

    for d in dups:
        mi = d['mi']
        oos_str = f"{mi['oos_roi']:+.1f}%" if mi['oos_roi'] is not None else '  —  '
        # Find best OOS ROI among covering models
        covering_details = []
        for cov_niche, cov_cnt in d['covering'][:3]:  # top 3 by coverage
            cov_mi = next((x for x in model_info if x['niche']==cov_niche), None)
            if cov_mi:
                cov_oos = f"{cov_mi['oos_roi']:+.1f}%" if cov_mi['oos_roi'] is not None else '—'
                covering_details.append(f"{cov_niche} [{cov_oos}, {cov_cnt}/{d['total']} матч]")

        # Is this dup ever the BEST model on a match?
        times_best = 0
        for idx in mi['ids']:
            models_on_match = idx_to_models[idx]
            all_oos_rois = []
            for m_niche in models_on_match:
                m_info = next((x for x in model_info if x['niche']==m_niche), None)
                if m_info and m_info['oos_roi'] is not None:
                    all_oos_rois.append((m_niche, m_info['oos_roi']))
            if all_oos_rois:
                best_niche = max(all_oos_rois, key=lambda x:x[1])[0]
                if best_niche == mi['niche']:
                    times_best += 1

        best_flag = f" ★ wins {times_best}/{d['total']}" if times_best > 0 else f" (ніколи не кращий)"
        print(f"  {mi['niche']:<40} {d['total']:>4} {oos_str:>8}{best_flag}")
        for cd in covering_details:
            print(f"    └─ {cd}")

print()
print('★ = ця модель має кращий OOS ROI на цьому матчі → при дедупі вона виграє')
print('(ніколи не кращий) → при дедупі завжди програє → можна прибрати')
