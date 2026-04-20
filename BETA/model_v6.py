"""
BETA/model_v6.py

V6 — Фінальний портфель. Єдиний банкрол, 5 ніш.

Ніші (відібрані зі sweep + підтверджені OOS де можливо):
  A: Away [1.55-1.80] + xg>=1.8 + elo<=-75 + form>=1.8   (sweep: WR=68.6%, ROI=+12.9%)
  B: Home [1.70-2.50] + xg>=1.5 + form>=1.5 + mkt>=0.45  (sweep: WR=63.8%, ROI=+17.9%, kelly=3.69x)
  C: Home [1.30-1.55] + xg>=1.5 + form>=1.8              (OOS confirmed: WR=79.2%, ROI=+37.6%)
  D: Away [2.20-2.80] + xg>=1.8 + mkt>=0.40              (sweep: WR=62.8%, ROI=+41.6%)
  E: Home [2.20-2.60] + xg>=1.3                           (sweep: WR=48.4%, ROI=+14.0%, odds=2.44)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from dateutil.relativedelta import relativedelta

from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_MONTHS          = 12
TEST_MONTHS           = 3
STEP_MONTHS           = 3
INITIAL_BANKROLL      = 1000.0
KELLY_FRACTIONAL      = 0.25
CONFIDENCE_PERCENTILE = 40

MARKET_COLS = {
    'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
    'home_odds_val','draw_odds_val','away_odds_val',
}

NICHES = {
    'A': {
        'label':    'A: Away [1.55-1.80] + xg>=1.8 + elo<=-75 + form>=1.8',
        'side':     'away',
        'odds_col': 'away_odds_val',
        'kelly_cap': 0.05,
        'min_train': 60,
        'filter': lambda df: (
            (df['xg_ratio_away_5'].fillna(0) >= 1.8) &
            (df['elo_diff'].fillna(0) <= -75) &
            (df['away_pts_5'].fillna(0) >= 1.8) &
            (df['away_odds_val'] >= 1.55) & (df['away_odds_val'] < 1.80)
        ),
    },
    'B': {
        'label':    'B: Home [1.70-2.50] + xg>=1.5 + form>=1.5 + mkt>=0.45',
        'side':     'home',
        'odds_col': 'home_odds_val',
        'kelly_cap': 0.06,
        'min_train': 80,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['home_pts_5'].fillna(0) >= 1.5) &
            (df['mkt_home_prob'].fillna(0) >= 0.45) &
            (df['home_odds_val'] >= 1.70) & (df['home_odds_val'] < 2.50)
        ),
    },
    'C': {
        'label':    'C: Home [1.30-1.55] + xg>=1.5 + form>=1.8',
        'side':     'home',
        'odds_col': 'home_odds_val',
        'kelly_cap': 0.06,
        'min_train': 100,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['home_pts_5'].fillna(0) >= 1.8) &
            (df['home_odds_val'] >= 1.30) & (df['home_odds_val'] < 1.55)
        ),
    },
    'D': {
        'label':    'D: Away [2.20-2.80] + xg>=1.8 + mkt>=0.40',
        'side':     'away',
        'odds_col': 'away_odds_val',
        'kelly_cap': 0.06,
        'min_train': 30,
        'filter': lambda df: (
            (df['xg_ratio_away_5'].fillna(0) >= 1.8) &
            (df['mkt_away_prob'].fillna(0) >= 0.40) &
            (df['away_odds_val'] >= 2.20) & (df['away_odds_val'] < 2.80)
        ),
    },
    'E': {
        'label':    'E: Home [2.20-2.60] + xg>=1.3',
        'side':     'home',
        'odds_col': 'home_odds_val',
        'kelly_cap': 0.05,
        'min_train': 60,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.3) &
            (df['home_odds_val'] >= 2.20) & (df['home_odds_val'] < 2.60)
        ),
    },
}


# ── LightGBM ───────────────────────────────────────────────────────────────────
def fit_lgbm(X_tr, y_tr, X_cal, y_cal):
    params = {
        'objective':         'binary',
        'num_leaves':        31,
        'learning_rate':     0.03,
        'min_child_samples': 10,
        'subsample':         0.8,
        'colsample_bytree':  0.8,
        'reg_alpha':         0.1,
        'reg_lambda':        0.5,
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(params, dtrain, num_boost_round=400,
                        callbacks=[lgb.log_evaluation(period=-1)])
    raw = booster.predict(X_cal).reshape(-1, 1)
    cal = LogisticRegression(max_iter=300, C=1.0)
    cal.fit(raw, y_cal)
    return booster, cal


def predict_prob(booster, cal, X):
    raw = booster.predict(X).reshape(-1, 1)
    return cal.predict_proba(raw)[:, 1]


# ── Walk-forward ───────────────────────────────────────────────────────────────
def run_portfolio(df, feat_cols, verbose=True):
    train_feat_cols = [c for c in feat_cols if c not in MARKET_COLS]
    df = df.sort_values('date').reset_index(drop=True)

    min_date = df['date'].min()
    max_date = df['date'].max()

    splits = []
    ts = min_date + relativedelta(months=TRAIN_MONTHS)
    while ts + relativedelta(months=TEST_MONTHS) <= max_date + relativedelta(days=1):
        splits.append((ts, ts + relativedelta(months=TEST_MONTHS)))
        ts += relativedelta(months=STEP_MONTHS)

    if verbose:
        print(f"Walk-forward: {len(splits)} periods | "
              f"train={TRAIN_MONTHS}m test={TEST_MONTHS}m step={STEP_MONTHS}m")
        print(f"Niches: {list(NICHES.keys())} | Confidence top {CONFIDENCE_PERCENTILE}%\n")

    all_bets     = []
    period_stats = []

    for test_start, test_end in splits:
        train_all = df[df['date'] < test_start].copy()
        test_all  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        period_bets  = []
        period_info  = {'test_start': test_start, 'niches': {}}
        used_matches = set()

        for nid, cfg in NICHES.items():
            side     = cfg['side']
            odds_col = cfg['odds_col']
            win_val  = 'H' if side == 'home' else 'A'
            filt     = cfg['filter']
            cap      = cfg['kelly_cap']

            train_n = train_all[filt(train_all)].copy()
            test_n  = test_all[
                filt(test_all) & test_all[odds_col].notna()
            ].copy()
            test_n = test_n[~test_n['match_id'].isin(used_matches)]

            if len(train_n) < cfg['min_train'] or len(test_n) < 2:
                period_info['niches'][nid] = {'n': 0, 'skipped': True}
                continue

            y_train  = (train_n['result'] == win_val).astype(int).values
            cal_size = max(20, int(len(train_n) * 0.20))
            X_tr  = train_n[train_feat_cols].fillna(0).values[:-cal_size]
            y_tr  = y_train[:-cal_size]
            X_cal = train_n[train_feat_cols].fillna(0).values[-cal_size:]
            y_cal = y_train[-cal_size:]

            if len(np.unique(y_cal)) < 2 or len(np.unique(y_tr)) < 2:
                period_info['niches'][nid] = {'n': 0, 'skipped': True}
                continue

            booster, cal_model = fit_lgbm(X_tr, y_tr, X_cal, y_cal)
            X_test = test_n[train_feat_cols].fillna(0).values
            probs  = predict_prob(booster, cal_model, X_test)
            thr    = np.percentile(probs, 100 - CONFIDENCE_PERCENTILE)

            niche_bets = []
            for i, (_, row) in enumerate(test_n.iterrows()):
                p    = probs[i]
                odds = row[odds_col]
                ev   = p * odds - 1
                if p < thr or ev < 0:
                    continue
                mid = int(row['match_id'])
                won = (row['result'] == win_val)
                niche_bets.append({
                    'match_id':  mid,
                    'date':      row['date'],
                    'niche':     nid,
                    'side':      side,
                    'prob':      round(float(p), 4),
                    'odds':      round(float(odds), 2),
                    'ev':        round(float(ev), 4),
                    'kelly_cap': cap,
                    'result':    'win' if won else 'loss',
                })
                used_matches.add(mid)

            n  = len(niche_bets)
            nw = sum(1 for b in niche_bets if b['result'] == 'win')
            period_info['niches'][nid] = {
                'n': n, 'wins': nw, 'skipped': False,
                'wr': round(nw / n * 100, 1) if n > 0 else 0,
            }
            period_bets.extend(niche_bets)

        n_total  = len(period_bets)
        nw_total = sum(1 for b in period_bets if b['result'] == 'win')
        period_info['n_total'] = n_total
        period_info['wr_total'] = round(nw_total / n_total * 100, 1) if n_total > 0 else 0
        period_stats.append(period_info)
        all_bets.extend(period_bets)

        if verbose:
            parts = []
            for nid, v in period_info['niches'].items():
                val = 'skip' if v.get('skipped') else f"{v['wr']:.0f}%"
                parts.append(f"{nid}={v['n']}({val})")
            print(f"  {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"total={n_total:3d} win={period_info['wr_total']:.0f}%  |  "
                  + '  '.join(parts))

    # ── Єдиний банкрол ─────────────────────────────────────────────────────────
    all_bets.sort(key=lambda b: b['date'])
    bankroll = INITIAL_BANKROLL
    peak     = INITIAL_BANKROLL
    max_dd   = 0.0
    series   = [bankroll]
    n_wins = n_losses = 0
    niche_stats = {nid: {'wins': 0, 'losses': 0, 'profit': 0.0} for nid in NICHES}

    for bet in all_bets:
        b     = bet['odds'] - 1.0
        q     = 1.0 - bet['prob']
        f     = max(0.0, (bet['prob'] * b - q) / b) * KELLY_FRACTIONAL
        stake = min(bankroll * f, bankroll * bet['kelly_cap'])
        stake = round(stake, 2)

        if bet['result'] == 'win':
            profit = stake * b
            bankroll += profit
            n_wins += 1
            niche_stats[bet['niche']]['wins']   += 1
            niche_stats[bet['niche']]['profit'] += profit
        else:
            bankroll -= stake
            n_losses += 1
            niche_stats[bet['niche']]['losses'] += 1
            niche_stats[bet['niche']]['profit'] -= stake

        bankroll = max(bankroll, 0.01)
        peak     = max(peak, bankroll)
        max_dd   = max(max_dd, (peak - bankroll) / peak * 100)
        series.append(round(bankroll, 2))

    n_bets = n_wins + n_losses
    test_months = len(period_stats) * STEP_MONTHS

    return {
        'all_bets':        all_bets,
        'period_stats':    period_stats,
        'bankroll_series': series,
        'final_bankroll':  round(bankroll, 2),
        'roi':             round((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100, 2),
        'win_rate':        round(n_wins / n_bets * 100, 1) if n_bets > 0 else 0,
        'max_drawdown':    round(max_dd, 1),
        'n_bets':          n_bets,
        'n_wins':          n_wins,
        'n_losses':        n_losses,
        'avg_odds':        round(np.mean([b['odds'] for b in all_bets]), 2) if all_bets else 0,
        'avg_ev':          round(np.mean([b['ev'] for b in all_bets]) * 100, 1) if all_bets else 0,
        'niche_stats':     niche_stats,
        'test_months':     test_months,
    }


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(r):
    x    = r['final_bankroll'] / INITIAL_BANKROLL
    tm   = r['test_months']
    bpy  = r['n_bets'] / (tm / 12) if tm > 0 else 0
    proj = INITIAL_BANKROLL * (x ** (24 / tm)) if tm > 0 and x > 0 else 0

    print(f"\n{'='*65}")
    print(f"  BETA v6 — Portfolio Single Bankroll")
    print(f"{'='*65}")
    print(f"  Bets:       {r['n_bets']} ({r['n_wins']}W / {r['n_losses']}L)")
    print(f"  Win rate:   {r['win_rate']:.1f}%")
    print(f"  Avg odds:   {r['avg_odds']:.2f}")
    print(f"  Avg EV:     {r['avg_ev']:.1f}%")
    print(f"  Bankroll:   ${INITIAL_BANKROLL:.0f} → ${r['final_bankroll']:.0f}  ({x:.2f}x)")
    print(f"  ROI:        {r['roi']:+.1f}%")
    print(f"  Max DD:     {r['max_drawdown']:.1f}%")
    print(f"  Bets/year:  {bpy:.0f}")
    print(f"  2yr proj:   ${proj:.0f}")

    print(f"\n  По нішах:")
    print(f"  {'Ніша':<50} {'Ставок':>7} {'WR':>6} {'Profit':>9} {'Verdict':>8}")
    print(f"  {'-'*82}")
    for nid, cfg in NICHES.items():
        ns = r['niche_stats'][nid]
        n  = ns['wins'] + ns['losses']
        wr = ns['wins'] / n * 100 if n > 0 else 0
        verdict = '✅' if ns['profit'] > 0 else '❌'
        print(f"  {cfg['label']:<50} {n:>7} {wr:>5.1f}% ${ns['profit']:>+8.1f}  {verdict}")

    print(f"\n  По периодах:")
    for p in r['period_stats']:
        parts = []
        for nid, v in p['niches'].items():
            val = 'skip' if v.get('skipped') else f"{v['wr']:.0f}%"
            parts.append(f"{nid}={v['n']}({val})")
        print(f"    {p['test_start'].strftime('%Y-%m')} | "
              f"total={p['n_total']:3d} win={p['wr_total']:5.1f}%  |  "
              + '  '.join(parts))

    series     = r['bankroll_series']
    milestones = [1500, 2000, 3000, 5000, 7500, 10000]
    seen, reached = set(), []
    for i, b in enumerate(series):
        for m in milestones:
            if b >= m and m not in seen:
                reached.append((m, i)); seen.add(m)
    if reached:
        print(f"\n  Milestones:")
        for target, idx in reached:
            print(f"    ${target:>6} → після ставки #{idx}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 65)
    print("  BETA v6 — Loading data...")
    print("=" * 65)

    matches, stats, odds, injuries = load_all()
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)

    print(f"  {len(df)} matches | {len(feat_cols)} features")
    print(f"  Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    df = df[df['home_odds_val'].notna() | df['away_odds_val'].notna()].copy()
    print(f"  With odds: {len(df)}\n")

    result = run_portfolio(df, feat_cols, verbose=True)
    print_report(result)
    print(f"\n  Total time: {time.time()-t0:.0f}s")
