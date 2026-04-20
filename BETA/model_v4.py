"""
BETA/model_v4.py

V4 — Portfolio of Niches. Кожна ніша тестується окремо walk-forward.

Ніші (з research_clusters.py):
  A: Away [1.55-1.80] + elo<=-30               (in-sample: WR=62.7%, kelly=1.28x)
  B: Home [1.80-2.20] + xg>=1.5 + form>=1.8   (in-sample: WR=60.4%, kelly=2.13x)
  C: Home [1.30-1.55] + xg>=1.5 + form>=1.8   (in-sample: WR=75.6%, kelly=1.87x)
  D: Away [2.20-2.80] + mkt>=0.40 + xg>=1.2   (in-sample: WR=50.9%, kelly=1.90x) — під питанням

Для кожної ніші:
  1. Pre-filter (odds + сигнальні умови)
  2. LightGBM binary → ранжування всередині ніші
  3. Top CONFIDENCE_PERCENTILE% за prob
  4. EV > 0
  5. Kelly 25% fractional, cap індивідуальний
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

# ── Shared config ──────────────────────────────────────────────────────────────
TRAIN_MONTHS         = 12
TEST_MONTHS          = 3
STEP_MONTHS          = 3
INITIAL_BANKROLL     = 1000.0
KELLY_FRACTIONAL     = 0.25
CONFIDENCE_PERCENTILE = 40   # top 40% within pre-filtered set

MARKET_COLS = {
    'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
    'home_odds_val','draw_odds_val','away_odds_val',
}

# ── Ніші ──────────────────────────────────────────────────────────────────────
NICHES = {
    'A_away_mild': {
        # OOS: 65.1% WR, +12.3% ROI при cap=5%. cap=6% дав гірший результат → повертаємо 5%.
        'label':   'A: Away [1.55-1.80] + elo<=-30',
        'side':    'away',
        'odds_col': 'away_odds_val',
        'min_odds': 1.55,
        'max_odds': 1.80,
        'kelly_cap': 0.05,
        'min_train': 80,
        'filter': lambda df: (
            (df['elo_diff'].fillna(0) <= -30) &
            (df['away_odds_val'] >= 1.55) & (df['away_odds_val'] < 1.80)
        ),
    },
    'B_home_value': {
        # Урок: xg>=1.3 без form дає шум (-20% ROI). Повертаємо xg>=1.5,
        # додаємо market confirmation mkt>=0.45 (ринок теж ставить на home при odds 1.8-2.2).
        'label':   'B: Home [1.80-2.20] + xg>=1.5 + mkt>=0.45',
        'side':    'home',
        'odds_col': 'home_odds_val',
        'min_odds': 1.80,
        'max_odds': 2.20,
        'kelly_cap': 0.07,
        'min_train': 60,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['mkt_home_prob'].fillna(0) >= 0.45) &
            (df['home_odds_val'] >= 1.80) & (df['home_odds_val'] < 2.20)
        ),
    },
    'C_home_fav': {
        # Урок: послаблення фільтру вбило нішу (79%→30% в одному периоді).
        # Тісний фільтр xg>=1.5 + form>=1.8 — це і є сигнал, повертаємо.
        'label':   'C: Home [1.30-1.55] + xg>=1.5 + form>=1.8',
        'side':    'home',
        'odds_col': 'home_odds_val',
        'min_odds': 1.30,
        'max_odds': 1.55,
        'kelly_cap': 0.06,
        'min_train': 100,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['home_pts_5'].fillna(0) >= 1.8) &
            (df['home_odds_val'] >= 1.30) & (df['home_odds_val'] < 1.55)
        ),
    },
    'D_home_mild': {
        # Нова ніша OOS: 75.0% WR, +31.0% ROI, DD 14%. Найкращий результат.
        # xG домінування + market confirmation = дуже сильний спільний сигнал.
        'label':   'D: Home [1.55-1.80] + xg>=1.5 + mkt>=0.50',
        'side':    'home',
        'odds_col': 'home_odds_val',
        'min_odds': 1.55,
        'max_odds': 1.80,
        'kelly_cap': 0.06,
        'min_train': 80,
        'filter': lambda df: (
            (df['xg_ratio_home_5'].fillna(0) >= 1.5) &
            (df['mkt_home_prob'].fillna(0) >= 0.50) &
            (df['home_odds_val'] >= 1.55) & (df['home_odds_val'] < 1.80)
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


# ── Kelly ──────────────────────────────────────────────────────────────────────
def kelly_stake(prob, odds, bankroll, cap):
    b = odds - 1.0
    q = 1.0 - prob
    f = max(0.0, (prob * b - q) / b) * KELLY_FRACTIONAL
    return min(bankroll * f, bankroll * cap)


# ── Walk-forward для однієї ніші ───────────────────────────────────────────────
def run_niche(df, feat_cols, niche_id, cfg, verbose=True):
    train_feat_cols = [c for c in feat_cols if c not in MARKET_COLS]
    df = df.sort_values('date').reset_index(drop=True)

    min_date = df['date'].min()
    max_date = df['date'].max()

    splits = []
    ts = min_date + relativedelta(months=TRAIN_MONTHS)
    while ts + relativedelta(months=TEST_MONTHS) <= max_date + relativedelta(days=1):
        splits.append((ts, ts + relativedelta(months=TEST_MONTHS)))
        ts += relativedelta(months=STEP_MONTHS)

    side     = cfg['side']
    odds_col = cfg['odds_col']
    win_col  = 'result'
    win_val  = 'H' if side == 'home' else 'A'
    cap      = cfg['kelly_cap']
    filt     = cfg['filter']
    min_tr   = cfg['min_train']

    all_bets     = []
    period_stats = []

    for test_start, test_end in splits:
        train_all = df[df['date'] < test_start].copy()
        test_all  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        train_niche = train_all[filt(train_all)].copy()
        test_niche  = test_all[filt(test_all) & test_all[odds_col].notna()].copy()

        if len(train_niche) < min_tr or len(test_niche) < 3:
            period_stats.append({
                'test_start': test_start, 'n_train': len(train_niche),
                'n_bets': 0, 'win_rate': 0, 'skipped': True,
            })
            continue

        y_train  = (train_niche[win_col] == win_val).astype(int).values
        cal_size = max(30, int(len(train_niche) * 0.20))
        X_tr  = train_niche[train_feat_cols].fillna(0).values[:-cal_size]
        y_tr  = y_train[:-cal_size]
        X_cal = train_niche[train_feat_cols].fillna(0).values[-cal_size:]
        y_cal = y_train[-cal_size:]

        if len(np.unique(y_cal)) < 2 or len(np.unique(y_tr)) < 2:
            period_stats.append({
                'test_start': test_start, 'n_train': len(train_niche),
                'n_bets': 0, 'win_rate': 0, 'skipped': True,
            })
            continue

        booster, cal_model = fit_lgbm(X_tr, y_tr, X_cal, y_cal)
        X_test = test_niche[train_feat_cols].fillna(0).values
        probs  = predict_prob(booster, cal_model, X_test)

        thr = np.percentile(probs, 100 - CONFIDENCE_PERCENTILE)

        period_bets = []
        for i, (_, row) in enumerate(test_niche.iterrows()):
            p    = probs[i]
            odds = row[odds_col]
            ev   = p * odds - 1

            if p < thr or ev < 0:
                continue

            won = (row[win_col] == win_val)
            period_bets.append({
                'match_id': int(row['match_id']),
                'date':     row['date'],
                'side':     side,
                'prob':     round(float(p), 4),
                'odds':     round(float(odds), 2),
                'ev':       round(float(ev), 4),
                'result':   'win' if won else 'loss',
            })

        n  = len(period_bets)
        nw = sum(1 for b in period_bets if b['result'] == 'win')
        wr = nw / n * 100 if n > 0 else 0

        period_stats.append({
            'test_start': test_start, 'n_train': len(train_niche),
            'n_bets': n, 'win_rate': round(wr, 1), 'skipped': False,
        })
        all_bets.extend(period_bets)

        if verbose:
            print(f"    {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"train={len(train_niche):4d} | selected={n:3d} | win={wr:.0f}%")

    # Bankroll simulation
    all_bets.sort(key=lambda b: b['date'])
    bankroll = INITIAL_BANKROLL
    peak     = INITIAL_BANKROLL
    max_dd   = 0.0
    series   = [bankroll]
    n_wins = n_losses = 0

    for bet in all_bets:
        stake = kelly_stake(bet['prob'], bet['odds'], bankroll, cap)
        if bet['result'] == 'win':
            bankroll += stake * (bet['odds'] - 1)
            n_wins += 1
        else:
            bankroll -= stake
            n_losses += 1
        bankroll = max(bankroll, 0.01)
        peak     = max(peak, bankroll)
        max_dd   = max(max_dd, (peak - bankroll) / peak * 100)
        series.append(round(bankroll, 2))

    n_bets = n_wins + n_losses
    return {
        'niche_id':        niche_id,
        'label':           cfg['label'],
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
        'kelly_cap':       cap,
    }


# ── Print results ──────────────────────────────────────────────────────────────
def print_niche_report(r):
    x = r['final_bankroll'] / INITIAL_BANKROLL
    verdict = '✅ PROFITABLE' if r['roi'] > 0 else '❌ LOSING'
    print(f"\n{'─'*60}")
    print(f"  {r['label']}")
    print(f"  {verdict}")
    print(f"{'─'*60}")
    print(f"  Bets:      {r['n_bets']} ({r['n_wins']}W / {r['n_losses']}L)")
    print(f"  Win rate:  {r['win_rate']:.1f}%")
    print(f"  Avg odds:  {r['avg_odds']:.2f}")
    print(f"  Avg EV:    {r['avg_ev']:.1f}%")
    print(f"  Bankroll:  ${INITIAL_BANKROLL:.0f} → ${r['final_bankroll']:.0f}  ({x:.2f}x)")
    print(f"  ROI:       {r['roi']:+.1f}%")
    print(f"  Max DD:    {r['max_drawdown']:.1f}%")
    print(f"  Kelly cap: {r['kelly_cap']*100:.0f}%")

    print(f"\n  Per period:")
    for p in r['period_stats']:
        if p.get('skipped'):
            print(f"    {p['test_start'].strftime('%Y-%m')} | train={p['n_train']:4d} | SKIPPED")
        else:
            print(f"    {p['test_start'].strftime('%Y-%m')} | "
                  f"train={p['n_train']:4d} | bets={p['n_bets']:3d} | win={p['win_rate']:5.1f}%")


def print_summary(results):
    print(f"\n\n{'='*70}")
    print(f"  PORTFOLIO SUMMARY — Walk-forward OOS Results")
    print(f"{'='*70}")
    print(f"  {'Niche':<45} {'Bets':>5} {'WR':>6} {'Odds':>6} {'ROI':>8} {'x':>6} {'DD':>6}")
    print(f"  {'-'*68}")

    profitable = []
    for r in results:
        if r['n_bets'] == 0:
            print(f"  {r['label']:<45} {'NO BETS':>5}")
            continue
        x = r['final_bankroll'] / INITIAL_BANKROLL
        marker = ' ✅' if r['roi'] > 0 else ' ❌'
        print(f"  {r['label']:<45} {r['n_bets']:>5} {r['win_rate']:>5.1f}% "
              f"{r['avg_odds']:>6.2f} {r['roi']:>+7.1f}% {x:>5.2f}x {r['max_drawdown']:>5.1f}%{marker}")
        if r['roi'] > 0:
            profitable.append(r)

    # Combined bankroll (independent niches, separate bankrolls)
    print(f"\n  Profitable niches: {len(profitable)}/{len(results)}")
    if profitable:
        # Simple combined: average final bankroll of profitable niches
        avg_x = np.mean([r['final_bankroll'] / INITIAL_BANKROLL for r in profitable])
        print(f"  Avg x-factor (profitable): {avg_x:.2f}x")

        # Estimate combined if $1000 split equally
        n = len(profitable)
        split = INITIAL_BANKROLL / n
        combined_final = sum(split * (r['final_bankroll'] / INITIAL_BANKROLL) for r in profitable)
        print(f"  Combined (${INITIAL_BANKROLL:.0f} split {n} ways): ${combined_final:.0f}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("  BETA v4 — Portfolio of Niches")
    print("=" * 60)

    matches, stats, odds, injuries = load_all()
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)

    print(f"  {len(df)} matches | {len(feat_cols)} features")
    print(f"  Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    df = df[df['home_odds_val'].notna() | df['away_odds_val'].notna()].copy()
    print(f"  With odds: {len(df)}\n")

    results = []
    for niche_id, cfg in NICHES.items():
        print(f"\n[{niche_id}] {cfg['label']}")
        r = run_niche(df, feat_cols, niche_id, cfg, verbose=True)
        print_niche_report(r)
        results.append(r)

    print_summary(results)
    print(f"\n  Total time: {time.time()-t0:.0f}s")
