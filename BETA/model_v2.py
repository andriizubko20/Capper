"""
BETA/model_v2.py

V2 Selective Betting Model — walk-forward backtest.

Strategy:
  1. Pre-filter: only home bets where xg_ratio_home_5 >= XG_THRESHOLD
     (empirically proven 67.7% win rate at odds 1.69, +3.1% flat ROI)
  2. Binary LightGBM: P(home_win) trained WITHOUT market features
  3. Confidence ranking: take only top CONFIDENCE_PERCENTILE % by model prob
  4. EV check: model_prob * odds - 1 > 0
  5. Kelly 25% fractional, cap at MAX_STAKE_PCT

Goal: 60-65%+ win rate, $4-5k from $1000 initial over backtest period.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from dateutil.relativedelta import relativedelta
from collections import defaultdict

from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols

# ── Config ─────────────────────────────────────────────────────────────────────

XG_THRESHOLD         = 1.5    # minimum xg_ratio_home_5 to consider
CONFIDENCE_PERCENTILE = 35    # take top 35% by model confidence within pre-filtered set
MIN_EV               = 0.0    # EV > 0 (positive expected value)
MIN_ODDS             = 1.70   # focus on value odds range
MAX_ODDS             = 2.50   # maximum odds
INITIAL_BANKROLL     = 1000.0
KELLY_FRACTIONAL     = 0.25
MAX_STAKE_PCT        = 0.04   # 4% cap per bet
TRAIN_MONTHS         = 12
TEST_MONTHS          = 3
STEP_MONTHS          = 3

# Market features excluded from model training (used only for EV calc)
MARKET_COLS = {
    'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
    'home_odds_val','draw_odds_val','away_odds_val',
}


# ── LightGBM binary model ──────────────────────────────────────────────────────

def fit_lgbm(X_train, y_train, X_cal, y_cal):
    """Train LightGBM binary classifier + Platt calibration."""
    params = {
        'objective':         'binary',
        'num_leaves':        31,
        'learning_rate':     0.03,
        'n_estimators':      500,
        'min_child_samples': 15,
        'subsample':         0.8,
        'colsample_bytree':  0.8,
        'reg_alpha':         0.1,
        'reg_lambda':        0.5,
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(
        params, dtrain,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    fi = booster.feature_importance(importance_type='gain')

    # Platt calibration on held-out cal set
    raw_cal = booster.predict(X_cal).reshape(-1, 1)
    cal = LogisticRegression(max_iter=300, C=1.0)
    cal.fit(raw_cal, y_cal)

    return booster, cal, fi


def predict_prob(booster, cal, X):
    raw = booster.predict(X).reshape(-1, 1)
    return cal.predict_proba(raw)[:, 1]


# ── Kelly sizing ────────────────────────────────────────────────────────────────

def kelly_stake(prob, odds, bankroll, fractional=0.25, cap=0.04):
    b = odds - 1.0
    q = 1.0 - prob
    f = max(0.0, (prob * b - q) / b) * fractional
    return min(bankroll * f, bankroll * cap)


# ── Walk-forward backtest ───────────────────────────────────────────────────────

def run_backtest(df, feat_cols, verbose=True):
    df = df.sort_values('date').reset_index(drop=True)
    train_feat_cols = [c for c in feat_cols if c not in MARKET_COLS]

    min_date = df['date'].min()
    max_date = df['date'].max()

    # Build time splits
    splits = []
    ts = min_date + relativedelta(months=TRAIN_MONTHS)
    while ts + relativedelta(months=TEST_MONTHS) <= max_date + relativedelta(days=1):
        splits.append((ts, ts + relativedelta(months=TEST_MONTHS)))
        ts += relativedelta(months=STEP_MONTHS)

    if verbose:
        print(f"\nWalk-forward: {len(splits)} periods | "
              f"train={TRAIN_MONTHS}m test={TEST_MONTHS}m step={STEP_MONTHS}m")
        print(f"Pre-filter: xg_ratio_home_5 >= {XG_THRESHOLD}")
        print(f"Confidence: top {CONFIDENCE_PERCENTILE}% within pre-filtered set")
        print(f"Odds range: [{MIN_ODDS}, {MAX_ODDS}]")

    all_bets      = []
    period_stats  = []
    importances   = []

    for test_start, test_end in splits:
        train_all = df[df['date'] < test_start].copy()
        test_all  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        if len(train_all) < 300:
            continue

        # Pre-filter: only home bets with xg_ratio >= threshold
        xg_col = 'xg_ratio_home_5'
        train_xg = train_all[train_all[xg_col].fillna(0) >= XG_THRESHOLD].copy()
        test_xg  = test_all[test_all[xg_col].fillna(0) >= XG_THRESHOLD].copy()
        test_xg  = test_xg[
            test_xg['home_odds_val'].notna() &
            (test_xg['home_odds_val'] >= MIN_ODDS) &
            (test_xg['home_odds_val'] <= MAX_ODDS)
        ].copy()

        if len(train_xg) < 100 or len(test_xg) < 5:
            continue

        y_train = (train_xg['result'] == 'H').astype(int).values
        y_test  = (test_xg['result']  == 'H').astype(int).values

        # Split train → fit + calibration
        cal_size = max(40, int(len(train_xg) * 0.20))
        X_tr = train_xg[train_feat_cols].fillna(0).values[:-cal_size]
        y_tr = y_train[:-cal_size]
        X_cal = train_xg[train_feat_cols].fillna(0).values[-cal_size:]
        y_cal = y_train[-cal_size:]

        # Need both classes in cal set
        if len(np.unique(y_cal)) < 2:
            continue

        booster, cal_model, fi = fit_lgbm(X_tr, y_tr, X_cal, y_cal)
        importances.append(fi)

        X_test = test_xg[train_feat_cols].fillna(0).values
        probs  = predict_prob(booster, cal_model, X_test)

        # Confidence filter: take only top CONFIDENCE_PERCENTILE %
        threshold_prob = np.percentile(probs, 100 - CONFIDENCE_PERCENTILE)

        period_bets = []
        for i, (_, row) in enumerate(test_xg.iterrows()):
            p    = probs[i]
            odds = row['home_odds_val']
            ev   = p * odds - 1

            if p < threshold_prob:
                continue
            if ev < MIN_EV:
                continue

            won = (row['result'] == 'H')
            period_bets.append({
                'match_id': int(row['match_id']),
                'date':     row['date'],
                'prob':     round(float(p), 4),
                'odds':     round(float(odds), 2),
                'ev':       round(float(ev), 4),
                'xg_ratio': round(float(row[xg_col]), 2),
                'result':   'win' if won else 'loss',
            })

        n   = len(period_bets)
        nw  = sum(1 for b in period_bets if b['result'] == 'win')
        wr  = nw / n * 100 if n > 0 else 0

        pre_n = len(test_xg)
        period_stats.append({
            'test_start': test_start,
            'n_train': len(train_xg),
            'n_pre':   pre_n,
            'n_bets':  n,
            'win_rate': round(wr, 1),
        })
        all_bets.extend(period_bets)

        if verbose:
            print(f"  {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"train={len(train_xg):4d} | pre-filter={pre_n:3d} | "
                  f"selected={n:3d} | win={wr:.0f}%")

    # Compound bankroll simulation
    all_bets.sort(key=lambda b: b['date'])
    bankroll = INITIAL_BANKROLL
    peak     = INITIAL_BANKROLL
    max_dd   = 0.0
    series   = [bankroll]
    n_wins = n_losses = 0

    for bet in all_bets:
        stake = kelly_stake(bet['prob'], bet['odds'], bankroll,
                            KELLY_FRACTIONAL, MAX_STAKE_PCT)
        stake = round(stake, 2)
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
    avg_fi = np.mean(importances, axis=0) if importances else None

    return {
        'all_bets':      all_bets,
        'period_stats':  period_stats,
        'bankroll_series': series,
        'final_bankroll': round(bankroll, 2),
        'roi':           round((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100, 2),
        'win_rate':      round(n_wins / n_bets * 100, 1) if n_bets > 0 else 0,
        'max_drawdown':  round(max_dd, 1),
        'n_bets':        n_bets,
        'n_wins':        n_wins,
        'n_losses':      n_losses,
        'avg_odds':      round(np.mean([b['odds'] for b in all_bets]), 2) if all_bets else 0,
        'avg_ev':        round(np.mean([b['ev'] for b in all_bets]) * 100, 1) if all_bets else 0,
        'avg_fi':        avg_fi,
        'train_feat_cols': [c for c in feat_cols if c not in MARKET_COLS],
    }


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(result, feat_cols=None):
    r = result
    x = r['final_bankroll'] / INITIAL_BANKROLL

    print(f"\n{'='*55}")
    print(f"  BETA v2 — Selective xG Model")
    print(f"{'='*55}")
    print(f"  Bets:       {r['n_bets']} ({r['n_wins']}W / {r['n_losses']}L)")
    print(f"  Win rate:   {r['win_rate']:.1f}%")
    print(f"  Avg odds:   {r['avg_odds']:.2f}")
    print(f"  Avg EV:     {r['avg_ev']:.1f}%")
    print(f"  Bankroll:   ${INITIAL_BANKROLL:.0f} → ${r['final_bankroll']:.0f}")
    print(f"  ROI:        {r['roi']:+.1f}%")
    print(f"  Max DD:     {r['max_drawdown']:.1f}%")
    print(f"  x-factor:   {x:.2f}x")

    print(f"\n  Per period:")
    for p in r['period_stats']:
        print(f"    {p['test_start'].strftime('%Y-%m')} | "
              f"train={p['n_train']:4d} | pre-filter={p['n_pre']:3d} | "
              f"selected={p['n_bets']:3d} | win={p['win_rate']:5.1f}%")

    if r['avg_fi'] is not None and feat_cols:
        fc = result.get('train_feat_cols', feat_cols)
        imp = sorted(zip(fc, r['avg_fi']), key=lambda x: -x[1])
        print(f"\n  Top-15 features (gain importance):")
        for fname, fval in imp[:15]:
            print(f"    {fval:8.1f}  {fname}")

    # Bankroll milestones
    series = r['bankroll_series']
    milestones = [2000, 3000, 4000, 5000, 7500, 10000]
    reached = []
    for i, b in enumerate(series):
        for m in milestones:
            if b >= m and m not in [x[0] for x in reached]:
                reached.append((m, i))
    if reached:
        print(f"\n  Bankroll milestones:")
        for target, bet_idx in reached:
            print(f"    ${target:>6} reached after bet #{bet_idx}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 55)
    print("  BETA v2 — Loading data...")
    print("=" * 55)

    matches, stats, odds, injuries = load_all()
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)

    print(f"  {len(df)} matches | {len(feat_cols)} features")
    print(f"  Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    # Filter: need odds
    df = df[df['home_odds_val'].notna()].copy()
    print(f"  With odds: {len(df)}")

    result = run_backtest(df, feat_cols, verbose=True)
    print_report(result, feat_cols)

    print(f"\n  Total time: {time.time()-t0:.0f}s")
