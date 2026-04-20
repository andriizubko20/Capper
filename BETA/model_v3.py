"""
BETA/model_v3.py

V3 Combined Signal Model — walk-forward backtest.

Strategy:
  HOME bets [1.70–2.50]:
    - xg_ratio_home_5 >= XG_HOME_THRESHOLD
    - LightGBM binary → P(home_win)
    - Top CONFIDENCE_PERCENTILE% by model prob
    - EV > 0

  AWAY bets [1.70–2.50]:
    - elo_diff <= ELO_AWAY_THRESHOLD  (strong away team)
    - home_pts_5 <= HOME_POOR_FORM    (weak home team)
    - away_pts_5 >= AWAY_MIN_FORM
    - LightGBM binary → P(away_win)
    - Top CONFIDENCE_PERCENTILE% by model prob
    - EV > 0

Goal: 55%+ combined win rate, 150-200 bets/year, $4-5k from $1000.
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

# Home signal
XG_HOME_THRESHOLD    = 1.5    # xg_ratio_home_5 >= this
HOME_MIN_ODDS        = 1.70
HOME_MAX_ODDS        = 2.50

# Away signal
ELO_AWAY_THRESHOLD   = -50    # elo_diff <= this (away team stronger)
HOME_POOR_FORM       = 1.4    # home_pts_5 <= this
AWAY_MIN_FORM        = 1.6    # away_pts_5 >= this
AWAY_MIN_ODDS        = 1.70
AWAY_MAX_ODDS        = 2.50

# Shared
CONFIDENCE_PERCENTILE = 40    # top 40% by model confidence (within pre-filtered set)
MIN_EV               = 0.0    # EV >= 0
INITIAL_BANKROLL     = 1000.0
KELLY_FRACTIONAL     = 0.25
MAX_STAKE_PCT        = 0.06   # 6% cap (research showed 6-8% is optimal)
TRAIN_MONTHS         = 12
TEST_MONTHS          = 3
STEP_MONTHS          = 3

MARKET_COLS = {
    'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
    'home_odds_val','draw_odds_val','away_odds_val',
}


# ── LightGBM binary model ──────────────────────────────────────────────────────

def fit_lgbm(X_train, y_train, X_cal, y_cal, label=''):
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
    dtrain = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(
        params, dtrain,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    raw_cal = booster.predict(X_cal).reshape(-1, 1)
    cal = LogisticRegression(max_iter=300, C=1.0)
    cal.fit(raw_cal, y_cal)
    fi = booster.feature_importance(importance_type='gain')
    return booster, cal, fi


def predict_prob(booster, cal, X):
    raw = booster.predict(X).reshape(-1, 1)
    return cal.predict_proba(raw)[:, 1]


# ── Kelly sizing ────────────────────────────────────────────────────────────────

def kelly_stake(prob, odds, bankroll):
    b = odds - 1.0
    q = 1.0 - prob
    f = max(0.0, (prob * b - q) / b) * KELLY_FRACTIONAL
    return min(bankroll * f, bankroll * MAX_STAKE_PCT)


# ── Walk-forward backtest ───────────────────────────────────────────────────────

def run_backtest(df, feat_cols, verbose=True):
    df = df.sort_values('date').reset_index(drop=True)
    train_feat_cols = [c for c in feat_cols if c not in MARKET_COLS]

    min_date = df['date'].min()
    max_date = df['date'].max()

    splits = []
    ts = min_date + relativedelta(months=TRAIN_MONTHS)
    while ts + relativedelta(months=TEST_MONTHS) <= max_date + relativedelta(days=1):
        splits.append((ts, ts + relativedelta(months=TEST_MONTHS)))
        ts += relativedelta(months=STEP_MONTHS)

    if verbose:
        print(f"\nWalk-forward: {len(splits)} periods | "
              f"train={TRAIN_MONTHS}m test={TEST_MONTHS}m step={STEP_MONTHS}m")
        print(f"HOME: xg>={XG_HOME_THRESHOLD}, odds[{HOME_MIN_ODDS},{HOME_MAX_ODDS}]")
        print(f"AWAY: elo<={ELO_AWAY_THRESHOLD}, home_pts<={HOME_POOR_FORM}, "
              f"away_pts>={AWAY_MIN_FORM}, odds[{AWAY_MIN_ODDS},{AWAY_MAX_ODDS}]")
        print(f"Confidence top {CONFIDENCE_PERCENTILE}% | Kelly {KELLY_FRACTIONAL*100:.0f}% "
              f"cap {MAX_STAKE_PCT*100:.0f}%")

    all_bets     = []
    period_stats = []

    for test_start, test_end in splits:
        train_all = df[df['date'] < test_start].copy()
        test_all  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        if len(train_all) < 300:
            continue

        period_bets = []

        # ── HOME signal ────────────────────────────────────────────────────────
        xg_col = 'xg_ratio_home_5'
        train_home = train_all[train_all[xg_col].fillna(0) >= XG_HOME_THRESHOLD].copy()
        test_home  = test_all[
            (test_all[xg_col].fillna(0) >= XG_HOME_THRESHOLD) &
            test_all['home_odds_val'].notna() &
            (test_all['home_odds_val'] >= HOME_MIN_ODDS) &
            (test_all['home_odds_val'] <= HOME_MAX_ODDS)
        ].copy()

        if len(train_home) >= 100 and len(test_home) >= 3:
            y_train_h = (train_home['result'] == 'H').astype(int).values
            cal_size  = max(40, int(len(train_home) * 0.20))
            X_tr  = train_home[train_feat_cols].fillna(0).values[:-cal_size]
            y_tr  = y_train_h[:-cal_size]
            X_cal = train_home[train_feat_cols].fillna(0).values[-cal_size:]
            y_cal = y_train_h[-cal_size:]

            if len(np.unique(y_cal)) == 2:
                booster_h, cal_h, _ = fit_lgbm(X_tr, y_tr, X_cal, y_cal, 'home')
                X_test_h = test_home[train_feat_cols].fillna(0).values
                probs_h  = predict_prob(booster_h, cal_h, X_test_h)
                thr_h    = np.percentile(probs_h, 100 - CONFIDENCE_PERCENTILE)

                for i, (_, row) in enumerate(test_home.iterrows()):
                    p    = probs_h[i]
                    odds = row['home_odds_val']
                    ev   = p * odds - 1
                    if p < thr_h or ev < MIN_EV:
                        continue
                    won = (row['result'] == 'H')
                    period_bets.append({
                        'match_id': int(row['match_id']),
                        'date':     row['date'],
                        'side':     'home',
                        'prob':     round(float(p), 4),
                        'odds':     round(float(odds), 2),
                        'ev':       round(float(ev), 4),
                        'result':   'win' if won else 'loss',
                    })

        # ── AWAY signal ────────────────────────────────────────────────────────
        away_mask_train = (
            (train_all['elo_diff'].fillna(0) <= ELO_AWAY_THRESHOLD) &
            (train_all['home_pts_5'].fillna(3) <= HOME_POOR_FORM) &
            (train_all['away_pts_5'].fillna(0) >= AWAY_MIN_FORM)
        )
        away_mask_test = (
            (test_all['elo_diff'].fillna(0) <= ELO_AWAY_THRESHOLD) &
            (test_all['home_pts_5'].fillna(3) <= HOME_POOR_FORM) &
            (test_all['away_pts_5'].fillna(0) >= AWAY_MIN_FORM) &
            test_all['away_odds_val'].notna() &
            (test_all['away_odds_val'] >= AWAY_MIN_ODDS) &
            (test_all['away_odds_val'] <= AWAY_MAX_ODDS)
        )
        train_away = train_all[away_mask_train].copy()
        test_away  = test_all[away_mask_test].copy()

        if len(train_away) >= 80 and len(test_away) >= 3:
            y_train_a = (train_away['result'] == 'A').astype(int).values
            cal_size  = max(30, int(len(train_away) * 0.20))
            X_tr  = train_away[train_feat_cols].fillna(0).values[:-cal_size]
            y_tr  = y_train_a[:-cal_size]
            X_cal = train_away[train_feat_cols].fillna(0).values[-cal_size:]
            y_cal = y_train_a[-cal_size:]

            if len(np.unique(y_cal)) == 2:
                booster_a, cal_a, _ = fit_lgbm(X_tr, y_tr, X_cal, y_cal, 'away')
                X_test_a = test_away[train_feat_cols].fillna(0).values
                probs_a  = predict_prob(booster_a, cal_a, X_test_a)
                thr_a    = np.percentile(probs_a, 100 - CONFIDENCE_PERCENTILE)

                for i, (_, row) in enumerate(test_away.iterrows()):
                    p    = probs_a[i]
                    odds = row['away_odds_val']
                    ev   = p * odds - 1
                    if p < thr_a or ev < MIN_EV:
                        continue
                    # Skip if this match already has a home bet
                    mid = int(row['match_id'])
                    if any(b['match_id'] == mid for b in period_bets):
                        continue
                    won = (row['result'] == 'A')
                    period_bets.append({
                        'match_id': mid,
                        'date':     row['date'],
                        'side':     'away',
                        'prob':     round(float(p), 4),
                        'odds':     round(float(odds), 2),
                        'ev':       round(float(ev), 4),
                        'result':   'win' if won else 'loss',
                    })

        # Per-period stats
        n   = len(period_bets)
        nw  = sum(1 for b in period_bets if b['result'] == 'win')
        wr  = nw / n * 100 if n > 0 else 0
        nh  = sum(1 for b in period_bets if b['side'] == 'home')
        na  = sum(1 for b in period_bets if b['side'] == 'away')

        period_stats.append({
            'test_start': test_start,
            'n_train': len(train_all),
            'n_home':  nh,
            'n_away':  na,
            'n_bets':  n,
            'win_rate': round(wr, 1),
        })
        all_bets.extend(period_bets)

        if verbose:
            print(f"  {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"train={len(train_all):4d} | home={nh:3d} away={na:3d} | "
                  f"total={n:3d} | win={wr:.0f}%")

    # ── Bankroll simulation ────────────────────────────────────────────────────
    all_bets.sort(key=lambda b: b['date'])
    bankroll = INITIAL_BANKROLL
    peak     = INITIAL_BANKROLL
    max_dd   = 0.0
    series   = [bankroll]
    n_wins = n_losses = 0

    for bet in all_bets:
        stake = kelly_stake(bet['prob'], bet['odds'], bankroll)
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
    home_bets = [b for b in all_bets if b['side'] == 'home']
    away_bets = [b for b in all_bets if b['side'] == 'away']

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
        'n_home':          len(home_bets),
        'n_away':          len(away_bets),
        'wr_home':         round(sum(1 for b in home_bets if b['result']=='win')/len(home_bets)*100,1) if home_bets else 0,
        'wr_away':         round(sum(1 for b in away_bets if b['result']=='win')/len(away_bets)*100,1) if away_bets else 0,
        'avg_odds':        round(np.mean([b['odds'] for b in all_bets]), 2) if all_bets else 0,
        'avg_ev':          round(np.mean([b['ev'] for b in all_bets]) * 100, 1) if all_bets else 0,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result):
    r = result
    x = r['final_bankroll'] / INITIAL_BANKROLL

    print(f"\n{'='*60}")
    print(f"  BETA v3 — Combined Home + Away Model")
    print(f"{'='*60}")
    print(f"  Bets:       {r['n_bets']} ({r['n_wins']}W / {r['n_losses']}L)")
    print(f"  Win rate:   {r['win_rate']:.1f}%")
    print(f"  HOME:       {r['n_home']} bets | WR={r['wr_home']:.1f}%")
    print(f"  AWAY:       {r['n_away']} bets | WR={r['wr_away']:.1f}%")
    print(f"  Avg odds:   {r['avg_odds']:.2f}")
    print(f"  Avg EV:     {r['avg_ev']:.1f}%")
    print(f"  Bankroll:   ${INITIAL_BANKROLL:.0f} → ${r['final_bankroll']:.0f}")
    print(f"  ROI:        {r['roi']:+.1f}%")
    print(f"  Max DD:     {r['max_drawdown']:.1f}%")
    print(f"  x-factor:   {x:.2f}x")

    total_months = len(result['period_stats']) * STEP_MONTHS + TRAIN_MONTHS
    bets_per_year = r['n_bets'] / (total_months / 12) if total_months > 0 else 0
    print(f"  Bets/year:  {bets_per_year:.0f}")

    print(f"\n  Per period:")
    for p in result['period_stats']:
        print(f"    {p['test_start'].strftime('%Y-%m')} | "
              f"home={p['n_home']:3d} away={p['n_away']:3d} "
              f"total={p['n_bets']:3d} | win={p['win_rate']:5.1f}%")

    # Milestones
    series = r['bankroll_series']
    milestones = [1500, 2000, 3000, 5000, 7500, 10000]
    reached = []
    seen = set()
    for i, b in enumerate(series):
        for m in milestones:
            if b >= m and m not in seen:
                reached.append((m, i))
                seen.add(m)
    if reached:
        print(f"\n  Bankroll milestones:")
        for target, bet_idx in reached:
            print(f"    ${target:>6} reached after bet #{bet_idx}")

    # Extrapolation to 2 years (730 days)
    if r['n_bets'] > 0 and total_months > TRAIN_MONTHS:
        test_months_total = total_months - TRAIN_MONTHS
        daily_roi = (r['final_bankroll'] / INITIAL_BANKROLL) ** (1 / (test_months_total / 12)) - 1
        proj_2yr  = INITIAL_BANKROLL * (1 + daily_roi) ** 2
        print(f"\n  2-year projection (annualized): ${proj_2yr:.0f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("  BETA v3 — Loading data...")
    print("=" * 60)

    matches, stats, odds, injuries = load_all()
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)

    print(f"  {len(df)} matches | {len(feat_cols)} features")
    print(f"  Date range: {df['date'].min().date()} – {df['date'].max().date()}")

    # Need at least one side's odds
    df = df[df['home_odds_val'].notna() | df['away_odds_val'].notna()].copy()
    print(f"  With odds: {len(df)}")

    result = run_backtest(df, feat_cols, verbose=True)
    print_report(result)
    print(f"\n  Total time: {time.time()-t0:.0f}s")
