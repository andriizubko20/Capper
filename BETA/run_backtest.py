"""
BETA/run_backtest.py

Entry point — runs walk-forward backtest for all 4 models and prints report.

Usage:
  docker exec capper_scheduler python -m BETA.run_backtest
  docker exec capper_scheduler python -m BETA.run_backtest --model m1
  docker exec capper_scheduler python -m BETA.run_backtest --model m3
"""
import sys
import os
import argparse
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from BETA.data.extract import load_all
from BETA.data.features import build_feature_matrix, get_feature_cols
from BETA.backtest.engine import run_walkforward, print_report
from BETA.backtest.kelly import simulate_compound


# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_BANKROLL = 1000.0
KELLY_CAP        = 0.04
FRACTIONAL       = 0.25
MIN_EV           = 0.05
MIN_ODDS         = 1.5
MAX_ODDS         = 4.0
TRAIN_MONTHS     = 12
TEST_MONTHS      = 3
STEP_MONTHS      = 3


# ── Custom walkforward for M3/M4 (need home_ids, away_ids) ───────────────────

def run_walkforward_with_ids(df, feature_cols, model, initial_bankroll,
                              kelly_cap, fractional, min_ev,
                              min_odds, max_odds, train_months, test_months,
                              step_months, use_agreement=False, verbose=True):
    """
    Extended walkforward that passes home_ids/away_ids to predict_proba.
    Used for M3 (Poisson) and M4 (Ensemble).
    """
    from dateutil.relativedelta import relativedelta
    from BETA.backtest.kelly import kelly_fraction as kf_fn

    df = df.dropna(subset=['result']).copy().sort_values('date').reset_index(drop=True)
    min_date = df['date'].min()
    max_date = df['date'].max()

    test_start_dt = min_date + relativedelta(months=train_months)
    splits = []
    while test_start_dt + relativedelta(months=test_months) <= max_date + relativedelta(days=1):
        splits.append((test_start_dt, test_start_dt + relativedelta(months=test_months)))
        test_start_dt += relativedelta(months=step_months)

    label_map = {'H': 0, 'D': 1, 'A': 2}
    all_bets = []
    period_results = []
    importances = []

    for test_start, test_end in splits:
        train_df = df[df['date'] < test_start].copy()
        test_df  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        if len(train_df) < 200 or len(test_df) < 10:
            continue

        X_train = train_df[feature_cols].fillna(0).values
        X_test  = test_df[feature_cols].fillna(0).values
        y_train = train_df['result'].map(label_map).values

        home_ids_train = train_df['home_team_id'].values
        away_ids_train = train_df['away_team_id'].values
        home_ids_test  = test_df['home_team_id'].values
        away_ids_test  = test_df['away_team_id'].values

        # Fit with train matches
        train_matches = train_df[['match_id','date','home_team_id','away_team_id',
                                   'home_score','away_score','result','league_id']].copy()
        model.fit(X_train, y_train, matches_df=train_matches)

        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importances.append(model.feature_importances_)

        h_odds_arr = test_df['home_odds_val'].values if 'home_odds_val' in test_df else np.full(len(test_df), np.nan)
        a_odds_arr = test_df['away_odds_val'].values if 'away_odds_val' in test_df else np.full(len(test_df), np.nan)

        if use_agreement:
            proba, agreement = model.predict_proba_with_agreement(
                X_test, home_ids=home_ids_test, away_ids=away_ids_test,
                odds_home=h_odds_arr, odds_away=a_odds_arr, min_ev=min_ev,
            )
        else:
            proba = model.predict_proba(
                X_test, home_ids=home_ids_test, away_ids=away_ids_test
            )
            agreement = None

        period_bets = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            p_home = proba[i, 0]
            p_away = proba[i, 2]
            h_odds = h_odds_arr[i]
            a_odds = a_odds_arr[i]

            # Home
            if not np.isnan(h_odds) and min_odds <= h_odds <= max_odds:
                ev_h = p_home * h_odds - 1
                agrees = agreement[i, 0] if agreement is not None else True
                if ev_h >= min_ev and p_home > 0.30 and agrees:
                    period_bets.append({
                        'match_id': int(row.match_id), 'date': row.date,
                        'side': 'home', 'prob': p_home, 'odds': h_odds,
                        'ev': round(ev_h, 4),
                        'result': 'win' if row.result == 'H' else 'loss',
                    })

            # Away
            if not np.isnan(a_odds) and min_odds <= a_odds <= max_odds:
                ev_a = p_away * a_odds - 1
                agrees = agreement[i, 2] if agreement is not None else True
                if ev_a >= min_ev and p_away > 0.25 and agrees:
                    period_bets.append({
                        'match_id': int(row.match_id), 'date': row.date,
                        'side': 'away', 'prob': p_away, 'odds': a_odds,
                        'ev': round(ev_a, 4),
                        'result': 'win' if row.result == 'A' else 'loss',
                    })

        n_bets = len(period_bets)
        n_wins = sum(1 for b in period_bets if b['result'] == 'win')
        period_results.append({
            'test_start': test_start, 'test_end': test_end,
            'n_train': len(train_df), 'n_test': len(test_df),
            'n_bets': n_bets,
            'win_rate': round(n_wins / n_bets * 100, 1) if n_bets > 0 else 0,
        })
        all_bets.extend(period_bets)

        if verbose:
            print(f"  {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"train={len(train_df)} bets={n_bets} win={period_results[-1]['win_rate']:.0f}%")

    all_bets.sort(key=lambda b: b['date'])
    compound = simulate_compound(all_bets, initial_bankroll, kelly_cap, fractional)
    avg_imp = np.mean(importances, axis=0) if importances else None
    return {'all_bets': all_bets, 'period_results': period_results,
            'compound': compound, 'feature_importances': avg_imp}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', choices=['all','m1','m2','m3','m4'])
    parser.add_argument('--min-ev', type=float, default=MIN_EV)
    parser.add_argument('--kelly-cap', type=float, default=KELLY_CAP)
    args = parser.parse_args()

    print("=" * 60)
    print("BETA — Walk-forward backtest")
    print("=" * 60)

    # ── Load data ──
    t0 = time.time()
    matches, stats, odds, injuries = load_all()

    print("\nBuilding feature matrix...")
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)
    print(f"  {len(feat_cols)} features | {len(df)} matches")
    print(f"  Matches with odds: {df['home_odds_val'].notna().sum()}")
    print(f"  Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    print(f"  Data load: {time.time()-t0:.1f}s")

    # Drop matches with no odds (can't bet)
    df_bet = df[df['home_odds_val'].notna() & df['away_odds_val'].notna()].copy()
    print(f"  Rows with odds (for betting): {len(df_bet)}")

    results = {}

    # ── M1 LightGBM ──
    if args.model in ('all', 'm1'):
        print("\n[M1] LightGBM + Platt calibration")
        from BETA.models.m1_lgbm import M1LightGBM
        t = time.time()
        r = run_walkforward(
            df_bet, feat_cols, M1LightGBM,
            initial_bankroll=INITIAL_BANKROLL,
            kelly_cap=args.kelly_cap,
            fractional=FRACTIONAL,
            min_ev=args.min_ev,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS, step_months=STEP_MONTHS,
            verbose=True,
        )
        results['M1 LightGBM'] = r
        print(f"  [{time.time()-t:.1f}s]")
        print_report("M1 LightGBM", r, r.get('train_feat_cols', feat_cols))

    # ── M2 XGBoost ──
    if args.model in ('all', 'm2'):
        print("\n[M2] XGBoost + Isotonic calibration")
        from BETA.models.m2_xgb import M2XGBoost
        t = time.time()
        r = run_walkforward(
            df_bet, feat_cols, M2XGBoost,
            initial_bankroll=INITIAL_BANKROLL,
            kelly_cap=args.kelly_cap,
            fractional=FRACTIONAL,
            min_ev=args.min_ev,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS, step_months=STEP_MONTHS,
            verbose=True,
        )
        results['M2 XGBoost'] = r
        print(f"  [{time.time()-t:.1f}s]")
        print_report("M2 XGBoost", r, r.get('train_feat_cols', feat_cols))

    # ── M3 Poisson ──
    if args.model in ('all', 'm3'):
        print("\n[M3] Dixon-Coles Poisson")
        from BETA.models.m3_poisson import M3Poisson
        t = time.time()
        r = run_walkforward_with_ids(
            df_bet, feat_cols, M3Poisson(),
            initial_bankroll=INITIAL_BANKROLL,
            kelly_cap=args.kelly_cap,
            fractional=FRACTIONAL,
            min_ev=args.min_ev,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS, step_months=STEP_MONTHS,
            verbose=True,
        )
        results['M3 Poisson'] = r
        print(f"  [{time.time()-t:.1f}s]")
        print_report("M3 Poisson", r)

    # ── M4 Ensemble ──
    if args.model in ('all', 'm4'):
        print("\n[M4] Ensemble (M1+M2+M3)")
        from BETA.models.m1_lgbm import M1LightGBM
        from BETA.models.m2_xgb  import M2XGBoost
        from BETA.models.m3_poisson import M3Poisson
        from BETA.models.m4_ensemble import M4Ensemble
        t = time.time()
        ensemble = M4Ensemble(M1LightGBM(), M2XGBoost(), M3Poisson())
        r = run_walkforward_with_ids(
            df_bet, feat_cols, ensemble,
            initial_bankroll=INITIAL_BANKROLL,
            kelly_cap=args.kelly_cap,
            fractional=FRACTIONAL,
            min_ev=args.min_ev,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS,
            train_months=TRAIN_MONTHS, test_months=TEST_MONTHS, step_months=STEP_MONTHS,
            use_agreement=True,  # bet only when 2/3 agree
            verbose=True,
        )
        results['M4 Ensemble'] = r
        print(f"  [{time.time()-t:.1f}s]")
        print_report("M4 Ensemble", r, feat_cols)

    # ── Summary table ──
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Bets':>6} {'Win%':>6} {'ROI':>8} {'x-factor':>9} {'MaxDD':>7}")
        print("-" * 60)
        for name, r in results.items():
            c = r['compound']
            x = c['final_bankroll'] / c['initial']
            print(f"{name:<20} {c['n_bets']:>6} {c['win_rate']:>5.1f}% "
                  f"{c['roi']:>+7.1f}% {x:>8.2f}x {c['max_drawdown']:>6.1f}%")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
