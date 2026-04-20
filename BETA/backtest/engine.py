"""
BETA/backtest/engine.py

Walk-forward backtest engine.

Strategy:
  - Train on [start, test_start)
  - Test  on [test_start, test_start + test_months)
  - Slide by step_months
  - Repeat until end of data

EV filter + Kelly sizing applied during test.
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from BETA.backtest.kelly import simulate_compound, kelly_fraction


def _date_splits(min_date: pd.Timestamp, max_date: pd.Timestamp,
                 train_months: int, test_months: int,
                 step_months: int) -> list[tuple]:
    """Returns list of (train_end, test_start, test_end) tuples."""
    splits = []
    test_start = min_date + relativedelta(months=train_months)
    while test_start + relativedelta(months=test_months) <= max_date + relativedelta(days=1):
        test_end = test_start + relativedelta(months=test_months)
        splits.append((test_start, test_end))
        test_start += relativedelta(months=step_months)
    return splits


def run_walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_cls,                  # class with fit(X, y) and predict_proba(X)
    model_kwargs: dict = None,
    initial_bankroll: float = 1000.0,
    kelly_cap: float = 0.04,
    fractional: float = 0.25,
    min_ev: float = 0.05,
    min_edge: float = 0.05,     # our_prob - market_prob must exceed this
    min_odds: float = 1.5,
    max_odds: float = 4.0,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    min_bets_per_period: int = 10,
    bet_on: str = 'both',       # 'home', 'away', or 'both'
    verbose: bool = True,
) -> dict:
    """
    Runs walk-forward backtest.

    Returns dict with:
      - all_bets: list of all bet dicts (prob, odds, result, date, match_id, side)
      - period_results: list of per-period summary dicts
      - compound: result of simulate_compound over all_bets
      - feature_importances: averaged across folds (if model supports it)
    """
    model_kwargs = model_kwargs or {}
    df = df.dropna(subset=['result']).copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Remove market features from model training — model must find independent signal.
    # Market odds/probs are used only for edge filter AFTER prediction.
    MARKET_COLS = {'mkt_home_prob','mkt_draw_prob','mkt_away_prob',
                   'home_odds_val','draw_odds_val','away_odds_val'}
    train_feat_cols = [c for c in feature_cols if c not in MARKET_COLS]

    min_date = df['date'].min()
    max_date = df['date'].max()
    splits   = _date_splits(min_date, max_date, train_months, test_months, step_months)

    if verbose:
        print(f"  Walk-forward: {len(splits)} periods | "
              f"train={train_months}m test={test_months}m step={step_months}m")

    all_bets       = []
    period_results = []
    importances    = []

    for test_start, test_end in splits:
        train_df = df[df['date'] < test_start].copy()
        test_df  = df[(df['date'] >= test_start) & (df['date'] < test_end)].copy()

        if len(train_df) < 200 or len(test_df) < min_bets_per_period:
            continue

        # Prepare features + labels (market cols excluded from training)
        X_train = train_df[train_feat_cols].fillna(0).values
        X_test  = test_df[train_feat_cols].fillna(0).values

        # Multi-class label: H=0, D=1, A=2
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_train = train_df['result'].map(label_map).values

        # Fit
        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        # Predict proba → columns [P(H), P(D), P(A)]
        proba = model.predict_proba(X_test)   # shape (n, 3)

        # Feature importances
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importances.append(model.feature_importances_)

        # Generate bets
        period_bets = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            p_home = proba[i, 0]
            p_away = proba[i, 2]

            h_odds = row.get('home_odds_val', np.nan)
            a_odds = row.get('away_odds_val', np.nan)
            mkt_h  = row.get('mkt_home_prob', np.nan)
            mkt_a  = row.get('mkt_away_prob', np.nan)

            # Home bet — only when our model significantly exceeds market
            if bet_on in ('home', 'both') and not np.isnan(h_odds):
                ev_home  = p_home * h_odds - 1
                edge_h   = p_home - (mkt_h if not np.isnan(mkt_h) else 0)
                if (ev_home >= min_ev
                        and edge_h >= min_edge
                        and min_odds <= h_odds <= max_odds):
                    period_bets.append({
                        'match_id': int(row.match_id),
                        'date':     row.date,
                        'side':     'home',
                        'prob':     p_home,
                        'odds':     h_odds,
                        'ev':       round(ev_home, 4),
                        'edge':     round(edge_h, 4),
                        'result':   'win' if row.result == 'H' else 'loss',
                    })

            # Away bet
            if bet_on in ('away', 'both') and not np.isnan(a_odds):
                ev_away  = p_away * a_odds - 1
                edge_a   = p_away - (mkt_a if not np.isnan(mkt_a) else 0)
                if (ev_away >= min_ev
                        and edge_a >= min_edge
                        and min_odds <= a_odds <= max_odds):
                    period_bets.append({
                        'match_id': int(row.match_id),
                        'date':     row.date,
                        'side':     'away',
                        'prob':     p_away,
                        'odds':     a_odds,
                        'ev':       round(ev_away, 4),
                        'edge':     round(edge_a, 4),
                        'result':   'win' if row.result == 'A' else 'loss',
                    })

        n_bets = len(period_bets)
        n_wins = sum(1 for b in period_bets if b['result'] == 'win')
        win_rate = n_wins / n_bets * 100 if n_bets > 0 else 0

        period_results.append({
            'test_start': test_start,
            'test_end':   test_end,
            'n_train':    len(train_df),
            'n_test':     len(test_df),
            'n_bets':     n_bets,
            'win_rate':   round(win_rate, 1),
        })

        all_bets.extend(period_bets)

        if verbose:
            print(f"  {test_start.strftime('%Y-%m')}–{test_end.strftime('%Y-%m')}: "
                  f"train={len(train_df)} test={len(test_df)} "
                  f"bets={n_bets} win={win_rate:.0f}%")

    # Compound simulation over all bets in chronological order
    all_bets.sort(key=lambda b: b['date'])
    compound = simulate_compound(all_bets, initial_bankroll, kelly_cap, fractional)

    # Average feature importances
    avg_importance = None
    if importances:
        avg_importance = np.mean(importances, axis=0)

    return {
        'all_bets':            all_bets,
        'period_results':      period_results,
        'compound':            compound,
        'feature_importances': avg_importance,
        'train_feat_cols':     train_feat_cols,
    }


def print_report(name: str, result: dict, feature_cols: list[str] = None) -> None:
    c = result['compound']
    bets = result['all_bets']
    avg_odds = np.mean([b['odds'] for b in bets]) if bets else 0
    avg_ev   = np.mean([b['ev']   for b in bets]) if bets else 0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Bets:        {c['n_bets']} ({c['n_wins']}W / {c['n_losses']}L)")
    print(f"  Win rate:    {c['win_rate']:.1f}%")
    print(f"  Avg odds:    {avg_odds:.2f}")
    print(f"  Avg EV:      {avg_ev*100:.1f}%")
    print(f"  Bankroll:    ${c['initial']:.0f} → ${c['final_bankroll']:.0f}")
    print(f"  ROI:         {c['roi']:+.1f}%")
    print(f"  Max DD:      {c['max_drawdown']:.1f}%")
    print(f"  x-factor:    {c['final_bankroll']/c['initial']:.2f}x")

    # Per period
    print(f"\n  Per period:")
    for p in result['period_results']:
        print(f"    {p['test_start'].strftime('%Y-%m')} | "
              f"bets={p['n_bets']:>3} | win={p['win_rate']:>5.1f}%")

    # Top features
    if result['feature_importances'] is not None and feature_cols:
        imp = sorted(zip(feature_cols, result['feature_importances']),
                     key=lambda x: -x[1])
        print(f"\n  Top-10 features:")
        for fname, fval in imp[:10]:
            print(f"    {fval:.4f}  {fname}")
