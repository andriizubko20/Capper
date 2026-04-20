"""
BETA/backtest/kelly.py

Kelly Criterion + compound bankroll simulation.
"""


def kelly_fraction(prob: float, odds: float, fractional: float = 0.25) -> float:
    """
    Full Kelly fraction × fractional multiplier.
    f = (p*b - q) / b   where b = odds - 1
    Returns 0 if negative (no bet).
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(0.0, f * fractional)


def simulate_compound(bets: list[dict],
                       initial: float = 1000.0,
                       kelly_cap: float = 0.04,
                       fractional: float = 0.25) -> dict:
    """
    Simulates compound bankroll growth over a list of bets.

    Each bet dict must have:
      - 'prob'   : our estimated probability
      - 'odds'   : decimal odds
      - 'result' : 'win' or 'loss'

    Returns dict with:
      final_bankroll, roi, win_rate, max_drawdown,
      n_bets, n_wins, n_losses,
      bankroll_series (list of bankroll after each bet)
    """
    bankroll = initial
    peak     = initial
    max_dd   = 0.0
    series   = [bankroll]
    n_wins = n_losses = 0

    for bet in bets:
        kf    = kelly_fraction(bet['prob'], bet['odds'], fractional)
        stake = min(bankroll * kf, bankroll * kelly_cap)
        stake = round(stake, 2)

        if bet['result'] == 'win':
            bankroll += stake * (bet['odds'] - 1)
            n_wins += 1
        else:
            bankroll -= stake
            n_losses += 1

        bankroll = max(bankroll, 0.01)   # floor to avoid negative
        peak     = max(peak, bankroll)
        dd       = (peak - bankroll) / peak * 100
        max_dd   = max(max_dd, dd)
        series.append(round(bankroll, 2))

    n_bets = n_wins + n_losses
    return {
        'final_bankroll':  round(bankroll, 2),
        'roi':             round((bankroll - initial) / initial * 100, 2),
        'win_rate':        round(n_wins / n_bets * 100, 1) if n_bets > 0 else 0.0,
        'max_drawdown':    round(max_dd, 1),
        'n_bets':          n_bets,
        'n_wins':          n_wins,
        'n_losses':        n_losses,
        'bankroll_series': series,
        'initial':         initial,
    }
