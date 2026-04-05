"""
Cap sweep: scenario=3, EV=20%, 10 різних стартових точок.
Usage: python -m model.backtest_sweep
"""
import statistics
import model.backtest as bt_module
from model.train import load_data_from_db
from model.features.builder import build_dataset
from loguru import logger

TRAIN_WINDOW = 1500
N_OFFSETS = 10
OFFSET_STEP = 50


def main():
    logger.info("Loading data from DB...")
    matches, stats, odds, teams, injuries = load_data_from_db()
    logger.info("Building features...")
    dataset = build_dataset(matches, stats, odds, teams, injuries_df=injuries)
    logger.info(f"Dataset: {len(dataset)} rows")

    bt_module.MIN_SCENARIO_SCORE = 3
    bt_module.FRACTIONAL_KELLY = 0.25

    ev_values = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    cap_values = [0.04]

    print(f"\n{'EV':>6} | {'Cap':>5} | {'ROI mean':>10} | {'Win rate':>10} | {'Bets':>6} | {'Bankroll':>10} | {'STD':>8}")
    print("-" * 75)

    for ev in ev_values:
        for cap in cap_values:
            bt_module.MIN_EV = ev
            bt_module.MAX_STAKE_PCT = cap
            rois, win_rates, bets_list, bankrolls = [], [], [], []

            for i in range(N_OFFSETS):
                offset = i * OFFSET_STEP
                result = bt_module.backtest(
                    dataset.iloc[offset:].reset_index(drop=True),
                    train_window=TRAIN_WINDOW,
                )
                if result:
                    rois.append(result["roi"])
                    win_rates.append(result["win_rate"])
                    bets_list.append(result["total_bets"])
                    bankrolls.append(result["final_bankroll"])

            if not rois:
                continue

            print(
                f"{ev:>6.0%} | "
                f"{cap:>5.0%} | "
                f"{statistics.mean(rois):>+10.2%} | "
                f"{statistics.mean(win_rates):>10.2%} | "
                f"{statistics.mean(bets_list):>6.0f} | "
                f"${statistics.mean(bankrolls):>9.0f} | "
                f"{statistics.stdev(rois) if len(rois) > 1 else 0:>8.2%}"
            )
        print()

    print()


if __name__ == "__main__":
    main()
