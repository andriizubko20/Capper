"""
EV threshold sweep: runs N backtests per threshold and reports mean ± std.
Usage: python -m model.backtest_sweep
"""
import statistics
import model.backtest as bt_module
from model.train import load_data_from_db
from model.features.builder import build_dataset
from loguru import logger

EV_VALUES = [0.18, 0.19, 0.20, 0.21, 0.22]
N_RUNS = 10
TRAIN_WINDOW = 1500


def main():
    logger.info("Loading data from DB...")
    matches, stats, odds, teams, injuries = load_data_from_db()
    logger.info("Building features...")
    dataset = build_dataset(matches, stats, odds, teams, injuries_df=injuries)
    logger.info(f"Dataset: {len(dataset)} rows")

    print(f"\n{'EV':>6} | {'ROI mean':>10} | {'ROI std':>9} | {'Bets mean':>10} | {'Bankroll mean':>14}")
    print("-" * 60)

    for ev in EV_VALUES:
        bt_module.MIN_EV = ev
        rois, bets_list, bankrolls = [], [], []

        for i in range(N_RUNS):
            result = bt_module.backtest(dataset, train_window=TRAIN_WINDOW)
            if result:
                rois.append(result["roi"])
                bets_list.append(result["total_bets"])
                bankrolls.append(result["final_bankroll"])

        if not rois:
            print(f"{ev:>6.0%} | {'no bets':>10}")
            continue

        roi_mean = statistics.mean(rois)
        roi_std = statistics.stdev(rois) if len(rois) > 1 else 0.0
        bets_mean = statistics.mean(bets_list)
        bankroll_mean = statistics.mean(bankrolls)

        print(
            f"{ev:>6.0%} | "
            f"{roi_mean:>+10.2%} | "
            f"{roi_std:>9.2%} | "
            f"{bets_mean:>10.0f} | "
            f"${bankroll_mean:>13.0f}"
        )

    print()


if __name__ == "__main__":
    main()
