import pandas as pd
from loguru import logger


def compute_clv(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Розраховує Closing Line Value для кожного пику.
    CLV = (наша_ймовірність / closing_implied_prob) - 1

    predictions — DataFrame з колонками:
        probability, closing_odds, outcome
    """
    predictions = predictions.copy()
    predictions["closing_implied"] = 1 / predictions["closing_odds"]
    predictions["clv"] = predictions["probability"] / predictions["closing_implied"] - 1
    return predictions


def compute_roi(results: pd.DataFrame) -> dict:
    """
    ROI симуляція по результатам ставок.

    results — DataFrame з колонками:
        stake, odds, outcome (pick), actual_outcome
    """
    results = results.copy()
    results["profit"] = results.apply(
        lambda r: r["stake"] * (r["odds"] - 1) if r["outcome"] == r["actual_outcome"] else -r["stake"],
        axis=1,
    )
    total_staked = results["stake"].sum()
    total_profit = results["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0

    return {
        "total_bets": len(results),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "win_rate": round((results["outcome"] == results["actual_outcome"]).mean(), 4),
        "avg_clv": round(results["clv"].mean(), 4) if "clv" in results.columns else None,
    }


def log_metrics(metrics: dict) -> None:
    logger.info("=== Model Evaluation ===")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
