"""Rebuild the frozen legacy/execution-aware timing comparison in memory."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_asset_allocation import phase0_legacy
from multi_asset_allocation.execution_backtest import (
    build_oos_return_frame,
    build_timing_comparison_table,
    compare_adjusted_close_snapshots,
    run_strategy_backtests,
)
from multi_asset_allocation.execution_returns import (
    build_adjusted_execution_open_matrix,
    calculate_execution_open_returns,
)
from multi_asset_allocation.execution_timing import build_execution_schedule


TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]
ANALYSIS_END = "2026-06-30"
OOS_START = "2019-01-01"
TREND_LOOKBACK = 12
VOLATILITY_WINDOW = 63
TRANSACTION_COST = 0.001
CASH_RETURN = 0.0
PHASE0_SNAPSHOT = PROJECT_ROOT / "data/phase0/phase0_adjusted_close_snapshot.csv"
EXECUTION_SNAPSHOT = PROJECT_ROOT / "data/execution/execution_market_snapshot.csv"
OUTPUT_PATH = PROJECT_ROOT / "results/phase1_execution_timing_comparison.csv"


def build_primary_targets(phase0_prices):
    """Build the fixed Phase 0 target/cash matrices using only legacy helpers."""
    analysis_prices = phase0_legacy.prepare_analysis_prices(
        phase0_prices,
        TICKERS,
        ANALYSIS_END,
    )
    daily_returns = phase0_legacy.calculate_daily_returns(analysis_prices)
    monthly_prices = phase0_legacy.construct_monthly_prices(analysis_prices)
    monthly_returns = phase0_legacy.calculate_monthly_returns(monthly_prices)
    trend_signal = phase0_legacy.calculate_trend_signal(
        monthly_prices,
        TREND_LOOKBACK,
    )
    monthly_volatility = phase0_legacy.calculate_monthly_realized_volatility(
        daily_returns,
        VOLATILITY_WINDOW,
    )
    equal_weight = phase0_legacy.equal_weight_targets(
        monthly_prices.index,
        TICKERS,
    )
    inverse_volatility = phase0_legacy.inverse_volatility_targets(
        monthly_volatility,
        len(TICKERS),
    )
    trend_equal_weight = phase0_legacy.trend_equal_weight_targets(
        trend_signal,
        len(TICKERS),
    )
    trend_inverse_volatility = phase0_legacy.trend_inverse_volatility_targets(
        trend_signal,
        inverse_volatility,
    )
    targets = {
        "Equal Weight": equal_weight,
        "Inverse Volatility": inverse_volatility,
        "Trend + Equal Weight": trend_equal_weight,
        "Trend + Inverse Volatility": trend_inverse_volatility,
    }
    cash = {
        "Equal Weight": phase0_legacy.zero_cash_weights(equal_weight.index),
        "Inverse Volatility": phase0_legacy.zero_cash_weights(
            inverse_volatility.index
        ),
        "Trend + Equal Weight": phase0_legacy.cash_weights(trend_equal_weight),
        "Trend + Inverse Volatility": phase0_legacy.cash_weights(
            trend_inverse_volatility
        ),
    }
    common_index, common_start, common_end = phase0_legacy.common_formation_index(
        targets,
        cash,
        monthly_prices,
    )
    for name in targets:
        targets[name] = targets[name].reindex(common_index)
        cash[name] = cash[name].reindex(common_index)
    return {
        "analysis_prices": analysis_prices,
        "monthly_prices": monthly_prices,
        "monthly_returns": monthly_returns,
        "targets": targets,
        "cash": cash,
        "common_index": common_index,
        "common_start": common_start,
        "common_end": common_end,
    }


def build_execution_aware_returns(monthly_prices, common_index, market_data):
    """Build unshifted adjusted execution-open returns on the common index."""
    spy_dates = market_data.loc[market_data["Ticker"] == "SPY", "Date"]
    schedule = build_execution_schedule(spy_dates, monthly_prices.index)
    adjusted_execution_opens = build_adjusted_execution_open_matrix(
        schedule,
        market_data,
        TICKERS,
    )
    execution_returns = calculate_execution_open_returns(adjusted_execution_opens)
    return schedule, adjusted_execution_opens, execution_returns.reindex(common_index)


def run_comparison():
    """Return the deterministic comparison table and supporting diagnostics."""
    phase0_prices = pd.read_csv(PHASE0_SNAPSHOT, index_col="Date", parse_dates=True)
    market_data = pd.read_csv(EXECUTION_SNAPSHOT, parse_dates=["Date"])
    primary = build_primary_targets(phase0_prices)
    schedule, adjusted_execution_opens, execution_returns = (
        build_execution_aware_returns(
            primary["monthly_prices"],
            primary["common_index"],
            market_data,
        )
    )

    legacy_gross = run_strategy_backtests(
        primary["targets"],
        primary["cash"],
        primary["monthly_returns"],
        transaction_cost=0.0,
        cash_return=CASH_RETURN,
    )
    execution_gross = run_strategy_backtests(
        primary["targets"],
        primary["cash"],
        execution_returns,
        transaction_cost=0.0,
        cash_return=CASH_RETURN,
    )
    legacy_net = run_strategy_backtests(
        primary["targets"],
        primary["cash"],
        primary["monthly_returns"],
        transaction_cost=TRANSACTION_COST,
        cash_return=CASH_RETURN,
    )
    execution_net = run_strategy_backtests(
        primary["targets"],
        primary["cash"],
        execution_returns,
        transaction_cost=TRANSACTION_COST,
        cash_return=CASH_RETURN,
    )
    table = build_timing_comparison_table(
        legacy_gross,
        execution_gross,
        legacy_net,
        execution_net,
        OOS_START,
    )
    diagnostic = compare_adjusted_close_snapshots(
        primary["analysis_prices"],
        market_data,
        TICKERS,
    )
    return {
        "comparison": table,
        "diagnostic": diagnostic,
        "primary": primary,
        "schedule": schedule,
        "adjusted_execution_opens": adjusted_execution_opens,
        "execution_returns": execution_returns,
        "legacy_gross": legacy_gross,
        "execution_gross": execution_gross,
        "legacy_net": legacy_net,
        "execution_net": execution_net,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare frozen legacy and execution-aware timing results."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing comparison CSV",
    )
    args = parser.parse_args()
    if OUTPUT_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists; rerun with --overwrite to replace it."
        )

    output = run_comparison()
    output["comparison"].to_csv(OUTPUT_PATH, index=False, lineterminator="\n")
    diagnostic = output["diagnostic"]
    print("=== CROSS-SNAPSHOT ADJUSTED-CLOSE DIAGNOSTIC ===")
    print("Max absolute difference:", diagnostic["max_absolute_difference"])
    print(
        "Max absolute location:",
        diagnostic["max_absolute_date"],
        diagnostic["max_absolute_ticker"],
    )
    print("Max relative difference:", diagnostic["max_relative_difference"])
    print(
        "Max relative location:",
        diagnostic["max_relative_date"],
        diagnostic["max_relative_ticker"],
    )
    print(
        "OOS observations:",
        len(build_oos_return_frame(output["legacy_net"], OOS_START)),
    )
    print("=== EXECUTION TIMING COMPARISON ===")
    print(output["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
