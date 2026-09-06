"""Research-only comparison of legacy and execution-aware holding returns."""

import numpy as np
import pandas as pd

from multi_asset_allocation import phase0_legacy


def run_strategy_backtests(
    portfolio_targets,
    portfolio_cash,
    asset_returns,
    transaction_cost,
    cash_return,
):
    """Run the frozen Phase 0 backtest engine for every supplied strategy."""
    return {
        name: phase0_legacy.run_monthly_backtest(
            target_weights=portfolio_targets[name],
            cash_weights=portfolio_cash[name],
            monthly_asset_returns=asset_returns,
            transaction_cost=transaction_cost,
            cash_return=cash_return,
        )
        for name in portfolio_targets
    }


def build_oos_return_frame(backtests, oos_start):
    """Collect net backtest returns using the legacy inclusive OOS slice."""
    return pd.DataFrame(
        {name: result["net_return"] for name, result in backtests.items()}
    ).loc[oos_start:].copy()


def build_timing_comparison_table(
    legacy_gross_backtests,
    execution_gross_backtests,
    legacy_net_backtests,
    execution_net_backtests,
    oos_start,
):
    """Build a deterministic OOS timing-comparison table.

    Every delta is execution-aware minus legacy. Turnover summaries retain the
    legacy engine's full-sample convention.
    """
    legacy_gross = build_oos_return_frame(legacy_gross_backtests, oos_start)
    execution_gross = build_oos_return_frame(
        execution_gross_backtests,
        oos_start,
    )
    legacy_net = build_oos_return_frame(legacy_net_backtests, oos_start)
    execution_net = build_oos_return_frame(execution_net_backtests, oos_start)

    if not legacy_net.columns.equals(execution_net.columns):
        raise ValueError("Legacy and execution strategy names do not match.")
    if not legacy_net.index.equals(execution_net.index):
        raise ValueError("Legacy and execution OOS labels do not match.")

    legacy_gross_metrics = phase0_legacy.build_period_performance_table(
        legacy_gross
    )
    execution_gross_metrics = phase0_legacy.build_period_performance_table(
        execution_gross
    )
    legacy_net_metrics = phase0_legacy.build_period_performance_table(legacy_net)
    execution_net_metrics = phase0_legacy.build_period_performance_table(
        execution_net
    )
    legacy_turnover = phase0_legacy.summarize_turnover(legacy_net_backtests)
    execution_turnover = phase0_legacy.summarize_turnover(
        execution_net_backtests
    )

    rows = []
    for strategy in legacy_net.columns:
        paired_returns = pd.concat(
            [legacy_net[strategy], execution_net[strategy]],
            axis=1,
            keys=["legacy", "execution"],
        ).dropna()
        return_gaps = (paired_returns["execution"] - paired_returns["legacy"]).abs()

        rows.append(
            {
                "Strategy": strategy,
                "Legacy Gross CAGR": legacy_gross_metrics.loc[strategy, "CAGR"],
                "Execution Gross CAGR": execution_gross_metrics.loc[
                    strategy,
                    "CAGR",
                ],
                "Gross CAGR Delta": (
                    execution_gross_metrics.loc[strategy, "CAGR"]
                    - legacy_gross_metrics.loc[strategy, "CAGR"]
                ),
                "Legacy Gross Sharpe": legacy_gross_metrics.loc[
                    strategy,
                    "Sharpe Ratio",
                ],
                "Execution Gross Sharpe": execution_gross_metrics.loc[
                    strategy,
                    "Sharpe Ratio",
                ],
                "Gross Sharpe Delta": (
                    execution_gross_metrics.loc[strategy, "Sharpe Ratio"]
                    - legacy_gross_metrics.loc[strategy, "Sharpe Ratio"]
                ),
                "Legacy Net CAGR": legacy_net_metrics.loc[strategy, "CAGR"],
                "Execution Net CAGR": execution_net_metrics.loc[
                    strategy,
                    "CAGR",
                ],
                "Net CAGR Delta": (
                    execution_net_metrics.loc[strategy, "CAGR"]
                    - legacy_net_metrics.loc[strategy, "CAGR"]
                ),
                "Legacy Net Annualized Volatility": legacy_net_metrics.loc[
                    strategy,
                    "Annualized Volatility",
                ],
                "Execution Net Annualized Volatility": execution_net_metrics.loc[
                    strategy,
                    "Annualized Volatility",
                ],
                "Net Volatility Delta": (
                    execution_net_metrics.loc[strategy, "Annualized Volatility"]
                    - legacy_net_metrics.loc[strategy, "Annualized Volatility"]
                ),
                "Legacy Net Sharpe": legacy_net_metrics.loc[
                    strategy,
                    "Sharpe Ratio",
                ],
                "Execution Net Sharpe": execution_net_metrics.loc[
                    strategy,
                    "Sharpe Ratio",
                ],
                "Net Sharpe Delta": (
                    execution_net_metrics.loc[strategy, "Sharpe Ratio"]
                    - legacy_net_metrics.loc[strategy, "Sharpe Ratio"]
                ),
                "Legacy Net Maximum Drawdown": legacy_net_metrics.loc[
                    strategy,
                    "Maximum Drawdown",
                ],
                "Execution Net Maximum Drawdown": execution_net_metrics.loc[
                    strategy,
                    "Maximum Drawdown",
                ],
                "Net MDD Delta": (
                    execution_net_metrics.loc[strategy, "Maximum Drawdown"]
                    - legacy_net_metrics.loc[strategy, "Maximum Drawdown"]
                ),
                "Legacy Annualized Turnover": legacy_turnover.loc[
                    strategy,
                    "Annualized Turnover",
                ],
                "Execution Annualized Turnover": execution_turnover.loc[
                    strategy,
                    "Annualized Turnover",
                ],
                "Turnover Delta": (
                    execution_turnover.loc[strategy, "Annualized Turnover"]
                    - legacy_turnover.loc[strategy, "Annualized Turnover"]
                ),
                "Net Monthly Return Correlation": paired_returns["legacy"].corr(
                    paired_returns["execution"]
                ),
                "Mean Absolute Net Monthly Return Gap": return_gaps.mean(),
                "Maximum Absolute Net Monthly Return Gap": return_gaps.max(),
            }
        )

    return pd.DataFrame(rows)


def compare_adjusted_close_snapshots(
    phase0_adjusted_close_prices,
    execution_market_data,
    tickers,
):
    """Measure same-date adjusted-close differences across frozen snapshots."""
    legacy_long = (
        phase0_adjusted_close_prices.reindex(columns=tickers)
        .rename_axis("Date")
        .stack(future_stack=True)
        .rename("Legacy AdjClose")
        .reset_index(name="Legacy AdjClose")
        .rename(columns={"level_1": "Ticker"})
    )
    execution_long = execution_market_data[["Date", "Ticker", "AdjClose"]].copy()
    execution_long["Date"] = pd.to_datetime(execution_long["Date"])
    execution_long = execution_long.rename(
        columns={"AdjClose": "Execution AdjClose"}
    )
    merged = legacy_long.merge(
        execution_long,
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    ).dropna()
    if merged.empty:
        raise ValueError("No overlapping adjusted-close observations.")

    merged["Absolute Difference"] = (
        merged["Execution AdjClose"] - merged["Legacy AdjClose"]
    ).abs()
    merged["Relative Difference"] = (
        merged["Absolute Difference"] / merged["Legacy AdjClose"].abs()
    )
    absolute_row = merged.loc[merged["Absolute Difference"].idxmax()]
    relative_row = merged.loc[merged["Relative Difference"].idxmax()]
    return {
        "overlap_count": len(merged),
        "max_absolute_difference": absolute_row["Absolute Difference"],
        "max_absolute_date": absolute_row["Date"],
        "max_absolute_ticker": absolute_row["Ticker"],
        "max_relative_difference": relative_row["Relative Difference"],
        "max_relative_date": relative_row["Date"],
        "max_relative_ticker": relative_row["Ticker"],
    }
