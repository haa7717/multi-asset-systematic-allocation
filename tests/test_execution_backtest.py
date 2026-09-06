from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from multi_asset_allocation import execution_backtest, phase0_legacy
from multi_asset_allocation.execution_backtest import (
    build_oos_return_frame,
    build_timing_comparison_table,
    run_strategy_backtests,
)
from multi_asset_allocation.execution_returns import (
    build_adjusted_execution_open_matrix,
    calculate_execution_open_returns,
)
from multi_asset_allocation.execution_timing import build_execution_schedule


ROOT = Path(__file__).resolve().parents[1]
PHASE0_SNAPSHOT = ROOT / "data" / "phase0" / "phase0_adjusted_close_snapshot.csv"
EXECUTION_SNAPSHOT = ROOT / "data" / "execution" / "execution_market_snapshot.csv"
TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]
OOS_START = "2019-01-01"


@pytest.fixture(scope="session")
def comparison_inputs():
    prices = pd.read_csv(PHASE0_SNAPSHOT, index_col="Date", parse_dates=True)
    market_data = pd.read_csv(EXECUTION_SNAPSHOT, parse_dates=["Date"])
    analysis_prices = phase0_legacy.prepare_analysis_prices(
        prices,
        TICKERS,
        "2026-06-30",
    )
    daily_returns = phase0_legacy.calculate_daily_returns(analysis_prices)
    monthly_prices = phase0_legacy.construct_monthly_prices(analysis_prices)
    monthly_returns = phase0_legacy.calculate_monthly_returns(monthly_prices)
    signal = phase0_legacy.calculate_trend_signal(monthly_prices, 12)
    volatility = phase0_legacy.calculate_monthly_realized_volatility(
        daily_returns,
        63,
    )
    equal_weight = phase0_legacy.equal_weight_targets(monthly_prices.index, TICKERS)
    inverse_volatility = phase0_legacy.inverse_volatility_targets(
        volatility,
        len(TICKERS),
    )
    trend_equal_weight = phase0_legacy.trend_equal_weight_targets(
        signal,
        len(TICKERS),
    )
    trend_inverse_volatility = phase0_legacy.trend_inverse_volatility_targets(
        signal,
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
    common_index, _, _ = phase0_legacy.common_formation_index(
        targets,
        cash,
        monthly_prices,
    )
    for name in targets:
        targets[name] = targets[name].reindex(common_index)
        cash[name] = cash[name].reindex(common_index)

    schedule = build_execution_schedule(
        market_data.loc[market_data["Ticker"] == "SPY", "Date"],
        monthly_prices.index,
    )
    execution_returns = calculate_execution_open_returns(
        build_adjusted_execution_open_matrix(schedule, market_data, TICKERS)
    ).reindex(common_index)
    return {
        "monthly_returns": monthly_returns,
        "targets": targets,
        "cash": cash,
        "common_index": common_index,
        "execution_returns": execution_returns,
    }


@pytest.fixture(scope="session")
def comparison_runs(comparison_inputs):
    targets = comparison_inputs["targets"]
    cash = comparison_inputs["cash"]
    return {
        "legacy_gross": run_strategy_backtests(
            targets,
            cash,
            comparison_inputs["monthly_returns"],
            0.0,
            0.0,
        ),
        "execution_gross": run_strategy_backtests(
            targets,
            cash,
            comparison_inputs["execution_returns"],
            0.0,
            0.0,
        ),
        "legacy_net": run_strategy_backtests(
            targets,
            cash,
            comparison_inputs["monthly_returns"],
            0.001,
            0.0,
        ),
        "execution_net": run_strategy_backtests(
            targets,
            cash,
            comparison_inputs["execution_returns"],
            0.001,
            0.0,
        ),
    }


def test_target_and_cash_inputs_are_identical_for_both_return_conventions(
    comparison_inputs,
    comparison_runs,
):
    for strategy in comparison_inputs["targets"]:
        pdt.assert_frame_equal(
            comparison_runs["legacy_net"][strategy]["target_weights"],
            comparison_runs["execution_net"][strategy]["target_weights"],
        )
        pdt.assert_series_equal(
            comparison_runs["legacy_net"][strategy]["target_cash"],
            comparison_runs["execution_net"][strategy]["target_cash"],
        )


def test_runner_uses_the_frozen_legacy_backtest_engine(monkeypatch):
    calls = []
    original = phase0_legacy.run_monthly_backtest

    def recorded_runner(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        execution_backtest.phase0_legacy,
        "run_monthly_backtest",
        recorded_runner,
    )
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    targets = {"Strategy": pd.DataFrame({"A": [1.0, 1.0]}, index=dates)}
    cash = {"Strategy": pd.Series([0.0, 0.0], index=dates)}
    returns = pd.DataFrame({"A": [0.0, 0.01]}, index=dates)

    run_strategy_backtests(targets, cash, returns, 0.001, 0.0)

    assert len(calls) == 1
    assert calls[0]["transaction_cost"] == 0.001
    assert calls[0]["cash_return"] == 0.0


def test_execution_return_alignment_and_oos_coverage(
    comparison_inputs,
    comparison_runs,
):
    assert comparison_inputs["execution_returns"].index.equals(
        comparison_inputs["common_index"]
    )
    legacy_oos = build_oos_return_frame(comparison_runs["legacy_net"], OOS_START)
    execution_oos = build_oos_return_frame(
        comparison_runs["execution_net"],
        OOS_START,
    )
    assert legacy_oos.columns.equals(execution_oos.columns)
    assert legacy_oos.index.equals(execution_oos.index)
    assert len(legacy_oos) == len(execution_oos) == 90
    assert legacy_oos.columns.tolist() == [
        "Equal Weight",
        "Inverse Volatility",
        "Trend + Equal Weight",
        "Trend + Inverse Volatility",
    ]


def test_comparison_uses_gross_and_net_cost_rules_and_aligned_gaps(
    comparison_runs,
):
    table = build_timing_comparison_table(
        comparison_runs["legacy_gross"],
        comparison_runs["execution_gross"],
        comparison_runs["legacy_net"],
        comparison_runs["execution_net"],
        OOS_START,
    ).set_index("Strategy")
    strategy = "Trend + Equal Weight"
    legacy = build_oos_return_frame(comparison_runs["legacy_net"], OOS_START)[strategy]
    execution = build_oos_return_frame(
        comparison_runs["execution_net"],
        OOS_START,
    )[strategy]
    gaps = (execution - legacy).abs()

    assert table.loc[strategy, "Net Monthly Return Correlation"] == legacy.corr(
        execution
    )
    assert table.loc[strategy, "Mean Absolute Net Monthly Return Gap"] == gaps.mean()
    assert table.loc[strategy, "Maximum Absolute Net Monthly Return Gap"] == gaps.max()
    assert not np.allclose(
        table["Legacy Annualized Turnover"],
        table["Execution Annualized Turnover"],
    )
    for convention in ["legacy", "execution"]:
        gross = comparison_runs[f"{convention}_gross"]
        net = comparison_runs[f"{convention}_net"]
        for result in gross.values():
            np.testing.assert_allclose(
                result["transaction_cost"].dropna(),
                0.0,
                rtol=0.0,
                atol=0.0,
            )
        for strategy_name, result in net.items():
            np.testing.assert_allclose(
                result["transaction_cost"].dropna(),
                result["turnover"].shift(1).dropna() * 0.001,
                rtol=1e-12,
                atol=1e-12,
            )
            assert result["target_cash"].equals(
                comparison_runs[f"{convention}_gross"][strategy_name]["target_cash"]
            )


def test_final_execution_return_uses_prior_month_target(
    comparison_inputs,
    comparison_runs,
):
    strategy = "Inverse Volatility"
    june_date = pd.Timestamp("2026-06-30")
    may_date = pd.Timestamp("2026-05-31")
    target_at_may = comparison_inputs["targets"][strategy].loc[may_date]
    target_at_june = comparison_inputs["targets"][strategy].loc[june_date]
    expected = (
        target_at_may * comparison_inputs["execution_returns"].loc[june_date]
    ).sum() + comparison_inputs["cash"][strategy].loc[may_date] * 0.0
    actual = comparison_runs["execution_gross"][strategy]["gross_return"].loc[
        june_date
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(target_at_may, target_at_june)
