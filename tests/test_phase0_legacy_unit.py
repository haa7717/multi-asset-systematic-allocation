import numpy as np
import pandas as pd
import pandas.testing as pdt

from multi_asset_allocation.phase0_legacy import run_monthly_backtest


def test_target_and_turnover_are_applied_to_the_following_holding_period():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    targets = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=dates)
    cash = pd.Series([0.5, 0.5, 0.5], index=dates)
    returns = pd.DataFrame({"A": [0.10, 0.20, -0.10]}, index=dates)

    result = run_monthly_backtest(
        targets,
        cash,
        returns,
        transaction_cost=0.01,
        cash_return=0.0,
    )

    assert np.isnan(result["gross_return"].iloc[0])
    assert result["turnover"].iloc[0] == 0.5
    assert result["gross_return"].iloc[1] == 0.10
    assert result["transaction_cost"].iloc[1] == 0.005
    assert result["net_return"].iloc[1] == 0.095
    assert result["gross_return"].iloc[2] == -0.05
    # The February holding-period return drifts the risky sleeve before the
    # February formation rebalance, whose cost is charged in March.
    np.testing.assert_allclose(
        result["transaction_cost"].iloc[2],
        0.01 * abs(0.5 - (0.5 * 1.2 / 1.1)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_opening_from_cash_and_drift_normalization_are_legacy_compatible():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    targets = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
        index=dates,
    )
    cash = pd.Series([0.0, 0.0, 0.0], index=dates)
    returns = pd.DataFrame(
        {"A": [0.0, 0.10, 0.0], "B": [0.0, -0.10, 0.0]},
        index=dates,
    )

    result = run_monthly_backtest(
        targets,
        cash,
        returns,
        transaction_cost=0.0,
        cash_return=0.0,
    )

    pdt.assert_series_equal(
        result["pre_rebalance_weights"].iloc[0],
        pd.Series({"A": 0.0, "B": 0.0}, name=dates[0]),
        check_names=True,
    )
    pdt.assert_series_equal(
        result["pre_rebalance_weights"].iloc[1],
        pd.Series({"A": 0.55, "B": 0.45}, name=dates[1]),
        check_names=True,
    )
    assert result["pre_rebalance_cash"].iloc[0] == 1.0
    assert result["turnover"].iloc[0] == 1.0
    np.testing.assert_allclose(
        result["turnover"].iloc[1],
        0.10,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_return_cash_contributes_no_holding_return():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    targets = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
    cash = pd.Series([1.0, 1.0], index=dates)
    returns = pd.DataFrame({"A": [0.25, -0.25]}, index=dates)

    result = run_monthly_backtest(
        targets,
        cash,
        returns,
        transaction_cost=0.0,
        cash_return=0.0,
    )

    assert result["turnover"].iloc[0] == 0.0
    assert result["gross_return"].iloc[1] == 0.0
    assert result["net_return"].iloc[1] == 0.0
