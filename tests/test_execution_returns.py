from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from multi_asset_allocation.execution_returns import (
    build_adjusted_execution_open_matrix,
    calculate_execution_open_returns,
)
from multi_asset_allocation.execution_timing import build_execution_schedule


ROOT = Path(__file__).resolve().parents[1]
PHASE0_SNAPSHOT = ROOT / "data" / "phase0" / "phase0_adjusted_close_snapshot.csv"
EXECUTION_SNAPSHOT = ROOT / "data" / "execution" / "execution_market_snapshot.csv"
TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]


def _schedule(execution_dates):
    execution_dates = pd.to_datetime(execution_dates)
    return pd.DataFrame(
        {
            "formation_month": execution_dates.to_period("M"),
            "legacy_formation_date": execution_dates - pd.offsets.MonthEnd(1),
            "signal_date": execution_dates - pd.Timedelta(days=1),
            "order_date": execution_dates - pd.Timedelta(days=1),
            "execution_date": execution_dates,
        }
    )


def _frozen_schedule_and_market_data():
    phase0_prices = pd.read_csv(
        PHASE0_SNAPSHOT,
        index_col="Date",
        parse_dates=True,
    )
    market_data = pd.read_csv(EXECUTION_SNAPSHOT, parse_dates=["Date"])
    formation_dates = phase0_prices.resample("ME").last().index
    spy_dates = market_data.loc[market_data["Ticker"] == "SPY", "Date"]
    return build_execution_schedule(spy_dates, formation_dates), market_data


def test_exact_adjustment_formula_and_equal_adjusted_close_case():
    schedule = _schedule(["2024-02-01"])
    market_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
            "Ticker": ["A", "B"],
            "Open": [100.0, 200.0],
            "Close": [50.0, 80.0],
            "AdjClose": [75.0, 80.0],
        }
    )

    adjusted_opens = build_adjusted_execution_open_matrix(
        schedule,
        market_data,
        ["A", "B"],
    )

    np.testing.assert_allclose(adjusted_opens.iloc[0], [150.0, 200.0])


def test_missing_or_invalid_exact_fields_remain_nan():
    schedule = _schedule(["2024-02-01"])
    market_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-02-01"] * 5),
            "Ticker": ["missing_open", "missing_close", "missing_adj", "zero", "negative"],
            "Open": [np.nan, 10.0, 10.0, 10.0, 10.0],
            "Close": [10.0, np.nan, 10.0, 0.0, -5.0],
            "AdjClose": [10.0, 10.0, np.nan, 10.0, 10.0],
        }
    )
    tickers = ["missing_open", "missing_close", "missing_adj", "zero", "negative"]

    adjusted_opens = build_adjusted_execution_open_matrix(
        schedule,
        market_data,
        tickers,
    )

    assert adjusted_opens.iloc[0].isna().all()
    assert not np.isinf(adjusted_opens.to_numpy(dtype=float)).any()


def test_adjustment_factor_uses_only_the_exact_execution_date():
    schedule = _schedule(["2024-02-02"])
    market_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-02-01", "2024-02-02"]),
            "Ticker": ["A", "A"],
            "Open": [100.0, 100.0],
            "Close": [100.0, 50.0],
            "AdjClose": [200.0, 75.0],
        }
    )

    adjusted_opens = build_adjusted_execution_open_matrix(
        schedule,
        market_data,
        ["A"],
    )

    assert adjusted_opens.iloc[0, 0] == 150.0


def test_frozen_adjusted_execution_opens_and_returns_have_expected_alignment():
    schedule, market_data = _frozen_schedule_and_market_data()
    adjusted_opens = build_adjusted_execution_open_matrix(
        schedule,
        market_data,
        TICKERS,
    )
    execution_returns = calculate_execution_open_returns(adjusted_opens)

    assert adjusted_opens.columns.tolist() == TICKERS
    pdt.assert_index_equal(
        adjusted_opens.index,
        pd.DatetimeIndex(
            schedule["legacy_formation_date"],
            name="legacy_formation_date",
        ),
    )
    assert adjusted_opens.shape == (234, 8)
    assert adjusted_opens.notna().all(axis=1).all()

    assert execution_returns.shape == (234, 8)
    assert execution_returns.iloc[0].isna().all()
    assert execution_returns.iloc[1:].notna().all(axis=1).all()

    may_row = pd.Timestamp("2026-05-31")
    april_row = pd.Timestamp("2026-04-30")
    june_row = pd.Timestamp("2026-06-30")
    np.testing.assert_allclose(
        execution_returns.loc[may_row],
        adjusted_opens.loc[may_row] / adjusted_opens.loc[april_row] - 1.0,
    )
    np.testing.assert_allclose(
        execution_returns.loc[june_row],
        adjusted_opens.loc[june_row] / adjusted_opens.loc[may_row] - 1.0,
    )

    timing = schedule.set_index("legacy_formation_date")
    assert timing.loc[may_row, "execution_date"] == pd.Timestamp("2026-06-01")
    assert timing.loc[june_row, "execution_date"] == pd.Timestamp("2026-07-01")
