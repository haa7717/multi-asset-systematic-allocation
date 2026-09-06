from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from multi_asset_allocation.execution_prices import (
    build_execution_open_matrix,
    build_execution_reference,
)
from multi_asset_allocation.execution_timing import build_execution_schedule


ROOT = Path(__file__).resolve().parents[1]
PHASE0_SNAPSHOT = ROOT / "data" / "phase0" / "phase0_adjusted_close_snapshot.csv"
EXECUTION_SNAPSHOT = ROOT / "data" / "execution" / "execution_market_snapshot.csv"
TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]


def _frozen_schedule_and_market_data():
    phase0_prices = pd.read_csv(
        PHASE0_SNAPSHOT,
        index_col="Date",
        parse_dates=True,
    )
    execution_market_data = pd.read_csv(
        EXECUTION_SNAPSHOT,
        parse_dates=["Date"],
    )
    formation_dates = phase0_prices.resample("ME").last().index
    spy_dates = execution_market_data.loc[
        execution_market_data["Ticker"] == "SPY",
        "Date",
    ]
    schedule = build_execution_schedule(spy_dates, formation_dates)
    return schedule, execution_market_data


def test_exact_date_raw_open_mapping_preserves_order_and_ticker_columns():
    schedule = pd.DataFrame(
        {
            "formation_month": pd.PeriodIndex(["2024-01", "2024-02"], freq="M"),
            "legacy_formation_date": pd.to_datetime(
                ["2024-01-31", "2024-02-29"]
            ),
            "signal_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "order_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "execution_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
        }
    )
    market_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-02-01", "2024-02-01", "2024-03-01", "2024-03-01"]
            ),
            "Ticker": ["B", "A", "B", "A"],
            "Open": [200.0, 100.0, 201.0, 101.0],
            "Close": [999.0, 999.0, 999.0, 999.0],
            "AdjClose": [888.0, 888.0, 888.0, 888.0],
        }
    )

    matrix = build_execution_open_matrix(schedule, market_data, ["A", "B"])

    assert matrix.columns.tolist() == ["A", "B"]
    pdt.assert_index_equal(
        matrix.index,
        pd.DatetimeIndex(["2024-01-31", "2024-02-29"], name="legacy_formation_date"),
    )
    np.testing.assert_allclose(matrix.iloc[0], [100.0, 200.0])
    np.testing.assert_allclose(matrix.iloc[1], [101.0, 201.0])


def test_missing_exact_open_remains_nan_without_price_or_date_substitution():
    schedule = pd.DataFrame(
        {
            "formation_month": pd.PeriodIndex(["2024-01"], freq="M"),
            "legacy_formation_date": pd.to_datetime(["2024-01-31"]),
            "signal_date": pd.to_datetime(["2024-01-31"]),
            "order_date": pd.to_datetime(["2024-01-31"]),
            "execution_date": pd.to_datetime(["2024-02-01"]),
        }
    )
    market_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-01"]),
            "Ticker": ["A", "A"],
            "Open": [10.0, np.nan],
            "Close": [12.0, 13.0],
            "AdjClose": [11.0, 12.0],
        }
    )

    matrix = build_execution_open_matrix(schedule, market_data, ["A", "B"])

    assert pd.isna(matrix.loc[pd.Timestamp("2024-01-31"), "A"])
    assert pd.isna(matrix.loc[pd.Timestamp("2024-01-31"), "B"])


def test_frozen_execution_reference_uses_spy_sessions_and_raw_next_open():
    schedule, market_data = _frozen_schedule_and_market_data()
    matrix = build_execution_open_matrix(schedule, market_data, TICKERS)
    reference = build_execution_reference(schedule, market_data, TICKERS)

    assert matrix.columns.tolist() == TICKERS
    assert matrix.index.tolist() == schedule["legacy_formation_date"].tolist()
    assert len(matrix) == len(schedule) == 234
    assert reference.columns.tolist() == [
        "formation_month",
        "legacy_formation_date",
        "signal_date",
        "order_date",
        "execution_date",
        *TICKERS,
    ]

    may_2026 = reference.loc[
        reference["legacy_formation_date"] == pd.Timestamp("2026-05-31")
    ].iloc[0]
    assert may_2026["signal_date"] == pd.Timestamp("2026-05-29")
    assert may_2026["execution_date"] == pd.Timestamp("2026-06-01")

    final_row = reference.iloc[-1]
    assert final_row["legacy_formation_date"] == pd.Timestamp("2026-06-30")
    assert final_row["signal_date"] == pd.Timestamp("2026-06-30")
    assert final_row["execution_date"] == pd.Timestamp("2026-07-01")

    expected_final_opens = market_data.loc[
        (market_data["Date"] == pd.Timestamp("2026-07-01"))
        & market_data["Ticker"].isin(TICKERS),
        ["Ticker", "Open"],
    ].set_index("Ticker").reindex(TICKERS)["Open"]
    np.testing.assert_allclose(final_row[TICKERS].astype(float), expected_final_opens)


def test_frozen_completed_rows_have_strictly_later_dates_and_complete_opens():
    schedule, market_data = _frozen_schedule_and_market_data()
    matrix = build_execution_open_matrix(schedule, market_data, TICKERS)

    completed = schedule.dropna(subset=["execution_date"])
    assert (completed["execution_date"] > completed["signal_date"]).all()
    assert matrix.loc[completed["legacy_formation_date"]].notna().all(axis=1).all()
