from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from multi_asset_allocation.execution_timing import build_execution_schedule


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "phase0" / "phase0_adjusted_close_snapshot.csv"


def test_actual_trading_month_end_is_the_signal_date():
    observed_dates = pd.to_datetime(
        ["2024-01-30", "2024-01-31", "2024-02-01"]
    )
    schedule = build_execution_schedule(
        observed_dates,
        pd.to_datetime(["2024-01-31"]),
    )

    assert schedule.loc[0, "formation_month"] == pd.Period("2024-01", "M")
    assert schedule.loc[0, "legacy_formation_date"] == pd.Timestamp(
        "2024-01-31"
    )
    assert schedule.loc[0, "signal_date"] == pd.Timestamp("2024-01-31")
    assert schedule.loc[0, "order_date"] == pd.Timestamp("2024-01-31")
    assert schedule.loc[0, "execution_date"] == pd.Timestamp("2024-02-01")


def test_weekend_month_end_uses_observed_sessions_not_calendar_arithmetic():
    observed_dates = pd.to_datetime(
        ["2024-03-28", "2024-03-29", "2024-04-01", "2024-04-02"]
    )
    schedule = build_execution_schedule(
        observed_dates,
        pd.to_datetime(["2024-03-31"]),
    )

    assert schedule.loc[0, "signal_date"] == pd.Timestamp("2024-03-29")
    assert schedule.loc[0, "order_date"] == pd.Timestamp("2024-03-29")
    assert schedule.loc[0, "execution_date"] == pd.Timestamp("2024-04-01")
    assert schedule.loc[0, "execution_date"] > schedule.loc[0, "signal_date"]


def test_missing_calendar_sessions_are_not_replaced_with_invented_dates():
    observed_dates = pd.to_datetime(
        ["2024-05-24", "2024-05-28", "2024-06-03"]
    )
    schedule = build_execution_schedule(
        observed_dates,
        pd.to_datetime(["2024-05-31"]),
    )

    assert schedule.loc[0, "signal_date"] == pd.Timestamp("2024-05-28")
    assert schedule.loc[0, "execution_date"] == pd.Timestamp("2024-06-03")


def test_order_and_execution_dates_preserve_formation_order_and_missing_next_date():
    observed_dates = pd.to_datetime(
        ["2024-01-31", "2024-02-01", "2024-02-29"]
    )
    formation_dates = pd.to_datetime(["2024-02-29", "2024-01-31"])
    schedule = build_execution_schedule(observed_dates, formation_dates)

    pdt.assert_series_equal(
        schedule["legacy_formation_date"],
        pd.Series(formation_dates, name="legacy_formation_date"),
    )
    pdt.assert_series_equal(
        schedule["order_date"],
        schedule["signal_date"],
        check_names=False,
    )
    assert pd.isna(schedule.loc[0, "execution_date"])
    assert schedule.loc[1, "execution_date"] == pd.Timestamp("2024-02-01")


def test_frozen_snapshot_schedule_preserves_legacy_labels_and_actual_sessions():
    prices = pd.read_csv(SNAPSHOT, index_col="Date", parse_dates=True)
    legacy_formation_dates = prices.resample("ME").last().index
    schedule = build_execution_schedule(prices.index, legacy_formation_dates)

    pdt.assert_index_equal(
        pd.DatetimeIndex(schedule["legacy_formation_date"]),
        legacy_formation_dates,
        check_names=False,
    )
    assert len(schedule) == len(legacy_formation_dates) == 234

    may_2026 = schedule.loc[
        schedule["legacy_formation_date"] == pd.Timestamp("2026-05-31")
    ].iloc[0]
    assert may_2026["signal_date"] == pd.Timestamp("2026-05-29")
    assert may_2026["execution_date"] == pd.Timestamp("2026-06-01")

    june_2026 = schedule.iloc[-1]
    assert june_2026["legacy_formation_date"] == pd.Timestamp("2026-06-30")
    assert june_2026["signal_date"] == pd.Timestamp("2026-06-30")
    assert june_2026["order_date"] == pd.Timestamp("2026-06-30")
    assert pd.isna(june_2026["execution_date"])

    complete_rows = schedule.dropna(subset=["execution_date"])
    assert (complete_rows["execution_date"] > complete_rows["signal_date"]).all()
