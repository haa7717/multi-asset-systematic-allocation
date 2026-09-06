"""Exact-date mapping from execution schedules to frozen raw Open prices."""

import pandas as pd


TIMING_COLUMNS = [
    "formation_month",
    "legacy_formation_date",
    "signal_date",
    "order_date",
    "execution_date",
]


def build_execution_open_matrix(schedule, execution_market_data, tickers):
    """Return raw Opens observed exactly on each scheduled execution date.

    Rows retain schedule order and are indexed by legacy formation date. Missing
    exact-date data remains ``NaN``; no price or date substitution is applied.
    """
    market_data = execution_market_data[["Date", "Ticker", "Open"]].copy()
    market_data["Date"] = pd.to_datetime(market_data["Date"])
    open_by_date = market_data.pivot(
        index="Date",
        columns="Ticker",
        values="Open",
    ).reindex(columns=tickers)

    execution_dates = pd.DatetimeIndex(pd.to_datetime(schedule["execution_date"]))
    open_matrix = open_by_date.reindex(execution_dates)
    open_matrix.index = pd.DatetimeIndex(
        pd.to_datetime(schedule["legacy_formation_date"])
    )
    open_matrix.index.name = "legacy_formation_date"
    return open_matrix


def build_execution_reference(schedule, execution_market_data, tickers):
    """Combine the timing contract and exact-date raw execution Opens in memory."""
    timing = schedule[TIMING_COLUMNS].reset_index(drop=True).copy()
    opens = build_execution_open_matrix(
        schedule,
        execution_market_data,
        tickers,
    ).reset_index(drop=True)
    return pd.concat([timing, opens], axis=1)
