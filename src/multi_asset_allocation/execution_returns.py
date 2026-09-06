"""Research-only total-return adjusted execution-open price calculations."""

import numpy as np
import pandas as pd

from multi_asset_allocation.execution_prices import build_execution_open_matrix


def build_adjusted_execution_open_matrix(
    schedule,
    execution_market_data,
    tickers,
):
    """Map exact-date raw Opens to same-date total-return adjusted Opens.

    The result is a research return-measurement series. It is not a tradable
    price and must not be used for orders, quantities, or position valuation.
    """
    raw_execution_opens = build_execution_open_matrix(
        schedule,
        execution_market_data,
        tickers,
    )
    market_fields = execution_market_data[
        ["Date", "Ticker", "Close", "AdjClose"]
    ].copy()
    market_fields["Date"] = pd.to_datetime(market_fields["Date"])
    valid_fields = (
        market_fields["Close"].notna()
        & market_fields["AdjClose"].notna()
        & np.isfinite(market_fields["Close"])
        & np.isfinite(market_fields["AdjClose"])
        & (market_fields["Close"] > 0)
    )
    valid_adjustment_factors = (
        market_fields.assign(
            adjustment_factor=(
                market_fields["AdjClose"] / market_fields["Close"]
            ).where(valid_fields)
        )
        .pivot(index="Date", columns="Ticker", values="adjustment_factor")
        .reindex(columns=tickers)
    )

    execution_dates = pd.DatetimeIndex(pd.to_datetime(schedule["execution_date"]))
    factor_matrix = valid_adjustment_factors.reindex(execution_dates)
    factor_matrix.index = raw_execution_opens.index
    return raw_execution_opens * factor_matrix


def calculate_execution_open_returns(adjusted_execution_open_matrix):
    """Return execution-open-to-execution-open total returns without shifting."""
    return adjusted_execution_open_matrix.pct_change(fill_method=None)
