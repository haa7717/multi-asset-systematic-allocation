"""Date-level execution timing derived solely from observed market sessions."""

import pandas as pd


def build_execution_schedule(observed_market_dates, legacy_formation_dates):
    """Map legacy calendar month-end labels to signal and execution dates.

    ``signal_date`` is the final supplied market session in each formation
    month.  ``execution_date`` is the first supplied market session strictly
    after that signal date.  No calendar arithmetic or inferred sessions are
    used; a missing next session remains ``NaT``.
    """
    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(observed_market_dates)
    ).normalize().unique().sort_values()
    formation_dates = pd.DatetimeIndex(
        pd.to_datetime(legacy_formation_dates)
    ).normalize()
    formation_months = formation_dates.to_period("M")

    observed_months = observed_dates.to_period("M")
    last_signal_by_month = pd.Series(
        observed_dates,
        index=observed_months,
    ).groupby(level=0, sort=False).max()

    signal_dates = pd.DatetimeIndex(
        [last_signal_by_month.get(month, pd.NaT) for month in formation_months]
    )
    execution_dates = []
    for signal_date in signal_dates:
        if pd.isna(signal_date):
            execution_dates.append(pd.NaT)
            continue

        next_position = observed_dates.searchsorted(signal_date, side="right")
        execution_dates.append(
            observed_dates[next_position]
            if next_position < len(observed_dates)
            else pd.NaT
        )

    return pd.DataFrame(
        {
            "formation_month": formation_months,
            "legacy_formation_date": formation_dates,
            "signal_date": signal_dates,
            "order_date": signal_dates,
            "execution_date": pd.DatetimeIndex(execution_dates),
        }
    )
