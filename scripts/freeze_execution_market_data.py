"""Freeze raw daily execution-market data from yfinance into a long CSV."""

import argparse
import hashlib
import platform
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]
START_DATE = "2007-01-01"
END_DATE = "2026-07-02"
OUTPUT_PATH = Path("data/execution/execution_market_snapshot.csv")
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose"]
ACTION_COLUMNS = ["Dividends", "StockSplits", "CapitalGains"]
CANONICAL_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "AdjClose",
    "Volume",
    "Dividends",
    "StockSplits",
    "CapitalGains",
]


def _date_level_index(index):
    """Return a timezone-naive, normalized daily index without shifting dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _normalize_history(history, ticker):
    """Normalize one raw yfinance history frame to the canonical long schema."""
    history = history.copy()
    history.index = _date_level_index(history.index)
    history.index.name = "Date"
    history = history.rename(
        columns={
            "Adj Close": "AdjClose",
            "Stock Splits": "StockSplits",
            "Capital Gains": "CapitalGains",
        }
    )

    source_action_availability = {
        column: column in history.columns for column in ACTION_COLUMNS
    }
    for column in PRICE_COLUMNS + ["Volume"]:
        if column not in history.columns:
            history[column] = float("nan")
    for column in ACTION_COLUMNS:
        if column not in history.columns:
            history[column] = 0.0
        else:
            history[column] = history[column].fillna(0.0)

    normalized = history.reset_index()
    normalized["Ticker"] = ticker
    normalized = normalized[CANONICAL_COLUMNS]
    return normalized, source_action_availability


def _date_set_discrepancies(snapshot):
    """Describe observed ticker dates relative to SPY without changing any rows."""
    spy_dates = set(snapshot.loc[snapshot["Ticker"] == "SPY", "Date"])
    discrepancies = {}
    for ticker in TICKERS:
        ticker_dates = set(snapshot.loc[snapshot["Ticker"] == ticker, "Date"])
        missing_from_ticker = sorted(spy_dates - ticker_dates)
        extra_for_ticker = sorted(ticker_dates - spy_dates)
        discrepancies[ticker] = (missing_from_ticker, extra_for_ticker)
    return discrepancies


def _print_validation_summary(snapshot, action_availability, output_path):
    """Print concise reproducibility metadata for the newly frozen CSV."""
    missing_prices = snapshot[PRICE_COLUMNS].isna().sum()
    discrepancies = _date_set_discrepancies(snapshot)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print("=== EXECUTION MARKET SNAPSHOT ===")
    print("Python:", platform.python_version())
    print("pandas:", pd.__version__)
    print("yfinance:", yf.__version__)
    print("Rows:", len(snapshot))
    print("First date:", snapshot["Date"].min().date())
    print("Last date:", snapshot["Date"].max().date())
    print("Ticker count:", snapshot["Ticker"].nunique())
    print("Missing price values:")
    for column, count in missing_prices.items():
        print(f"  {column}: {count}")
    print("Corporate-action source availability:")
    for column in ACTION_COLUMNS:
        present = [
            ticker
            for ticker in TICKERS
            if action_availability[ticker][column]
        ]
        print(f"  {column}: {len(present)}/{len(TICKERS)} ({', '.join(present)})")
    print("Ticker date-set discrepancies relative to SPY:")
    for ticker, (missing, extra) in discrepancies.items():
        print(f"  {ticker}: missing={len(missing)}, extra={len(extra)}")
    print("SHA-256:", digest)


def freeze_execution_market_data(output_path, overwrite=False):
    """Download, normalize, validate, and deterministically write the snapshot."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; rerun with --overwrite to replace it."
        )

    frames = []
    action_availability = {}
    for ticker in TICKERS:
        history = yf.Ticker(ticker).history(
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            prepost=False,
            actions=True,
            auto_adjust=False,
            back_adjust=False,
            repair=False,
            keepna=True,
            rounding=False,
            timeout=30,
            raise_errors=True,
        )
        if history.empty:
            raise ValueError(f"No history returned for {ticker}.")
        normalized, availability = _normalize_history(history, ticker)
        frames.append(normalized)
        action_availability[ticker] = availability

    snapshot = pd.concat(frames, ignore_index=True)
    ticker_rank = {ticker: rank for rank, ticker in enumerate(TICKERS)}
    snapshot["_ticker_rank"] = snapshot["Ticker"].map(ticker_rank)
    snapshot = snapshot.sort_values(
        ["Date", "_ticker_rank"],
        kind="stable",
    ).drop(columns="_ticker_rank")
    snapshot = snapshot[CANONICAL_COLUMNS].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    _print_validation_summary(snapshot, action_availability, output_path)
    return snapshot


def main():
    parser = argparse.ArgumentParser(
        description="Freeze raw yfinance execution-market data."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing execution-market snapshot",
    )
    args = parser.parse_args()
    freeze_execution_market_data(OUTPUT_PATH, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
