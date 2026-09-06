from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "execution" / "execution_market_snapshot.csv"
TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "AdjClose"]
ACTION_COLUMNS = ["Dividends", "StockSplits", "CapitalGains"]
EXPECTED_COLUMNS = [
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


def _snapshot():
    return pd.read_csv(SNAPSHOT, parse_dates=["Date"])


def test_execution_market_snapshot_schema_order_and_uniqueness():
    snapshot = _snapshot()

    assert list(snapshot.columns) == EXPECTED_COLUMNS
    assert snapshot["Ticker"].drop_duplicates().tolist() == TICKERS
    assert snapshot[["Date", "Ticker"]].duplicated().sum() == 0

    ticker_rank = snapshot["Ticker"].map(
        {ticker: rank for rank, ticker in enumerate(TICKERS)}
    )
    expected = snapshot.assign(_ticker_rank=ticker_rank).sort_values(
        ["Date", "_ticker_rank"],
        kind="stable",
    )
    assert snapshot.index.tolist() == expected.index.tolist()
    assert snapshot["Date"].dt.tz is None


def test_execution_market_prices_volume_and_actions_are_valid():
    snapshot = _snapshot()

    for column in ["Open", "High", "Low", "Close"]:
        assert (snapshot[column].dropna() > 0).all()
    assert (snapshot["Volume"].dropna() >= 0).all()
    assert snapshot[ACTION_COLUMNS].isna().sum().sum() == 0


def test_execution_market_snapshot_has_july_first_open_if_session_is_present():
    snapshot = _snapshot()
    july_first = snapshot.loc[
        snapshot["Date"] == pd.Timestamp("2026-07-01")
    ]

    if not july_first.empty:
        assert july_first["Ticker"].tolist() == TICKERS
        assert july_first["Open"].notna().all()


def test_execution_market_reports_actual_date_set_relationships(capsys):
    snapshot = _snapshot()
    spy_dates = set(snapshot.loc[snapshot["Ticker"] == "SPY", "Date"])

    for ticker in TICKERS:
        ticker_dates = set(snapshot.loc[snapshot["Ticker"] == ticker, "Date"])
        missing = len(spy_dates - ticker_dates)
        extra = len(ticker_dates - spy_dates)
        print(f"{ticker}: matches_spy={ticker_dates == spy_dates}, missing={missing}, extra={extra}")

    output = capsys.readouterr().out
    for ticker in TICKERS:
        assert f"{ticker}:" in output
