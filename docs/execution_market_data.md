# Execution-market dataset

The frozen Phase 0 adjusted-close snapshot remains the signal and legacy
research-regression dataset. It is deliberately separate from
`data/execution/execution_market_snapshot.csv`.

The execution-market snapshot stores observed sessions, raw tradable Open,
High, Low, and Close prices, Volume, Adjusted Close for audit, and corporate
actions. Raw Open will later be the historical next-session execution
reference; raw Close will later support position mark-to-market accounting.
Adjusted prices are inappropriate for executable order quantities and actual
position valuation because they retrospectively incorporate corporate actions.

Dividends, stock splits, and capital gains are captured now, with absent action
values normalized to zero. Corporate-action, quantity, position-accounting,
order, fill, spread, and slippage logic is intentionally deferred to later
steps.

The CSV is a frozen historical snapshot created with yfinance daily history for
the eight-ETF universe, `auto_adjust=False`, `back_adjust=False`,
`repair=False`, `actions=True`, and `keepna=True`. It uses daily sessions only;
pre/post-market data is excluded, rounding is disabled, and no filling,
interpolation, or trading-calendar inference occurs. The script uses a
30-second request timeout and raises download errors rather than silently
continuing.

This snapshot is not a paper/live feed. Paper and live data will use a separate
future update workflow and must not overwrite this frozen historical file.
