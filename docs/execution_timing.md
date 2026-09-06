# Execution timing contract

Phase 0 retains its frozen research convention: information labelled at
calendar month-end produces the next monthly holding return.  The legacy
calendar label is preserved as `legacy_formation_date`.

For execution-aware research, `formation_month` is the calendar month of that
legacy label. `signal_date` is the final actually observed market session in
that month. The signal and target are known only after that session's close,
and `order_date` is therefore the same date: it denotes order generation after
the close, not an artificial intraday timestamp.

The historical execution reference is the Open of the first actually observed
trading session strictly after `signal_date`, recorded as `execution_date`.
The schedule derives both dates exclusively from supplied observed daily dates:
it uses no calendar-day arithmetic, business-day arithmetic, filling, or
invented market calendar. A final formation row with no later observed session
is retained with `execution_date = NaT`.

These historical values are date-level pandas timestamps. Future paper/live
order and fill timestamps will be timezone-aware in `America/New_York`.

Next-open is only a historical reference convention. It is not an assumption
about future paper or live fills. Spread, slippage, order, fill, quantity,
portfolio-accounting, and execution-simulation logic are intentionally
deferred. No Phase 0 calculation or reported result has changed.
