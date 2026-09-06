# Legacy versus execution-aware timing comparison

This is an implementation-gap experiment, not a new strategy. Signal and
target logic are held fixed across four Phase 0 strategies; only the
holding-period return convention changes.

Legacy research uses adjusted calendar-month-end Close-to-Close returns.
Execution-aware research uses total-return adjusted next-session Open-to-Open
returns. Raw Open remains the historical executable-price reference, while
adjusted Open remains research-only. The same 10-basis-point risky-turnover
cost rule is used for both conventions.

Turnover can differ because holding-period returns produce different drifted
pre-rebalance weights. The final execution-aware return labelled
`2026-06-30` closes at `2026-07-01` Open and still uses the target formed in
the prior month.

No conclusion about strategy quality is implied. Orders, quantities, actual
fills, spread, slippage, explicit dividends, portfolio state, P&L accounting,
and corporate-action accounting remain deferred.

## Data consistency diagnostic

The maximum absolute frozen adjusted-close difference is `0.000213623046875`,
at SPY on 2018-09-28. The maximum relative frozen adjusted-close difference is
`1.8770977036112281e-06`, at LQD on 2008-02-19. The predefined `1e-6`
human-review threshold was therefore exceeded.

Human review accepted the discrepancy as non-material for this timing
comparison. The maximum relative discrepancy is approximately 0.0188 basis
points in price, far smaller than the observed monthly legacy-versus-execution
return gaps. No input data, strategy formula, threshold, or result was changed
in response.
