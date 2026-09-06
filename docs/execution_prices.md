# Execution-price reference

Signal formation remains based on the frozen Phase 0 adjusted-close research
snapshot. Historical execution timing is derived from observed sessions in the
separate frozen execution-market snapshot, and each scheduled execution date
is mapped only to its exact same-date raw Open price.

Raw Open is a historical next-session execution reference, not an assumption
about future paper or live fills. Adjusted Close is never substituted for an
execution price and is not suitable for execution quantity or actual-position
valuation. A missing exact execution Open remains an explicit `NaN` data
exception; no date, price, or field substitution occurs.

The execution reference is calculated in memory only. Execution returns,
portfolio P&L, quantities, orders, fills, spread, slippage, transaction costs,
and corporate-action accounting are intentionally deferred.
