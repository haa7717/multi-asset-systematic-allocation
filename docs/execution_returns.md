# Execution-aware total-return asset returns

Legacy Phase 0 measures total returns from adjusted calendar-month-end Close
to adjusted calendar-month-end Close. Execution-aware research instead measures
next-session execution Open to next-session execution Open using a research-only
total-return adjusted-open series.

For each exact execution date and ticker:

```text
adjustment_factor = AdjClose / Close
adjusted_execution_open = Open * adjustment_factor
```

The adjustment factor uses only frozen same-date fields. Missing Open, Close,
or Adjusted Close, and zero or negative Close, produce an explicit missing
adjusted execution Open. No filling, interpolation, date substitution, or
ticker substitution is applied.

Adjusted execution Open is a research-only total-return construct. Raw Open
remains the historical execution-price reference and is never replaced for
orders, quantities, or actual-position valuation. This step does not model
broker fills, quantity accounting, explicit dividend or split booking, or
corporate-action/cash/position accounting; those are deferred.

The return matrix uses `pct_change(fill_method=None)` without a manual shift.
Consequently, the row labelled formation date `t` measures adjusted execution
Open at `t` relative to adjusted execution Open at `t-1`. Its purpose is to
isolate execution-timing effects before adding further implementation frictions.
