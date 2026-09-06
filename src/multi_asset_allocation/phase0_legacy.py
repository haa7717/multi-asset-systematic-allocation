"""Pure, legacy-compatible Phase 0 research calculations.

This module deliberately preserves the completed Phase 0 notebook's monthly
timing, turnover, and transaction-cost conventions.  It contains no data
download, execution, position, or live-trading logic.
"""

import numpy as np
import pandas as pd


def prepare_analysis_prices(prices, tickers, analysis_end):
    """Order a close-price panel and apply the legacy inclusive analysis cutoff."""
    return prices.reindex(columns=tickers).loc[:analysis_end].copy()


def calculate_daily_returns(analysis_prices):
    """Return daily close-to-close returns using the legacy no-fill convention."""
    return analysis_prices.pct_change(fill_method=None)


def construct_monthly_prices(analysis_prices):
    """Return calendar-month-end-labelled final daily closes."""
    return analysis_prices.resample("ME").last()


def calculate_monthly_returns(monthly_prices):
    """Return month-end price returns using the legacy no-fill convention."""
    return monthly_prices.pct_change(fill_method=None)


def calculate_trend_signal(monthly_prices, lookback):
    """Return the positive trailing-return signal while preserving unavailable NaNs."""
    trend_return = monthly_prices.pct_change(
        periods=lookback,
        fill_method=None,
    )
    trend_signal = (trend_return > 0).astype(float)
    return trend_signal.where(trend_return.notna())


def calculate_monthly_realized_volatility(daily_returns, window):
    """Estimate sample daily realized volatility and sample its final monthly value."""
    daily_volatility = (
        daily_returns
        .rolling(window=window, min_periods=window)
        .std()
        * np.sqrt(252)
    )
    return daily_volatility.resample("ME").last()


def equal_weight_targets(monthly_index, tickers):
    """Return fully invested equal-weight targets for every formation month."""
    return pd.DataFrame(
        1.0 / len(tickers),
        index=monthly_index,
        columns=tickers,
    )


def inverse_volatility_targets(monthly_volatility, n_assets):
    """Return full-universe inverse-volatility weights with legacy min-count rules."""
    inverse_volatility_raw = 1.0 / monthly_volatility
    inverse_volatility_denominator = inverse_volatility_raw.sum(
        axis=1,
        min_count=n_assets,
    )
    return inverse_volatility_raw.div(inverse_volatility_denominator, axis=0)


def trend_equal_weight_targets(trend_signal, n_assets):
    """Apply the trend filter to fixed equal-weight sleeves without redistribution."""
    return trend_signal * (1.0 / n_assets)


def trend_inverse_volatility_targets(trend_signal, inverse_volatility_weight):
    """Apply the trend filter to full-universe inverse-volatility targets."""
    return trend_signal * inverse_volatility_weight


def cash_weights(target_weights):
    """Return legacy residual cash weights, including min-count and clipping behavior."""
    return (
        1.0
        - target_weights.sum(axis=1, min_count=target_weights.shape[1])
    ).clip(lower=0.0, upper=1.0)


def zero_cash_weights(monthly_index):
    """Return the legacy zero-cash series used for fully invested strategies."""
    return pd.Series(0.0, index=monthly_index, name="Cash")


def common_formation_index(portfolio_targets, portfolio_cash, monthly_prices):
    """Find the notebook's shared valid target-formation index.

    The returned index deliberately comes from ``monthly_prices.loc`` rather
    than an index intersection so its calendar-month-end alignment is exactly
    the legacy notebook's alignment.
    """
    first_valid_dates = {}
    last_valid_dates = {}

    for name in portfolio_targets:
        weights = portfolio_targets[name]
        cash = portfolio_cash[name]
        valid_mask = weights.notna().all(axis=1) & cash.notna()
        valid_dates = weights.index[valid_mask]
        first_valid_dates[name] = valid_dates[0]
        last_valid_dates[name] = valid_dates[-1]

    common_start = max(first_valid_dates.values())
    common_end = min(last_valid_dates.values())
    index = monthly_prices.loc[common_start:common_end].index
    return index, common_start, common_end


def run_monthly_backtest(
    target_weights,
    cash_weights,
    monthly_asset_returns,
    transaction_cost,
    cash_return,
):
    """Run the Phase 0 drift-aware monthly backtest with explicit cash return.

    Targets formed at month-end ``t`` earn returns during ``t+1``.  Risky-only
    turnover calculated at ``t`` is charged to that following holding period.
    """
    target_weights = target_weights.copy()
    cash_weights = cash_weights.reindex(target_weights.index).copy()

    formation_dates = target_weights.index
    assets = target_weights.columns

    pre_rebalance_weights = pd.DataFrame(
        np.nan,
        index=formation_dates,
        columns=assets,
    )
    pre_rebalance_cash = pd.Series(
        np.nan,
        index=formation_dates,
        name="Pre-Rebalance Cash",
    )
    turnover = pd.Series(
        np.nan,
        index=formation_dates,
        name="Turnover",
    )

    previous_target = None
    previous_cash = None

    for i, date in enumerate(formation_dates):
        current_target = target_weights.loc[date]
        current_cash = float(cash_weights.loc[date])

        if i == 0:
            pre_risky = pd.Series(0.0, index=assets)
            pre_cash = 1.0
        else:
            asset_returns = monthly_asset_returns.loc[date]
            gross_portfolio_value = (
                (previous_target * (1.0 + asset_returns)).sum()
                + previous_cash * (1.0 + cash_return)
            )
            pre_risky = (
                previous_target
                * (1.0 + asset_returns)
                / gross_portfolio_value
            )
            pre_cash = (
                previous_cash
                * (1.0 + cash_return)
                / gross_portfolio_value
            )

        pre_rebalance_weights.loc[date] = pre_risky
        pre_rebalance_cash.loc[date] = pre_cash
        turnover.loc[date] = (current_target - pre_risky).abs().sum()

        previous_target = current_target
        previous_cash = current_cash

    aligned_returns = monthly_asset_returns.reindex(formation_dates)
    lagged_target = target_weights.shift(1)
    lagged_cash = cash_weights.shift(1)
    risky_asset_return = (lagged_target * aligned_returns).sum(
        axis=1,
        min_count=len(assets),
    )
    gross_return = risky_asset_return + lagged_cash * cash_return
    transaction_cost_return = turnover.shift(1) * transaction_cost
    net_return = gross_return - transaction_cost_return

    return {
        "target_weights": target_weights,
        "target_cash": cash_weights,
        "pre_rebalance_weights": pre_rebalance_weights,
        "pre_rebalance_cash": pre_rebalance_cash,
        "turnover": turnover,
        "gross_return": gross_return,
        "transaction_cost": transaction_cost_return,
        "net_return": net_return,
    }


def calculate_performance_metrics(returns):
    """Calculate the notebook's monthly, zero-risk-free-rate performance metrics."""
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "Months": np.nan,
            "CAGR": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Calmar Ratio": np.nan,
        }

    n_months = len(returns)
    n_years = n_months / 12.0
    wealth = (1.0 + returns).cumprod()
    cagr = wealth.iloc[-1] ** (1.0 / n_years) - 1.0
    annualized_volatility = returns.std(ddof=1) * np.sqrt(12)

    if returns.std(ddof=1) > 0:
        sharpe_ratio = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(12)
        )
    else:
        sharpe_ratio = np.nan

    drawdown = wealth / wealth.cummax() - 1.0
    maximum_drawdown = drawdown.min()
    calmar_ratio = (
        cagr / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else np.nan
    )

    return {
        "Months": n_months,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
        "Calmar Ratio": calmar_ratio,
    }


def build_period_performance_table(return_frame):
    """Build the legacy strategy-by-metric table for a supplied return period."""
    rows = []

    for strategy in return_frame.columns:
        metrics = calculate_performance_metrics(return_frame[strategy])
        metrics["Strategy"] = strategy
        rows.append(metrics)

    table = pd.DataFrame(rows).set_index("Strategy")
    return table[
        [
            "Months",
            "CAGR",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
        ]
    ]


def slice_oos_returns(strategy_returns, oos_start):
    """Apply the notebook's inclusive calendar-index OOS slice."""
    return strategy_returns.loc[oos_start:].copy()


def summarize_turnover(backtests):
    """Return the legacy full-sample turnover summary.

    Initial portfolio establishment is reported but excluded from average,
    annualized, and maximum ongoing turnover.
    """
    turnover_rows = []

    for name, result in backtests.items():
        applied_turnover = result["turnover"].shift(1).reindex(
            result["net_return"].dropna().index
        )
        initial_turnover = applied_turnover.iloc[0]
        ongoing_turnover = applied_turnover.iloc[1:]

        turnover_rows.append(
            {
                "Strategy": name,
                "Initial Turnover": initial_turnover,
                "Average Monthly Turnover": ongoing_turnover.mean(),
                "Annualized Turnover": ongoing_turnover.mean() * 12,
                "Maximum Monthly Turnover": ongoing_turnover.max(),
            }
        )

    return pd.DataFrame(turnover_rows).set_index("Strategy")
