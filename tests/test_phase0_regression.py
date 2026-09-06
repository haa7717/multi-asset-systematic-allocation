from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from multi_asset_allocation import phase0_legacy as legacy


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "phase0" / "phase0_adjusted_close_snapshot.csv"
RESULTS = ROOT / "results"
TICKERS = ["SPY", "EFA", "EEM", "IEF", "LQD", "GLD", "VNQ", "DBC"]
ANALYSIS_END = "2026-06-30"
OOS_START = "2019-01-01"
N_ASSETS = len(TICKERS)
TRANSACTION_COST = 0.001
CASH_RETURN = 0.0


def _literal_reference_pipeline():
    """Independent, literal transcription of the relevant notebook calculations."""
    prices = pd.read_csv(SNAPSHOT, index_col="Date", parse_dates=True)
    prices = prices.reindex(columns=TICKERS)
    analysis_prices = prices.loc[:ANALYSIS_END].copy()

    daily_returns = analysis_prices.pct_change(fill_method=None)
    monthly_prices = analysis_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change(fill_method=None)

    trend_12m_return = monthly_prices.pct_change(
        periods=12,
        fill_method=None,
    )
    trend_signal = (trend_12m_return > 0).astype(float)
    trend_signal = trend_signal.where(trend_12m_return.notna())

    daily_vol_63 = (
        daily_returns
        .rolling(window=63, min_periods=63)
        .std()
        * np.sqrt(252)
    )
    monthly_vol = daily_vol_63.resample("ME").last()

    equal_weight = pd.DataFrame(
        1.0 / N_ASSETS,
        index=monthly_prices.index,
        columns=TICKERS,
    )
    inverse_vol_raw = 1.0 / monthly_vol
    inverse_vol_denominator = inverse_vol_raw.sum(
        axis=1,
        min_count=N_ASSETS,
    )
    inverse_vol_weight = inverse_vol_raw.div(
        inverse_vol_denominator,
        axis=0,
    )
    trend_equal_weight = trend_signal * (1.0 / N_ASSETS)
    trend_equal_cash = 1.0 - trend_equal_weight.sum(
        axis=1,
        min_count=N_ASSETS,
    )
    trend_inverse_vol_weight = trend_signal * inverse_vol_weight
    trend_inverse_vol_cash = 1.0 - trend_inverse_vol_weight.sum(
        axis=1,
        min_count=N_ASSETS,
    )

    trend_equal_cash = trend_equal_cash.clip(lower=0.0, upper=1.0)
    trend_inverse_vol_cash = trend_inverse_vol_cash.clip(
        lower=0.0,
        upper=1.0,
    )
    equal_cash = pd.Series(0.0, index=equal_weight.index, name="Cash")
    inverse_vol_cash = pd.Series(
        0.0,
        index=inverse_vol_weight.index,
        name="Cash",
    )

    portfolio_targets = {
        "Equal Weight": equal_weight,
        "Inverse Volatility": inverse_vol_weight,
        "Trend + Equal Weight": trend_equal_weight,
        "Trend + Inverse Volatility": trend_inverse_vol_weight,
    }
    portfolio_cash = {
        "Equal Weight": equal_cash,
        "Inverse Volatility": inverse_vol_cash,
        "Trend + Equal Weight": trend_equal_cash,
        "Trend + Inverse Volatility": trend_inverse_vol_cash,
    }

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
    common_index = monthly_prices.loc[common_start:common_end].index

    for name in portfolio_targets:
        portfolio_targets[name] = portfolio_targets[name].reindex(common_index)
        portfolio_cash[name] = portfolio_cash[name].reindex(common_index)

    def run_monthly_backtest(
        target_weights,
        cash_weights,
        monthly_asset_returns,
        transaction_cost,
    ):
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
                    + previous_cash * (1.0 + CASH_RETURN)
                )
                pre_risky = (
                    previous_target
                    * (1.0 + asset_returns)
                    / gross_portfolio_value
                )
                pre_cash = (
                    previous_cash
                    * (1.0 + CASH_RETURN)
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
        gross_return = risky_asset_return + lagged_cash * CASH_RETURN
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

    backtests = {}
    for name in portfolio_targets:
        backtests[name] = run_monthly_backtest(
            target_weights=portfolio_targets[name],
            cash_weights=portfolio_cash[name],
            monthly_asset_returns=monthly_returns,
            transaction_cost=TRANSACTION_COST,
        )

    def calculate_performance_metrics(returns):
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
        annualized_vol = returns.std(ddof=1) * np.sqrt(12)
        if returns.std(ddof=1) > 0:
            sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(12)
        else:
            sharpe = np.nan
        drawdown = wealth / wealth.cummax() - 1.0
        max_drawdown = drawdown.min()
        if max_drawdown < 0:
            calmar = cagr / abs(max_drawdown)
        else:
            calmar = np.nan
        return {
            "Months": n_months,
            "CAGR": cagr,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe,
            "Maximum Drawdown": max_drawdown,
            "Calmar Ratio": calmar,
        }

    strategy_net_returns = pd.DataFrame(
        {name: result["net_return"] for name, result in backtests.items()}
    )
    first_holding_date = strategy_net_returns.dropna(how="all").index[0]
    last_holding_date = strategy_net_returns.dropna(how="all").index[-1]
    spy_returns = monthly_returns.loc[
        first_holding_date:last_holding_date,
        "SPY",
    ].copy()
    spy_net_returns = spy_returns.copy()
    spy_net_returns.iloc[0] = spy_net_returns.iloc[0] - TRANSACTION_COST
    strategy_net_returns["SPY Buy & Hold"] = spy_net_returns

    def build_period_performance_table(return_frame):
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
    turnover_table = pd.DataFrame(turnover_rows).set_index("Strategy")

    oos_returns = strategy_net_returns.loc[OOS_START:].copy()
    oos_performance = build_period_performance_table(oos_returns)
    final_summary = oos_performance[
        [
            "CAGR",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
        ]
    ].copy()
    final_summary["Annualized Turnover"] = np.nan
    for strategy in turnover_table.index:
        final_summary.loc[strategy, "Annualized Turnover"] = turnover_table.loc[
            strategy,
            "Annualized Turnover",
        ]

    ablation_summary = pd.DataFrame(
        {
            "OOS Sharpe": {
                "Equal Weight": oos_performance.loc[
                    "Equal Weight", "Sharpe Ratio"
                ],
                "Inverse Volatility": oos_performance.loc[
                    "Inverse Volatility", "Sharpe Ratio"
                ],
                "Trend + Equal Weight": oos_performance.loc[
                    "Trend + Equal Weight", "Sharpe Ratio"
                ],
                "Trend + Inverse Volatility": oos_performance.loc[
                    "Trend + Inverse Volatility", "Sharpe Ratio"
                ],
            },
            "OOS Maximum Drawdown": {
                "Equal Weight": oos_performance.loc[
                    "Equal Weight", "Maximum Drawdown"
                ],
                "Inverse Volatility": oos_performance.loc[
                    "Inverse Volatility", "Maximum Drawdown"
                ],
                "Trend + Equal Weight": oos_performance.loc[
                    "Trend + Equal Weight", "Maximum Drawdown"
                ],
                "Trend + Inverse Volatility": oos_performance.loc[
                    "Trend + Inverse Volatility", "Maximum Drawdown"
                ],
            },
        }
    )

    sensitivity_signals = {}
    for lookback in [9, 12, 15]:
        trailing_return = monthly_prices.pct_change(
            periods=lookback,
            fill_method=None,
        )
        signal = (trailing_return > 0).astype(float)
        signal = signal.where(trailing_return.notna())
        sensitivity_signals[lookback] = signal

    sensitivity_first_dates = []
    for lookback, signal in sensitivity_signals.items():
        valid_mask = (
            signal.notna().all(axis=1)
            & inverse_vol_weight.notna().all(axis=1)
        )
        sensitivity_first_dates.append(signal.index[valid_mask][0])
    sensitivity_start = max(sensitivity_first_dates)
    sensitivity_index = monthly_prices.loc[sensitivity_start:common_end].index
    trend_sensitivity_backtests = {}
    for lookback in [9, 12, 15]:
        signal = sensitivity_signals[lookback].reindex(sensitivity_index)
        tew_weights = signal * (1.0 / N_ASSETS)
        tew_cash = (
            1.0 - tew_weights.sum(axis=1, min_count=N_ASSETS)
        ).clip(lower=0.0, upper=1.0)
        trend_sensitivity_backtests[(lookback, "Trend + Equal Weight")] = (
            run_monthly_backtest(
                tew_weights,
                tew_cash,
                monthly_returns,
                TRANSACTION_COST,
            )
        )
        base_inverse_volatility = inverse_vol_weight.reindex(sensitivity_index)
        tiv_weights = signal * base_inverse_volatility
        tiv_cash = (
            1.0 - tiv_weights.sum(axis=1, min_count=N_ASSETS)
        ).clip(lower=0.0, upper=1.0)
        trend_sensitivity_backtests[(lookback, "Trend + Inverse Volatility")] = (
            run_monthly_backtest(
                tiv_weights,
                tiv_cash,
                monthly_returns,
                TRANSACTION_COST,
            )
        )

    lookback_rows = []
    for (lookback, strategy_name), result in trend_sensitivity_backtests.items():
        metrics = calculate_performance_metrics(
            result["net_return"].loc[OOS_START:]
        )
        lookback_rows.append(
            {
                "Trend Lookback": lookback,
                "Strategy": strategy_name,
                "CAGR": metrics["CAGR"],
                "Annualized Volatility": metrics["Annualized Volatility"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Maximum Drawdown": metrics["Maximum Drawdown"],
            }
        )
    lookback_sensitivity_table = pd.DataFrame(lookback_rows).sort_values(
        ["Strategy", "Trend Lookback"]
    )

    cost_sensitivity_rows = []
    for strategy_name, result in backtests.items():
        gross_return = result["gross_return"]
        applied_turnover = result["turnover"].shift(1)
        for cost_name, cost_rate in {
            "5 bps": 0.0005,
            "10 bps": 0.0010,
            "25 bps": 0.0025,
        }.items():
            alternative_net_return = gross_return - applied_turnover * cost_rate
            metrics = calculate_performance_metrics(
                alternative_net_return.loc[OOS_START:]
            )
            cost_sensitivity_rows.append(
                {
                    "Strategy": strategy_name,
                    "Transaction Cost": cost_name,
                    "Cost Rate": cost_rate,
                    "OOS CAGR": metrics["CAGR"],
                    "OOS Sharpe": metrics["Sharpe Ratio"],
                    "OOS Maximum Drawdown": metrics["Maximum Drawdown"],
                }
            )
    cost_sensitivity_table = pd.DataFrame(cost_sensitivity_rows)

    return {
        "prices": analysis_prices,
        "daily_returns": daily_returns,
        "monthly_prices": monthly_prices,
        "monthly_returns": monthly_returns,
        "trend_signal": trend_signal,
        "monthly_volatility": monthly_vol,
        "portfolio_targets": portfolio_targets,
        "portfolio_cash": portfolio_cash,
        "common_start": common_start,
        "common_end": common_end,
        "common_index": common_index,
        "backtests": backtests,
        "strategy_returns": strategy_net_returns,
        "oos_returns": oos_returns,
        "oos_performance": oos_performance,
        "turnover_table": turnover_table,
        "final_summary": final_summary,
        "ablation_summary": ablation_summary,
        "lookback_sensitivity_table": lookback_sensitivity_table,
        "cost_sensitivity_table": cost_sensitivity_table,
        "metrics_function": calculate_performance_metrics,
    }


def _module_pipeline():
    prices = pd.read_csv(SNAPSHOT, index_col="Date", parse_dates=True)
    analysis_prices = legacy.prepare_analysis_prices(
        prices,
        TICKERS,
        ANALYSIS_END,
    )
    daily_returns = legacy.calculate_daily_returns(analysis_prices)
    monthly_prices = legacy.construct_monthly_prices(analysis_prices)
    monthly_returns = legacy.calculate_monthly_returns(monthly_prices)
    trend_signal = legacy.calculate_trend_signal(monthly_prices, lookback=12)
    monthly_volatility = legacy.calculate_monthly_realized_volatility(
        daily_returns,
        window=63,
    )

    equal_weight = legacy.equal_weight_targets(monthly_prices.index, TICKERS)
    inverse_volatility = legacy.inverse_volatility_targets(
        monthly_volatility,
        N_ASSETS,
    )
    trend_equal_weight = legacy.trend_equal_weight_targets(
        trend_signal,
        N_ASSETS,
    )
    trend_inverse_volatility = legacy.trend_inverse_volatility_targets(
        trend_signal,
        inverse_volatility,
    )
    portfolio_targets = {
        "Equal Weight": equal_weight,
        "Inverse Volatility": inverse_volatility,
        "Trend + Equal Weight": trend_equal_weight,
        "Trend + Inverse Volatility": trend_inverse_volatility,
    }
    portfolio_cash = {
        "Equal Weight": legacy.zero_cash_weights(equal_weight.index),
        "Inverse Volatility": legacy.zero_cash_weights(
            inverse_volatility.index
        ),
        "Trend + Equal Weight": legacy.cash_weights(trend_equal_weight),
        "Trend + Inverse Volatility": legacy.cash_weights(
            trend_inverse_volatility
        ),
    }
    common_index, common_start, common_end = legacy.common_formation_index(
        portfolio_targets,
        portfolio_cash,
        monthly_prices,
    )
    for name in portfolio_targets:
        portfolio_targets[name] = portfolio_targets[name].reindex(common_index)
        portfolio_cash[name] = portfolio_cash[name].reindex(common_index)

    backtests = {
        name: legacy.run_monthly_backtest(
            target_weights=portfolio_targets[name],
            cash_weights=portfolio_cash[name],
            monthly_asset_returns=monthly_returns,
            transaction_cost=TRANSACTION_COST,
            cash_return=CASH_RETURN,
        )
        for name in portfolio_targets
    }
    strategy_returns = pd.DataFrame(
        {name: result["net_return"] for name, result in backtests.items()}
    )
    first_holding_date = strategy_returns.dropna(how="all").index[0]
    last_holding_date = strategy_returns.dropna(how="all").index[-1]
    spy_returns = monthly_returns.loc[
        first_holding_date:last_holding_date,
        "SPY",
    ].copy()
    spy_returns.iloc[0] = spy_returns.iloc[0] - TRANSACTION_COST
    strategy_returns["SPY Buy & Hold"] = spy_returns

    turnover_table = legacy.summarize_turnover(backtests)
    oos_returns = legacy.slice_oos_returns(strategy_returns, OOS_START)
    oos_performance = legacy.build_period_performance_table(oos_returns)
    final_summary = oos_performance[
        [
            "CAGR",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
        ]
    ].copy()
    final_summary["Annualized Turnover"] = np.nan
    for strategy in turnover_table.index:
        final_summary.loc[strategy, "Annualized Turnover"] = turnover_table.loc[
            strategy,
            "Annualized Turnover",
        ]
    ablation_summary = pd.DataFrame(
        {
            "OOS Sharpe": {
                strategy: oos_performance.loc[strategy, "Sharpe Ratio"]
                for strategy in portfolio_targets
            },
            "OOS Maximum Drawdown": {
                strategy: oos_performance.loc[
                    strategy,
                    "Maximum Drawdown",
                ]
                for strategy in portfolio_targets
            },
        }
    )

    sensitivity_signals = {
        lookback: legacy.calculate_trend_signal(monthly_prices, lookback)
        for lookback in [9, 12, 15]
    }
    sensitivity_start = max(
        signal.index[
            signal.notna().all(axis=1)
            & inverse_volatility.notna().all(axis=1)
        ][0]
        for signal in sensitivity_signals.values()
    )
    sensitivity_index = monthly_prices.loc[
        sensitivity_start:common_end
    ].index
    sensitivity_backtests = {}
    for lookback, signal in sensitivity_signals.items():
        signal = signal.reindex(sensitivity_index)
        tew_weights = legacy.trend_equal_weight_targets(signal, N_ASSETS)
        sensitivity_backtests[(lookback, "Trend + Equal Weight")] = (
            legacy.run_monthly_backtest(
                tew_weights,
                legacy.cash_weights(tew_weights),
                monthly_returns,
                TRANSACTION_COST,
                CASH_RETURN,
            )
        )
        tiv_weights = legacy.trend_inverse_volatility_targets(
            signal,
            inverse_volatility.reindex(sensitivity_index),
        )
        sensitivity_backtests[(lookback, "Trend + Inverse Volatility")] = (
            legacy.run_monthly_backtest(
                tiv_weights,
                legacy.cash_weights(tiv_weights),
                monthly_returns,
                TRANSACTION_COST,
                CASH_RETURN,
            )
        )
    lookback_rows = []
    for (lookback, strategy), result in sensitivity_backtests.items():
        metrics = legacy.calculate_performance_metrics(
            legacy.slice_oos_returns(
                result["net_return"].to_frame("return"),
                OOS_START,
            )["return"]
        )
        lookback_rows.append(
            {
                "Trend Lookback": lookback,
                "Strategy": strategy,
                "CAGR": metrics["CAGR"],
                "Annualized Volatility": metrics["Annualized Volatility"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Maximum Drawdown": metrics["Maximum Drawdown"],
            }
        )
    lookback_sensitivity_table = pd.DataFrame(lookback_rows).sort_values(
        ["Strategy", "Trend Lookback"]
    )

    cost_rows = []
    for strategy, result in backtests.items():
        for cost_name, cost_rate in {
            "5 bps": 0.0005,
            "10 bps": 0.0010,
            "25 bps": 0.0025,
        }.items():
            alternative_returns = (
                result["gross_return"]
                - result["turnover"].shift(1) * cost_rate
            )
            metrics = legacy.calculate_performance_metrics(
                alternative_returns.loc[OOS_START:]
            )
            cost_rows.append(
                {
                    "Strategy": strategy,
                    "Transaction Cost": cost_name,
                    "Cost Rate": cost_rate,
                    "OOS CAGR": metrics["CAGR"],
                    "OOS Sharpe": metrics["Sharpe Ratio"],
                    "OOS Maximum Drawdown": metrics["Maximum Drawdown"],
                }
            )
    cost_sensitivity_table = pd.DataFrame(cost_rows)

    return {
        "prices": analysis_prices,
        "daily_returns": daily_returns,
        "monthly_prices": monthly_prices,
        "monthly_returns": monthly_returns,
        "trend_signal": trend_signal,
        "monthly_volatility": monthly_volatility,
        "portfolio_targets": portfolio_targets,
        "portfolio_cash": portfolio_cash,
        "common_start": common_start,
        "common_end": common_end,
        "common_index": common_index,
        "backtests": backtests,
        "strategy_returns": strategy_returns,
        "oos_returns": oos_returns,
        "oos_performance": oos_performance,
        "turnover_table": turnover_table,
        "final_summary": final_summary,
        "ablation_summary": ablation_summary,
        "lookback_sensitivity_table": lookback_sensitivity_table,
        "cost_sensitivity_table": cost_sensitivity_table,
    }


@pytest.fixture(scope="session")
def reference():
    return _literal_reference_pipeline()


@pytest.fixture(scope="session")
def extracted():
    return _module_pipeline()


def _assert_frame_parity(actual, expected):
    pdt.assert_frame_equal(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
        check_exact=False,
    )


def _assert_series_parity(actual, expected):
    pdt.assert_series_equal(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
        check_exact=False,
    )


def test_same_input_shapes_dates_and_formation_window(reference, extracted):
    assert extracted["prices"].shape == (4903, 8)
    assert extracted["prices"].columns.tolist() == TICKERS
    assert extracted["monthly_prices"].shape == (234, 8)
    assert extracted["trend_signal"].dropna().index[0] == pd.Timestamp(
        "2008-01-31"
    )
    assert extracted["monthly_volatility"].dropna().index[0] == pd.Timestamp(
        "2007-04-30"
    )
    assert extracted["common_start"] == pd.Timestamp("2008-01-31")
    assert len(extracted["common_index"]) == 222
    assert len(extracted["strategy_returns"].dropna(how="all")) == 221
    assert len(extracted["oos_returns"].dropna(how="all")) == 90
    assert extracted["oos_returns"].dropna(how="all").index[0] == pd.Timestamp(
        "2019-01-31"
    )
    assert extracted["oos_returns"].dropna(how="all").index[-1] == pd.Timestamp(
        "2026-06-30"
    )
    assert extracted["common_start"] == reference["common_start"]
    assert extracted["common_end"] == reference["common_end"]


def test_same_input_intermediate_targets_and_cash_parity(reference, extracted):
    for key in [
        "daily_returns",
        "monthly_prices",
        "monthly_returns",
        "trend_signal",
        "monthly_volatility",
    ]:
        _assert_frame_parity(extracted[key], reference[key])

    for strategy in reference["portfolio_targets"]:
        _assert_frame_parity(
            extracted["portfolio_targets"][strategy],
            reference["portfolio_targets"][strategy],
        )
        _assert_series_parity(
            extracted["portfolio_cash"][strategy],
            reference["portfolio_cash"][strategy],
        )
        total_weight = (
            extracted["portfolio_targets"][strategy].sum(axis=1)
            + extracted["portfolio_cash"][strategy]
        )
        np.testing.assert_allclose(total_weight, 1.0, rtol=1e-12, atol=1e-12)


def test_same_input_backtest_returns_and_metrics_parity(reference, extracted):
    for strategy in reference["backtests"]:
        for key in [
            "pre_rebalance_weights",
            "pre_rebalance_cash",
            "turnover",
            "gross_return",
            "transaction_cost",
            "net_return",
        ]:
            actual = extracted["backtests"][strategy][key]
            expected = reference["backtests"][strategy][key]
            if isinstance(actual, pd.DataFrame):
                _assert_frame_parity(actual, expected)
            else:
                _assert_series_parity(actual, expected)

        actual_metrics = legacy.calculate_performance_metrics(
            extracted["backtests"][strategy]["net_return"]
        )
        expected_metrics = reference["metrics_function"](
            reference["backtests"][strategy]["net_return"]
        )
        np.testing.assert_allclose(
            list(actual_metrics.values()),
            list(expected_metrics.values()),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_same_input_final_tables_are_near_machine_precision(reference, extracted):
    tables = [
        "oos_performance",
        "turnover_table",
        "final_summary",
        "ablation_summary",
        "cost_sensitivity_table",
        "lookback_sensitivity_table",
    ]
    maximum = 0.0
    for key in tables:
        actual = extracted[key].sort_index(axis=0).sort_index(axis=1)
        expected = reference[key].sort_index(axis=0).sort_index(axis=1)
        _assert_frame_parity(actual, expected)
        numeric = (actual.select_dtypes(include=[np.number]) - expected.select_dtypes(include=[np.number])).abs()
        maximum = max(maximum, float(numeric.max().max()))
    print(f"same-input maximum absolute difference: {maximum:.17g}")


def _historical_table(filename):
    if filename == "phase0_oos_summary.csv":
        return pd.read_csv(RESULTS / filename, index_col="Strategy")
    if filename == "phase0_turnover_summary.csv":
        return pd.read_csv(RESULTS / filename, index_col="Strategy")
    if filename == "phase0_ablation_summary.csv":
        return pd.read_csv(RESULTS / filename, index_col=0)
    if filename == "phase0_transaction_cost_sensitivity.csv":
        return pd.read_csv(RESULTS / filename).set_index(
            ["Strategy", "Transaction Cost"]
        )
    return pd.read_csv(RESULTS / filename).set_index(
        ["Trend Lookback", "Strategy"]
    )


@pytest.mark.parametrize(
    ("filename", "extracted_key", "tolerance"),
    [
        ("phase0_oos_summary.csv", "final_summary", 5e-6),
        ("phase0_turnover_summary.csv", "turnover_table", 2e-5),
        ("phase0_ablation_summary.csv", "ablation_summary", 5e-6),
        (
            "phase0_transaction_cost_sensitivity.csv",
            "cost_sensitivity_table",
            5e-6,
        ),
        (
            "phase0_trend_lookback_sensitivity.csv",
            "lookback_sensitivity_table",
            5e-6,
        ),
    ],
)
def test_historical_artifacts_within_documented_tolerance(
    extracted,
    filename,
    extracted_key,
    tolerance,
):
    actual = extracted[extracted_key]
    if filename == "phase0_transaction_cost_sensitivity.csv":
        actual = actual.set_index(["Strategy", "Transaction Cost"])
    elif filename == "phase0_trend_lookback_sensitivity.csv":
        actual = actual.set_index(["Trend Lookback", "Strategy"])
    expected = _historical_table(filename)
    actual = actual.reindex(index=expected.index, columns=expected.columns)
    difference = (actual - expected).abs()
    maximum = float(difference.max().max())
    print(f"{filename} historical maximum absolute difference: {maximum:.17g}")
    assert maximum <= tolerance


def test_historical_cross_table_invariants(extracted):
    final_summary = extracted["final_summary"]
    ablation = extracted["ablation_summary"]
    cost = extracted["cost_sensitivity_table"]
    lookback = extracted["lookback_sensitivity_table"]
    turnover = extracted["turnover_table"]

    ten_bps = cost.loc[cost["Transaction Cost"] == "10 bps"].set_index(
        "Strategy"
    )
    systematic = turnover.index
    np.testing.assert_allclose(
        ten_bps.loc[systematic, "OOS CAGR"],
        final_summary.loc[systematic, "CAGR"],
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        ten_bps.loc[systematic, "OOS Sharpe"],
        final_summary.loc[systematic, "Sharpe Ratio"],
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        ten_bps.loc[systematic, "OOS Maximum Drawdown"],
        final_summary.loc[systematic, "Maximum Drawdown"],
        rtol=0.0,
        atol=5e-6,
    )

    trend_rows = lookback.loc[lookback["Trend Lookback"] == 12].set_index(
        "Strategy"
    )
    trend_strategies = trend_rows.index
    for lookback_column, summary_column in [
        ("CAGR", "CAGR"),
        ("Annualized Volatility", "Annualized Volatility"),
        ("Sharpe Ratio", "Sharpe Ratio"),
        ("Maximum Drawdown", "Maximum Drawdown"),
    ]:
        np.testing.assert_allclose(
            trend_rows[lookback_column],
            final_summary.loc[trend_strategies, summary_column],
            rtol=0.0,
            atol=5e-6,
        )
    np.testing.assert_allclose(
        ablation.loc[systematic, "OOS Sharpe"],
        final_summary.loc[systematic, "Sharpe Ratio"],
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        ablation.loc[systematic, "OOS Maximum Drawdown"],
        final_summary.loc[systematic, "Maximum Drawdown"],
        rtol=0.0,
        atol=5e-6,
    )
    for strategy, result in extracted["backtests"].items():
        applied_turnover = result["turnover"].shift(1).reindex(
            result["net_return"].dropna().index
        )
        expected_annualized_turnover = applied_turnover.iloc[1:].mean() * 12
        np.testing.assert_allclose(
            turnover.loc[strategy, "Annualized Turnover"],
            expected_annualized_turnover,
            rtol=1e-12,
            atol=1e-12,
        )
