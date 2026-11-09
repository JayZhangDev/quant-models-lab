"""Tests for the Kalman hedge-ratio filter, the rolling OLS baseline and the backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qml.pairs.backtest import (
    BacktestConfig,
    choose_delta,
    engle_granger_pvalue,
    run_backtest,
)
from qml.pairs.kalman import KalmanHedge, fit_obs_var, rolling_ols, run_filter


def _linear_pair(beta: np.ndarray, alpha: float, noise: float, seed: int) -> tuple:
    """Synthetic data y = beta_t * x + alpha + noise, with x moving around 5."""
    rng = np.random.default_rng(seed)
    x = 5.0 + rng.normal(0.0, 0.5, beta.size)
    y = beta * x + alpha + rng.normal(0.0, noise, beta.size)
    return pd.Series(x), pd.Series(y)


def test_converges_to_a_constant_beta() -> None:
    x, y = _linear_pair(np.full(1000, 1.5), alpha=0.3, noise=0.05, seed=1)
    out = run_filter(x, y, delta=1e-6, obs_var=0.05**2)
    assert out["beta"].iloc[-1] == pytest.approx(1.5, abs=0.02)
    assert out["alpha"].iloc[-1] == pytest.approx(0.3, abs=0.1)


def test_adapts_to_a_step_change_faster_than_rolling_ols() -> None:
    beta = np.concatenate([np.full(300, 1.0), np.full(300, 1.5)])
    x, y = _linear_pair(beta, alpha=0.3, noise=0.05, seed=2)

    kalman = run_filter(x, y, delta=1e-3, obs_var=0.05**2)["beta"]
    ols = rolling_ols(x, y, window=60)["beta"]

    # Average distance from the new true beta during the 60 days after the break.
    after = slice(300, 360)
    kalman_error = np.abs(kalman.iloc[after] - 1.5).mean()
    ols_error = np.abs(ols.iloc[after] - 1.5).mean()
    assert kalman_error < 0.5 * ols_error


def test_z_scores_are_standard_normal_when_the_model_is_right() -> None:
    # Simulate data from the filter's own model, then filter with the true parameters.
    rng = np.random.default_rng(3)
    n, delta, obs_var = 4000, 1e-4, 0.01
    state_noise = np.sqrt(delta / (1.0 - delta))
    beta = 1.0 + np.cumsum(rng.normal(0.0, state_noise, n))
    alpha = 0.5 + np.cumsum(rng.normal(0.0, state_noise, n))
    x = 5.0 + rng.normal(0.0, 0.5, n)
    y = beta * x + alpha + rng.normal(0.0, np.sqrt(obs_var), n)

    z = run_filter(x, y, delta, obs_var)["z"].to_numpy()[200:]  # skip the warm-up

    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.0, abs=0.07)
    assert stats.kstest(z, "norm").pvalue > 0.01


def test_update_returns_the_error_made_before_updating() -> None:
    kf = KalmanHedge(delta=1e-4, obs_var=1e-3, initial_state=(2.0, 1.0))
    step = kf.update(x=3.0, y=10.0)
    assert step.error == pytest.approx(10.0 - (2.0 * 3.0 + 1.0))
    assert step.error_std > 0.0
    assert step.z == pytest.approx(step.error / step.error_std)


def test_larger_delta_means_a_more_responsive_filter() -> None:
    x, y = _linear_pair(np.full(500, 1.0), alpha=0.0, noise=0.1, seed=4)
    slow = run_filter(x, y, delta=1e-7, obs_var=0.01)["beta"].iloc[100:]
    fast = run_filter(x, y, delta=1e-2, obs_var=0.01)["beta"].iloc[100:]
    assert fast.diff().abs().mean() > 10 * slow.diff().abs().mean()


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        KalmanHedge(delta=1.0)
    with pytest.raises(ValueError):
        KalmanHedge(obs_var=0.0)


def test_fit_obs_var_recovers_the_true_noise_level() -> None:
    x, y = _linear_pair(np.full(1500, 1.2), alpha=0.5, noise=0.05, seed=5)
    fitted = fit_obs_var(x, y, delta=1e-7)
    assert fitted == pytest.approx(0.05**2, rel=0.25)


def test_rolling_ols_recovers_beta_and_uses_only_past_data() -> None:
    x, y = _linear_pair(np.full(400, 0.8), alpha=1.0, noise=0.02, seed=6)
    out = rolling_ols(x, y, window=60)
    assert out["beta"].iloc[:60].isna().all()  # needs 60 past days before the first value
    assert out["beta"].iloc[-1] == pytest.approx(0.8, abs=0.03)

    # Changing today's prices must not change today's beta (it is built from the past).
    y_shocked = y.copy()
    y_shocked.iloc[-1] += 5.0
    assert rolling_ols(x, y_shocked, window=60)["beta"].iloc[-1] == out["beta"].iloc[-1]


def _signals(z: list[float], beta: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"z": z, "beta": beta})


def test_backtest_enters_on_wide_spread_and_exits_near_zero() -> None:
    signals = _signals([0.0, 2.5, 1.0, 0.3, 0.0, -2.5, -1.0, -0.2])
    flat = pd.Series(np.zeros(8))
    result = run_backtest(signals, flat, flat, BacktestConfig(cost_bps=0.0))
    assert result.positions.tolist() == [0, -1, -1, 0, 0, 1, 1, 0]
    assert result.n_trades == 2


def test_backtest_profits_when_the_spread_converges() -> None:
    # z > 2 on day 1 -> short y / long x at that close. On day 2, y falls 1% and x is flat.
    signals = _signals([0.0, 2.5, 0.0, 0.0])
    ret_y = pd.Series([0.0, 0.0, -0.01, 0.0])
    ret_x = pd.Series(np.zeros(4))
    result = run_backtest(signals, ret_x, ret_y, BacktestConfig(cost_bps=0.0))
    # Short $1 of y gains 1%; divided by gross capital of 1 + beta = 2.
    assert result.returns.iloc[2] == pytest.approx(0.005)
    assert result.returns.sum() == pytest.approx(0.005)


def test_backtest_does_not_use_same_day_returns() -> None:
    # A big move on the signal day itself must earn nothing: we only trade at the close.
    signals = _signals([0.0, 2.5, 2.5, 0.0])
    ret_y = pd.Series([0.0, -0.05, 0.0, 0.0])
    result = run_backtest(signals, pd.Series(np.zeros(4)), ret_y, BacktestConfig(cost_bps=0.0))
    assert result.returns.iloc[1] == 0.0


def test_costs_are_charged_on_entry_and_exit() -> None:
    signals = _signals([0.0, 2.5, 0.0, 0.0])
    flat = pd.Series(np.zeros(4))
    result = run_backtest(signals, flat, flat, BacktestConfig(cost_bps=5.0))
    assert result.returns.tolist() == pytest.approx([0.0, -0.0005, -0.0005, 0.0])


def test_no_signal_means_no_trades_and_no_returns() -> None:
    signals = _signals([0.1, -0.4, 1.9, np.nan, -1.9])
    flat = pd.Series(np.full(5, 0.01))
    result = run_backtest(signals, flat, flat)
    assert result.n_trades == 0
    assert (result.returns == 0.0).all()
    assert result.metrics()["sharpe"] == 0.0


def test_engle_granger_separates_cointegrated_from_unrelated_series() -> None:
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.normal(0.0, 1.0, 1500))
    cointegrated = 0.7 * x + rng.normal(0.0, 1.0, 1500)  # tied to x
    unrelated = np.cumsum(rng.normal(0.0, 1.0, 1500))  # its own random walk
    assert engle_granger_pvalue(cointegrated, x) < 0.01
    assert engle_granger_pvalue(unrelated, x) > 0.05


def test_choose_delta_ignores_lucky_low_trade_counts() -> None:
    sweep = pd.DataFrame(
        {
            "delta": [1e-9, 1e-7, 1e-6],
            "obs_var": [1e-3, 1e-4, 1e-5],
            "sharpe": [2.0, 0.3, 0.5],  # the 2.0 comes from only 3 trades
            "n_trades": [3, 40, 80],
        }
    )
    assert choose_delta(sweep, min_trades=20)["delta"] == 1e-6
    # If nothing trades often enough, fall back to the most active delta.
    assert choose_delta(sweep, min_trades=500)["delta"] == 1e-6
