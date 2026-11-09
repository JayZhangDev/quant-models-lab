"""Backtest of a z-score pairs-trading rule, with transaction costs.

The rule: when the spread between two related assets is unusually wide (``|z| > 2``),
bet that it will narrow; close the bet once it is back near normal (``|z| < 0.5``).

* ``z > +2``: ``y`` looks expensive relative to ``x`` -> short ``y``, long ``x``.
* ``z < -2``: ``y`` looks cheap -> long ``y``, short ``x``.

Signals use prices up to today's close, trades happen at today's close, and profit
starts accruing tomorrow - so the backtest never looks into the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.stattools import coint

from qml.common.metrics import annualized_return, max_drawdown, sharpe_ratio
from qml.pairs.kalman import fit_obs_var, run_filter

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BacktestConfig:
    """Trading rule and cost settings.

    Attributes
    ----------
    entry_z : float
        Open a position when ``|z|`` rises above this.
    exit_z : float
        Close it when ``|z|`` falls back below this.
    cost_bps : float
        Transaction cost in basis points (1 bp = 0.01%) of every dollar traded.
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    cost_bps: float = 5.0


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of one backtest.

    Attributes
    ----------
    returns : pd.Series
        Daily strategy returns after costs, as a fraction of gross capital.
    positions : pd.Series
        Position in the spread held *after* each day's close: +1, 0 or -1.
    n_trades : int
        Number of positions opened.
    """

    returns: pd.Series
    positions: pd.Series
    n_trades: int

    def metrics(self) -> dict[str, float]:
        """Sharpe ratio, annual return, max drawdown and trade count."""
        r = self.returns.to_numpy()
        return {
            "sharpe": sharpe_ratio(r),
            "annual_return": annualized_return(r),
            "max_drawdown": max_drawdown(r),
            "n_trades": float(self.n_trades),
        }


def engle_granger_pvalue(y: ArrayLike, x: ArrayLike) -> float:
    """P-value of the Engle-Granger cointegration test.

    Parameters
    ----------
    y, x : array-like
        Two price series (log prices in this project).

    Returns
    -------
    float
        Small values (< 0.05) are evidence that some combination ``y - beta * x``
        is mean-reverting, which is what pairs trading needs.
    """
    # The test regresses y on x, then asks whether the leftover (the residual) keeps
    # returning to its mean or wanders off like a random walk.
    return float(coint(np.asarray(y, dtype=float), np.asarray(x, dtype=float))[1])


def run_backtest(
    signals: pd.DataFrame,
    ret_x: pd.Series,
    ret_y: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Trade the z-score rule and record daily returns after costs.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain columns ``z`` and ``beta`` (from the Kalman filter or rolling OLS).
    ret_x, ret_y : pd.Series
        Daily simple returns of the two assets, on the same index as ``signals``.
    config : BacktestConfig, optional
        Entry/exit thresholds and cost per dollar traded.

    Returns
    -------
    BacktestResult
        Daily returns, positions and number of trades.

    Notes
    -----
    Position sizing: for every $1 of ``y`` we hold $``beta`` of ``x`` on the other
    side, with ``beta`` frozen at its value on the entry day (no daily re-hedging, so
    no hidden re-hedging costs). Returns are divided by the gross capital
    ``1 + |beta|``. The position is exactly dollar-neutral when ``beta = 1``.
    """
    config = config or BacktestConfig()
    z = signals["z"].to_numpy(dtype=float)
    beta = signals["beta"].to_numpy(dtype=float)
    rx = ret_x.to_numpy(dtype=float)
    ry = ret_y.to_numpy(dtype=float)
    cost = config.cost_bps / 1e4

    n = len(z)
    returns = np.zeros(n)
    positions = np.zeros(n)
    position, hedge, n_trades = 0, 0.0, 0

    for t in range(n):
        # 1. Earn today's return on the position carried over from yesterday's close.
        if position != 0:
            returns[t] = position * (ry[t] - hedge * rx[t]) / (1.0 + abs(hedge))

        # 2. Decide what to hold after today's close, using today's z-score.
        target = position
        if position == 0:
            if z[t] > config.entry_z:
                target = -1  # spread too high: short y, long x
            elif z[t] < -config.entry_z:
                target = 1  # spread too low: long y, short x
        elif (position == 1 and z[t] > -config.exit_z) or (position == -1 and z[t] < config.exit_z):
            target = 0  # spread is back to normal (or beyond): take the profit or loss

        # 3. Pay costs whenever the position changes. Both legs trade, so the dollars
        #    traded equal the gross capital and the cost is `cost` per unit of capital.
        if target != position:
            returns[t] -= cost
            if target != 0:
                hedge = beta[t]
                n_trades += 1
            position = target
        positions[t] = position

    index = signals.index
    return BacktestResult(
        pd.Series(returns, index=index), pd.Series(positions, index=index), n_trades
    )


def sweep_delta(
    log_x: pd.Series,
    log_y: pd.Series,
    ret_x: pd.Series,
    ret_y: pd.Series,
    deltas: ArrayLike,
    burn_in: int = 60,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Try several values of ``delta`` on training data and score each by Sharpe ratio.

    Parameters
    ----------
    log_x, log_y : pd.Series
        Log prices over the *training* window only.
    ret_x, ret_y : pd.Series
        Daily simple returns over the same window.
    deltas : array-like
        Candidate values of the process-noise parameter.
    burn_in : int, optional
        Days skipped at the start while the filter warms up.
    config : BacktestConfig, optional
        Trading rule and costs.

    Returns
    -------
    pd.DataFrame
        One row per delta with the fitted ``obs_var``, training ``sharpe`` and ``n_trades``.

    Notes
    -----
    This is the bias-variance tradeoff of the filter. A tiny delta gives a nearly
    constant beta (like one regression over all history) that reacts too slowly; a
    big delta lets beta chase every wiggle, so the "spread" it leaves behind is pure
    noise with nothing left to trade.
    """
    rows = []
    for delta in np.asarray(deltas, dtype=float):
        obs_var = fit_obs_var(log_x, log_y, delta, burn_in)
        signals = run_filter(log_x, log_y, delta, obs_var).iloc[burn_in:]
        result = run_backtest(signals, ret_x.iloc[burn_in:], ret_y.iloc[burn_in:], config)
        rows.append(
            {
                "delta": float(delta),
                "obs_var": obs_var,
                "sharpe": result.metrics()["sharpe"],
                "n_trades": result.n_trades,
            }
        )
    return pd.DataFrame(rows)


def choose_delta(sweep: pd.DataFrame, min_trades: int = 20) -> pd.Series:
    """Pick the row of a delta sweep to use out of sample.

    Parameters
    ----------
    sweep : pd.DataFrame
        Output of :func:`sweep_delta`.
    min_trades : int, optional
        A delta is only eligible if it traded at least this often in training.

    Returns
    -------
    pd.Series
        The eligible row with the highest training Sharpe ratio (or, if no delta
        traded often enough, simply the row with the most trades).
    """
    # A Sharpe ratio built from three or four trades is mostly luck. Requiring a
    # minimum number of trades stops us from "choosing" a delta on a lucky streak.
    eligible = sweep[sweep["n_trades"] >= min_trades]
    if eligible.empty:
        return sweep.loc[sweep["n_trades"].idxmax()]
    return eligible.loc[eligible["sharpe"].idxmax()]
