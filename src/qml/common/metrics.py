"""Performance metrics shared by every backtest in the repo.

All functions take *periodic* returns (daily unless stated otherwise) as plain arrays
and return a single float, so they are easy to check by hand.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

TRADING_DAYS = 252  # trading days in a typical US market year


def annualized_vol(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized volatility: the standard deviation of returns, scaled to one year.

    Parameters
    ----------
    returns : array-like
        Periodic (e.g. daily) returns.
    periods_per_year : int, optional
        How many periods make a year (252 for daily data).

    Returns
    -------
    float
        Volatility per year, e.g. 0.20 means 20%.
    """
    # Variance adds up over independent days, so volatility grows with the square root
    # of time: daily vol x sqrt(252) = yearly vol.
    r = np.asarray(returns, dtype=float)
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def annualized_return(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by a series of periodic returns.

    Parameters
    ----------
    returns : array-like
        Periodic simple returns.
    periods_per_year : int, optional
        How many periods make a year.

    Returns
    -------
    float
        Growth rate per year, e.g. 0.08 means 8%.
    """
    # Compounding: grow $1 by every return, then ask what constant yearly rate
    # would have produced the same final amount.
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0
    return growth ** (periods_per_year / r.size) - 1.0


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: int = TRADING_DAYS, risk_free: float = 0.0
) -> float:
    """Annualized Sharpe ratio: average excess return per unit of volatility.

    Parameters
    ----------
    returns : array-like
        Periodic returns.
    periods_per_year : int, optional
        How many periods make a year.
    risk_free : float, optional
        Risk-free rate *per period* to subtract (default 0).

    Returns
    -------
    float
        Sharpe ratio. Returns 0.0 when volatility is zero (e.g. a strategy that never trades).
    """
    # "How much return did I get for each unit of risk?" A Sharpe of 1 is good,
    # 2 is excellent, below 0 means you lost money on average.
    excess = np.asarray(returns, dtype=float) - risk_free
    if excess.size < 2:
        return 0.0
    vol = np.std(excess, ddof=1)
    # A strategy that never trades has ~zero volatility; call its Sharpe 0 rather than
    # dividing floating-point dust by floating-point dust.
    if vol < 1e-12:
        return 0.0
    return float(np.mean(excess) / vol * np.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    """Worst peak-to-trough loss of the equity curve built from ``returns``.

    Parameters
    ----------
    returns : array-like
        Periodic simple returns.

    Returns
    -------
    float
        A number between -1 and 0, e.g. -0.25 means the strategy once fell 25%
        below its previous high.
    """
    # Track the highest value seen so far ("the peak") and measure how far below it
    # we are each day. The most painful of those gaps is the max drawdown.
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def turnover(weights: ArrayLike) -> float:
    """Average one-way turnover per rebalance.

    Parameters
    ----------
    weights : array-like, shape (n_rebalances, n_assets)
        Portfolio weights at each rebalance date, one row per rebalance.

    Returns
    -------
    float
        Mean of ``0.5 * sum(|w_t - w_{t-1}|)``. A value of 0.30 means that, on a
        typical rebalance, 30% of the portfolio is sold and replaced.
    """
    # Every dollar sold is matched by a dollar bought, so the sum of absolute weight
    # changes counts each trade twice. Halving it gives "one-way" turnover.
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2:
        return 0.0
    return float(np.mean(0.5 * np.abs(np.diff(w, axis=0)).sum(axis=1)))
