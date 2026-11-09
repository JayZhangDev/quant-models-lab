"""Minimum-variance portfolios and a rolling out-of-sample test of covariance estimators.

The minimum-variance portfolio is the classic stress test for a covariance matrix: the
optimizer hunts for combinations of stocks that *look* almost riskless, and noise in
the matrix creates plenty of fake ones. A better matrix shows up as lower realized
risk and a smaller gap between predicted and realized risk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from qml.common.metrics import TRADING_DAYS
from qml.rmt.marchenko_pastur import clean_correlation, correlation_matrix

FloatArray = NDArray[np.float64]
CovEstimator = Callable[[FloatArray], FloatArray]
WeightSolver = Callable[[FloatArray], FloatArray]


def sample_covariance(returns: ArrayLike) -> FloatArray:
    """Plain sample covariance matrix of a (T, N) array of returns (the "raw" estimate).

    Parameters
    ----------
    returns : array-like, shape (T, N)
        One row per day, one column per asset.

    Returns
    -------
    np.ndarray, shape (N, N)
        Sample covariance matrix.
    """
    return np.cov(np.asarray(returns, dtype=float), rowvar=False)


def cleaned_covariance(returns: ArrayLike) -> FloatArray:
    """Covariance matrix whose correlations were denoised with Marchenko-Pastur clipping.

    Parameters
    ----------
    returns : array-like, shape (T, N)
        One row per day, one column per asset.

    Returns
    -------
    np.ndarray, shape (N, N)
        ``D @ C_clean @ D`` where ``D`` holds each stock's sample volatility.
    """
    # Only the correlations are cleaned. Each stock's own volatility is estimated from
    # T observations of one series, which is far more reliable than the N*(N-1)/2
    # pairwise correlations, so we keep the volatilities as measured.
    r = np.asarray(returns, dtype=float)
    n_obs, n_assets = r.shape
    vols = r.std(axis=0, ddof=1)
    corr = clean_correlation(correlation_matrix(r), q=n_assets / n_obs)
    return corr * np.outer(vols, vols)


def min_variance_weights(cov: ArrayLike) -> FloatArray:
    """Fully invested minimum-variance weights (short positions allowed).

    Parameters
    ----------
    cov : array-like, shape (N, N)
        Covariance matrix (must be invertible).

    Returns
    -------
    np.ndarray, shape (N,)
        ``w = inv(cov) @ 1 / (1' @ inv(cov) @ 1)``; the weights sum to 1.
    """
    # Solving cov @ x = 1 is faster and numerically safer than computing the full
    # inverse. Dividing by the sum then scales the answer to "100% invested".
    cov_arr = np.asarray(cov, dtype=float)
    x = np.linalg.solve(cov_arr, np.ones(cov_arr.shape[0]))
    return x / x.sum()


def min_variance_weights_long_only(cov: ArrayLike) -> FloatArray:
    """Fully invested minimum-variance weights with no short selling.

    Parameters
    ----------
    cov : array-like, shape (N, N)
        Covariance matrix.

    Returns
    -------
    np.ndarray, shape (N,)
        Non-negative weights summing to 1 that minimize ``w' @ cov @ w``.
    """
    # There is no formula once we forbid negative weights, so we hand the problem to a
    # numerical optimizer. Rescaling the matrix to "average variance = 1" keeps the
    # numbers the optimizer sees close to 1, which makes it converge reliably.
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]
    scaled = cov_arr / np.mean(np.diag(cov_arr))
    result = minimize(
        fun=lambda w: float(w @ scaled @ w),
        x0=np.full(n, 1.0 / n),
        jac=lambda w: 2.0 * scaled @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def portfolio_vol(
    weights: ArrayLike, cov: ArrayLike, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized volatility a covariance matrix predicts for a set of weights.

    Parameters
    ----------
    weights : array-like, shape (N,)
        Portfolio weights.
    cov : array-like, shape (N, N)
        Covariance matrix of *periodic* (daily) returns.
    periods_per_year : int, optional
        Periods per year used to annualize.

    Returns
    -------
    float
        ``sqrt(w' @ cov @ w * periods_per_year)``.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ np.asarray(cov, dtype=float) @ w * periods_per_year))


@dataclass(frozen=True)
class RollingResult:
    """Output of :func:`rolling_min_variance`.

    Attributes
    ----------
    daily_returns : pd.DataFrame
        Out-of-sample daily portfolio returns, one column per covariance estimator.
    windows : pd.DataFrame
        One row per (rebalance date, estimator) with the volatility the estimator
        predicted and the volatility that was then realized.
    weights : dict of str to np.ndarray
        For each estimator, an array of shape (n_rebalances, N) with the weights held.
    """

    daily_returns: pd.DataFrame
    windows: pd.DataFrame
    weights: dict[str, FloatArray]


def rolling_min_variance(
    returns: pd.DataFrame,
    estimators: Mapping[str, CovEstimator],
    est_window: int = TRADING_DAYS,
    hold_period: int = 21,
    solver: WeightSolver = min_variance_weights,
) -> RollingResult:
    """Walk forward through time: estimate, build a portfolio, hold it, repeat.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Daily returns, one column per asset.
    estimators : mapping of str to callable
        Named functions that turn a (est_window, N) return array into an (N, N)
        covariance matrix, e.g. ``{"raw": sample_covariance}``.
    est_window : int, optional
        Days of history used to estimate each covariance matrix (default one year).
    hold_period : int, optional
        Days each portfolio is held before re-estimating (default one month).
    solver : callable, optional
        Function turning a covariance matrix into weights.

    Returns
    -------
    RollingResult
        Daily out-of-sample returns, per-window predicted vs realized volatility,
        and the weights held in each window.

    Notes
    -----
    Nothing from the holding month is ever used to choose that month's weights, so
    every return in the result is genuinely out of sample. Weights are held constant
    within the month (rebalanced back to target daily), a standard simplification.
    """
    values = returns.to_numpy(dtype=float)
    n_obs = values.shape[0]
    daily: dict[str, list[pd.Series]] = {name: [] for name in estimators}
    weights: dict[str, list[FloatArray]] = {name: [] for name in estimators}
    rows: list[dict[str, object]] = []

    for start in range(est_window, n_obs - hold_period + 1, hold_period):
        past = values[start - est_window : start]  # what we are allowed to know
        future = values[start : start + hold_period]  # what happens next
        dates = returns.index[start : start + hold_period]
        for name, estimator in estimators.items():
            cov = estimator(past)
            w = solver(cov)
            realized = future @ w
            daily[name].append(pd.Series(realized, index=dates))
            weights[name].append(w)
            rows.append(
                {
                    "rebalance_date": dates[0],
                    "estimator": name,
                    "predicted_vol": portfolio_vol(w, cov),
                    "realized_vol": float(np.std(realized, ddof=1) * np.sqrt(TRADING_DAYS)),
                }
            )

    return RollingResult(
        daily_returns=pd.DataFrame({name: pd.concat(parts) for name, parts in daily.items()}),
        windows=pd.DataFrame(rows),
        weights={name: np.vstack(parts) for name, parts in weights.items()},
    )
