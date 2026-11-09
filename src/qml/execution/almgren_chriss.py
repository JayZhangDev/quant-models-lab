"""Almgren-Chriss optimal execution: how to sell a large block of shares.

Selling a million shares at once would crush the price (*market impact*). Selling
slowly avoids that, but leaves you exposed to the price drifting away while you wait
(*timing risk*). Almgren and Chriss (2000) find the schedule that best balances the
two for a trader with risk aversion ``lam``:

    minimize   E[cost] + lam * Var[cost]

Notation used throughout (same as the paper):

* ``x_k``: shares still held after step ``k`` (the *trajectory*), ``x_0 = X``, ``x_N = 0``
* ``n_k = x_{k-1} - x_k``: shares sold during step ``k`` (the *trade list*)
* ``tau = T / N``: length of one step

Discrete-time solution, derived
-------------------------------
With linear impact the objective is a sum of squares in the ``x_k``. Setting its
derivative with respect to each interior ``x_j`` to zero gives

    (x_{j-1} - 2 x_j + x_{j+1}) / tau**2 = kappa_tilde**2 * x_j,
    kappa_tilde**2 = lam * sigma**2 / eta_tilde,   eta_tilde = eta - gamma * tau / 2.

Trying ``x_j = exp(+/- kappa * t_j)`` in that equation gives the condition

    2 * (cosh(kappa * tau) - 1) / tau**2 = kappa_tilde**2
    =>  kappa = arccosh(kappa_tilde**2 * tau**2 / 2 + 1) / tau,

and matching the end points ``x_0 = X``, ``x_N = 0`` gives

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T).

As ``tau -> 0``, ``cosh(kappa * tau) - 1 ~ (kappa * tau)**2 / 2`` and ``eta_tilde -> eta``,
so ``kappa -> sqrt(lam * sigma**2 / eta)``: the continuous-time formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ACParams:
    """Inputs of the execution problem. Defaults follow the paper's worked example.

    Attributes
    ----------
    X : float
        Shares to sell (1,000,000).
    T : float
        Time allowed, in trading days (5).
    n_steps : int
        Number of equal trading intervals (65 = 13 half-hour buckets per day).
    s0 : float
        Starting share price in dollars (50).
    sigma : float
        Daily volatility in dollars per share: 30% a year on a $50 stock is
        ``0.30 * 50 / sqrt(250) = 0.95``.
    eta : float
        Temporary impact: dollars per share lost for each share/day of trading speed.
        Trading 1% of daily volume (50,000 shares/day) costs one bid-ask spread
        ($0.125), so ``eta = 0.125 / 50,000 = 2.5e-6``.
    gamma : float
        Permanent impact: dollars the price drops, for good, per share sold. Selling
        10% of daily volume (500,000 shares) moves the price one spread, so
        ``gamma = 0.125 / 500,000 = 2.5e-7``.
    epsilon : float
        Fixed cost per share: half the bid-ask spread, ``0.125 / 2 = 0.0625``.
    daily_volume : float
        Average shares traded per day in the whole market (5,000,000).
    """

    X: float = 1_000_000.0
    T: float = 5.0
    n_steps: int = 65
    s0: float = 50.0
    sigma: float = 0.95
    eta: float = 2.5e-6
    gamma: float = 2.5e-7
    epsilon: float = 0.0625
    daily_volume: float = 5_000_000.0

    @property
    def tau(self) -> float:
        """Length of one trading interval, in days."""
        return self.T / self.n_steps

    @property
    def times(self) -> FloatArray:
        """Grid of times ``t_0 = 0, ..., t_N = T``."""
        return np.linspace(0.0, self.T, self.n_steps + 1)


def kappa_continuous(sigma: float, eta: float, lam: float) -> float:
    """Urgency parameter of the continuous-time solution, ``sqrt(lam * sigma**2 / eta)``.

    Parameters
    ----------
    sigma : float
        Volatility in dollars per share per sqrt(day).
    eta : float
        Temporary impact coefficient.
    lam : float
        Risk aversion (per dollar).

    Returns
    -------
    float
        ``kappa`` in units of 1/day. ``1 / kappa`` is the trade's "half-life": the
        time it takes to work off most of the position.
    """
    return float(np.sqrt(lam * sigma**2 / eta))


def kappa_discrete(tau: float, sigma: float, eta: float, gamma: float, lam: float) -> float:
    """Urgency parameter of the discrete-time solution (see the module docstring).

    Parameters
    ----------
    tau : float
        Length of one trading interval.
    sigma, eta, gamma, lam : float
        Volatility, temporary impact, permanent impact and risk aversion.

    Returns
    -------
    float
        ``kappa`` solving ``2 * (cosh(kappa * tau) - 1) / tau**2 = kappa_tilde**2``.
    """
    # eta_tilde corrects for the fact that, in discrete time, part of the permanent
    # impact of a trade is already felt by that same trade.
    eta_tilde = eta - 0.5 * gamma * tau
    if eta_tilde <= 0.0:
        raise ValueError("Need eta > gamma * tau / 2 for the discrete solution to exist.")
    kappa_tilde_sq = lam * sigma**2 / eta_tilde
    return float(np.arccosh(0.5 * kappa_tilde_sq * tau**2 + 1.0) / tau)


def _sinh_ratio(kappa: float, T: float, times: FloatArray) -> FloatArray:
    """Compute ``sinh(kappa * (T - t)) / sinh(kappa * T)`` without overflowing."""
    # For an impatient trader kappa * T can be large and sinh() would overflow. Writing
    # both sinh's with exponentials and cancelling exp(kappa * T) keeps numbers small.
    if kappa * T < 1e-8:  # no urgency at all: the ratio tends to a straight line
        return 1.0 - times / T
    return (
        np.exp(-kappa * times)
        * (1.0 - np.exp(-2.0 * kappa * (T - times)))
        / (1.0 - np.exp(-2.0 * kappa * T))
    )


def optimal_trajectory(
    X: float, T: float, n_steps: int, sigma: float, eta: float, gamma: float, lam: float
) -> FloatArray:
    """Optimal holdings over time for linear impact (discrete-time closed form).

    Parameters
    ----------
    X : float
        Shares to sell.
    T : float
        Time allowed (days).
    n_steps : int
        Number of trading intervals.
    sigma : float
        Volatility in dollars per share per sqrt(day).
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    lam : float
        Risk aversion. 0 means "only expected cost matters".

    Returns
    -------
    np.ndarray, shape (n_steps + 1,)
        Shares held at each grid time: starts at ``X`` and ends at 0.
    """
    times = np.linspace(0.0, T, n_steps + 1)
    if lam <= 0.0:
        return X * (1.0 - times / T)  # risk-neutral: sell at a constant rate (TWAP)
    kappa = kappa_discrete(T / n_steps, sigma, eta, gamma, lam)
    trajectory = X * _sinh_ratio(kappa, T, times)
    trajectory[-1] = 0.0  # exact zero at the end, free of rounding error
    return trajectory


def continuous_trajectory(
    X: float, T: float, times: ArrayLike, sigma: float, eta: float, lam: float
) -> FloatArray:
    """Continuous-time optimum ``x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)``.

    Parameters
    ----------
    X, T : float
        Shares to sell and time allowed.
    times : array-like
        Times at which to evaluate the trajectory.
    sigma, eta, lam : float
        Volatility, temporary impact and risk aversion.

    Returns
    -------
    np.ndarray
        Holdings at each requested time.
    """
    t = np.asarray(times, dtype=float)
    return X * _sinh_ratio(kappa_continuous(sigma, eta, lam), T, t)


def twap_trajectory(X: float, n_steps: int) -> FloatArray:
    """Sell the same number of shares in every interval (time-weighted average price).

    Parameters
    ----------
    X : float
        Shares to sell.
    n_steps : int
        Number of trading intervals.

    Returns
    -------
    np.ndarray, shape (n_steps + 1,)
        Holdings falling in a straight line from ``X`` to 0.
    """
    return X * (1.0 - np.arange(n_steps + 1) / n_steps)


def expected_cost_and_variance(trajectory: ArrayLike, params: ACParams) -> tuple[float, float]:
    """Closed-form mean and variance of trading cost under linear impact.

    Parameters
    ----------
    trajectory : array-like, shape (n_steps + 1,)
        Holdings at each grid time.
    params : ACParams
        Market parameters.

    Returns
    -------
    expected_cost : float
        ``gamma * X**2 / 2 + epsilon * sum|n_k| + (eta_tilde / tau) * sum(n_k**2)`` dollars.
    variance : float
        ``sigma**2 * tau * sum(x_k**2 for k = 1..N)`` dollars squared.

    Notes
    -----
    "Cost" is implementation shortfall: what the shares were worth at the starting
    price minus the cash actually received.
    """
    x = np.asarray(trajectory, dtype=float)
    trades = -np.diff(x)
    tau = params.tau
    eta_tilde = params.eta - 0.5 * params.gamma * tau
    permanent = 0.5 * params.gamma * params.X**2  # same for every schedule
    fixed = params.epsilon * np.abs(trades).sum()  # half-spread on every share
    temporary = eta_tilde / tau * np.sum(trades**2)  # punishes large, fast trades
    # Risk comes from the shares still held while the price random-walks.
    variance = params.sigma**2 * tau * np.sum(x[1:] ** 2)
    return float(permanent + fixed + temporary), float(variance)


def simulate_costs(
    trajectory: ArrayLike, params: ACParams, n_paths: int = 10_000, seed: int = 0
) -> FloatArray:
    """Monte Carlo check: execute a schedule on simulated price paths.

    Parameters
    ----------
    trajectory : array-like, shape (n_steps + 1,)
        Holdings at each grid time.
    params : ACParams
        Market parameters.
    n_paths : int, optional
        Number of simulated price paths.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Implementation shortfall in dollars on each path.

    Notes
    -----
    Price dynamics, one step at a time (linear impact):

    * the trade in step ``k`` executes at ``S_{k-1} - epsilon - eta * n_k / tau``
      (last price, minus half the spread, minus temporary impact);
    * the price then moves to ``S_k = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k``
      (random shock plus permanent impact).
    """
    x = np.asarray(trajectory, dtype=float)
    trades = -np.diff(x)
    tau = params.tau
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, trades.size))

    # Price *before* each trade: start price plus all earlier shocks and impacts.
    moves = params.sigma * np.sqrt(tau) * shocks - params.gamma * trades
    price_before = params.s0 + np.cumsum(moves, axis=1) - moves
    execution_price = price_before - params.epsilon - params.eta * trades / tau
    revenue = execution_price @ trades
    return params.X * params.s0 - revenue


def sqrt_impact_coefficient(params: ACParams) -> float:
    """Scale of the square-root impact model, matched to the linear model at TWAP speed.

    Parameters
    ----------
    params : ACParams
        Market parameters.

    Returns
    -------
    float
        Coefficient ``c`` in ``impact = c * sigma * sqrt(v / V)`` such that trading at
        the TWAP rate ``v = X / T`` costs the same per share under both models.
    """
    # Matching the two models at one speed makes the comparison about *shape* (how
    # cost grows with speed) rather than about one model simply being more expensive.
    twap_rate = params.X / params.T
    linear_impact = params.eta * twap_rate
    return float(linear_impact / (params.sigma * np.sqrt(twap_rate / params.daily_volume)))


def expected_cost(
    trades: ArrayLike, params: ACParams, impact: str = "linear", sqrt_coef: float | None = None
) -> float:
    """Expected cost of a trade list under linear or square-root temporary impact.

    Parameters
    ----------
    trades : array-like, shape (n_steps,)
        Shares sold in each interval (all non-negative).
    params : ACParams
        Market parameters.
    impact : {"linear", "sqrt"}, optional
        Temporary impact model. ``"linear"``: per-share impact ``eta * v``.
        ``"sqrt"``: per-share impact ``c * sigma * sqrt(v / V)``, where ``v = n / tau``
        is the trading speed and ``V`` the market's daily volume.
    sqrt_coef : float, optional
        Coefficient ``c`` of the square-root model. Defaults to
        :func:`sqrt_impact_coefficient`.

    Returns
    -------
    float
        Expected implementation shortfall in dollars.
    """
    n = np.asarray(trades, dtype=float)
    tau = params.tau
    # Permanent impact (linear in both models): gamma/2 * (X^2 - sum n^2).
    permanent = 0.5 * params.gamma * (params.X**2 - np.sum(n**2))
    fixed = params.epsilon * n.sum()
    if impact == "linear":
        temporary = params.eta / tau * np.sum(n**2)
    elif impact == "sqrt":
        c = sqrt_impact_coefficient(params) if sqrt_coef is None else sqrt_coef
        temporary = c * params.sigma * np.sum(n**1.5) / np.sqrt(tau * params.daily_volume)
    else:
        raise ValueError(f"impact must be 'linear' or 'sqrt', got {impact!r}")
    return float(permanent + fixed + temporary)


def optimize_trajectory(
    params: ACParams, lam: float, impact: str = "linear", sqrt_coef: float | None = None
) -> FloatArray:
    """Find the optimal schedule numerically (needed when there is no formula).

    Parameters
    ----------
    params : ACParams
        Market parameters.
    lam : float
        Risk aversion.
    impact : {"linear", "sqrt"}, optional
        Temporary impact model (see :func:`expected_cost`).
    sqrt_coef : float, optional
        Coefficient of the square-root model.

    Returns
    -------
    np.ndarray, shape (n_steps + 1,)
        Holdings minimizing ``E[cost] + lam * Var[cost]`` subject to selling exactly
        ``X`` shares and never buying.

    Notes
    -----
    The optimizer works with *fractions* ``w_k = n_k / X`` (which sum to 1) and an
    objective divided by its value at TWAP. Both rescalings keep every number near 1,
    which numerical optimizers need to converge well.
    """
    n_steps, tau, X = params.n_steps, params.tau, params.X
    if impact == "sqrt" and sqrt_coef is None:
        sqrt_coef = sqrt_impact_coefficient(params)

    def raw_objective(w: FloatArray) -> float:
        trades = X * w
        holdings = X * (1.0 - np.cumsum(w))  # x_1 .. x_N
        variance = params.sigma**2 * tau * np.sum(holdings**2)
        return expected_cost(trades, params, impact, sqrt_coef) + lam * variance

    twap = np.full(n_steps, 1.0 / n_steps)
    scale = raw_objective(twap)

    def objective(w: FloatArray) -> float:
        return raw_objective(w) / scale

    def gradient(w: FloatArray) -> FloatArray:
        w = np.clip(w, 0.0, None)
        trades = X * w
        holdings = X * (1.0 - np.cumsum(w))
        # Selling more in step j lowers every later holding x_j, x_{j+1}, ..., so the
        # variance derivative is a "sum from j to the end" (a reversed cumulative sum).
        d_variance = -2.0 * X * params.sigma**2 * tau * np.cumsum(holdings[::-1])[::-1]
        d_cost = params.epsilon - params.gamma * trades
        if impact == "linear":
            d_cost = d_cost + 2.0 * params.eta / tau * trades
        else:
            d_cost = d_cost + 1.5 * sqrt_coef * params.sigma * np.sqrt(
                trades / (tau * params.daily_volume)
            )
        return (X * d_cost + lam * d_variance) / scale

    result = minimize(
        objective,
        x0=twap,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_steps,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n_steps)}
        ],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    trajectory = X * np.concatenate([[1.0], 1.0 - np.cumsum(weights)])
    trajectory[-1] = 0.0
    return trajectory
