"""A simulated market for testing market-making strategies (Avellaneda & Stoikov, 2008).

A market maker continuously posts a *bid* (a price at which they will buy) and an
*ask* (a price at which they will sell). They earn the gap between the two whenever
both get filled, but in between they hold *inventory* whose value moves with the
market. This module simulates that world; ``strategies.py`` decides where to quote.

The simulation is vectorized: every array has one column per simulated path, so
1,000 paths cost one pass through time rather than 1,000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ASParams:
    """Parameters of the Avellaneda-Stoikov model (defaults are the paper's).

    Attributes
    ----------
    s0 : float
        Starting mid-price.
    T : float
        Length of the trading session (1.0 = one session).
    sigma : float
        Volatility of the mid-price, in price units per sqrt(session).
    dt : float
        Length of one time step.
    gamma : float
        Risk aversion of the market maker. Higher = more afraid of holding inventory.
    k : float
        How fast the chance of a fill decays as a quote moves away from the mid-price.
    A : float
        Fill rate (orders per session) for a quote placed exactly at the mid-price.
    """

    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    gamma: float = 0.1
    k: float = 1.5
    A: float = 140.0

    @property
    def n_steps(self) -> int:
        """Number of time steps in one session."""
        return int(round(self.T / self.dt))


class QuoteStrategy(Protocol):
    """Anything that can turn the current state into a bid and an ask."""

    name: str

    def quotes(
        self, mid: FloatArray, inventory: FloatArray, t: float, params: ASParams
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(bid, ask)`` arrays, one entry per simulated path."""
        ...


def fill_intensity(delta: ArrayLike, params: ASParams) -> FloatArray:
    """Arrival rate of orders that fill a quote ``delta`` away from the mid-price.

    Parameters
    ----------
    delta : array-like
        Distance of the quote from the mid-price (ask - mid, or mid - bid).
    params : ASParams
        Model parameters; uses ``A`` and ``k``.

    Returns
    -------
    np.ndarray
        ``A * exp(-k * delta)`` fills per unit of time.
    """
    # The further your quote sits from the mid-price, the worse the deal for the other
    # side, so fewer traders take it. The paper models that drop-off as exponential.
    return params.A * np.exp(-params.k * np.asarray(delta, dtype=float))


@dataclass(frozen=True)
class SimulationResult:
    """Full history of a simulation. Arrays are indexed ``[time step, path]``.

    Attributes
    ----------
    times : np.ndarray, shape (n_steps + 1,)
        Time of each grid point.
    mid : np.ndarray, shape (n_steps + 1, n_paths)
        Mid-price paths.
    bid, ask : np.ndarray, shape (n_steps, n_paths)
        Quotes posted at the start of each step.
    inventory : np.ndarray, shape (n_steps + 1, n_paths)
        Shares held (negative = short).
    cash : np.ndarray, shape (n_steps + 1, n_paths)
        Cash from all trades so far (starts at 0).
    """

    times: FloatArray
    mid: FloatArray
    bid: FloatArray
    ask: FloatArray
    inventory: FloatArray
    cash: FloatArray

    @property
    def pnl(self) -> FloatArray:
        """Final profit of each path: cash plus leftover inventory valued at the last mid."""
        return self.cash[-1] + self.inventory[-1] * self.mid[-1]

    @property
    def final_inventory(self) -> FloatArray:
        """Inventory of each path at the end of the session."""
        return self.inventory[-1]

    @property
    def mean_spread(self) -> float:
        """Average quoted bid-ask spread over all steps and paths."""
        return float(np.mean(self.ask - self.bid))


def simulate(
    strategy: QuoteStrategy, params: ASParams, n_paths: int = 1000, seed: int = 0
) -> SimulationResult:
    """Simulate many independent trading sessions for one quoting strategy.

    Parameters
    ----------
    strategy : QuoteStrategy
        Object with a ``quotes(mid, inventory, t, params)`` method.
    params : ASParams
        Model parameters.
    n_paths : int, optional
        Number of independent sessions to simulate at once.
    seed : int, optional
        Seed for the random number generator. Same seed, same result.

    Returns
    -------
    SimulationResult
        Price, quote, inventory and cash histories for every path.

    Notes
    -----
    All random numbers are drawn up front. Two strategies simulated with the same
    seed therefore face exactly the same price paths and the same "luck" in order
    arrivals, which makes their comparison much less noisy.
    """
    n = params.n_steps
    rng = np.random.default_rng(seed)
    price_moves = rng.choice([-1.0, 1.0], size=(n, n_paths))  # up or down, 50/50
    u_bid = rng.random((n, n_paths))  # one uniform draw per possible fill
    u_ask = rng.random((n, n_paths))

    mid = np.empty((n + 1, n_paths))
    inventory = np.zeros((n + 1, n_paths))
    cash = np.zeros((n + 1, n_paths))
    bid = np.empty((n, n_paths))
    ask = np.empty((n, n_paths))
    mid[0] = params.s0
    tick = params.sigma * np.sqrt(params.dt)  # size of one price move

    # The loop runs over time only; each line inside handles all paths at once.
    for i in range(n):
        t = i * params.dt
        bid[i], ask[i] = strategy.quotes(mid[i], inventory[i], t, params)

        # Chance of a fill during this step = arrival rate x step length (capped at 1).
        p_bid = np.minimum(fill_intensity(mid[i] - bid[i], params) * params.dt, 1.0)
        p_ask = np.minimum(fill_intensity(ask[i] - mid[i], params) * params.dt, 1.0)
        bought = u_bid[i] < p_bid  # a seller hit our bid: we buy one share
        sold = u_ask[i] < p_ask  # a buyer lifted our ask: we sell one share

        inventory[i + 1] = inventory[i] + bought - sold
        cash[i + 1] = cash[i] - bid[i] * bought + ask[i] * sold
        mid[i + 1] = mid[i] + tick * price_moves[i]

    times = np.arange(n + 1) * params.dt
    return SimulationResult(times, mid, bid, ask, inventory, cash)
