"""Quoting strategies: the Avellaneda-Stoikov inventory strategy and a naive baseline.

Both strategies quote the *same* total spread. They differ only in where the quotes
are centered:

* ``SymmetricStrategy`` centers on the mid-price and ignores inventory.
* ``InventoryStrategy`` centers on the *reservation price*, which shifts away from
  the mid-price whenever the market maker is holding inventory they want to shed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qml.market_making.simulator import ASParams

FloatArray = NDArray[np.float64]


def reservation_price(
    mid: ArrayLike, inventory: ArrayLike, t: float, params: ASParams
) -> FloatArray:
    """Price at which the market maker is indifferent between buying and selling.

    Parameters
    ----------
    mid : array-like
        Current mid-price(s).
    inventory : array-like
        Current inventory ``q`` (positive = long, negative = short).
    t : float
        Current time, between 0 and ``params.T``.
    params : ASParams
        Model parameters; uses ``gamma``, ``sigma`` and ``T``.

    Returns
    -------
    np.ndarray
        ``r = s - q * gamma * sigma**2 * (T - t)``.
    """
    # If we are long (q > 0), r drops below the mid: both quotes shift down, making our
    # ask more attractive and our bid less so - we lean toward selling. Short does the
    # opposite. The shift fades to zero at the close, when there is no risk left to hold.
    s = np.asarray(mid, dtype=float)
    q = np.asarray(inventory, dtype=float)
    return s - q * params.gamma * params.sigma**2 * (params.T - t)


def optimal_spread(t: float, params: ASParams) -> float:
    """Total bid-ask spread the model says to quote at time ``t``.

    Parameters
    ----------
    t : float
        Current time, between 0 and ``params.T``.
    params : ASParams
        Model parameters; uses ``gamma``, ``sigma``, ``k`` and ``T``.

    Returns
    -------
    float
        ``gamma * sigma**2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)``.
    """
    # Two parts: (1) a risk premium for the price moving while we hold inventory, which
    # shrinks as the session ends; (2) a constant part that balances "wider spread =
    # more profit per trade" against "wider spread = fewer trades".
    risk_part = params.gamma * params.sigma**2 * (params.T - t)
    competition_part = (2.0 / params.gamma) * np.log(1.0 + params.gamma / params.k)
    return float(risk_part + competition_part)


@dataclass(frozen=True)
class InventoryStrategy:
    """Avellaneda-Stoikov: quote the optimal spread around the reservation price."""

    name: str = "Inventory"

    def quotes(
        self, mid: FloatArray, inventory: FloatArray, t: float, params: ASParams
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(bid, ask)`` centered on the reservation price."""
        center = reservation_price(mid, inventory, t, params)
        half = optimal_spread(t, params) / 2.0
        return center - half, center + half


@dataclass(frozen=True)
class SymmetricStrategy:
    """Baseline: quote the same spread around the mid-price, whatever the inventory."""

    name: str = "Symmetric"

    def quotes(
        self, mid: FloatArray, inventory: FloatArray, t: float, params: ASParams
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(bid, ask)`` centered on the mid-price."""
        center = np.asarray(mid, dtype=float)
        half = optimal_spread(t, params) / 2.0
        return center - half, center + half
