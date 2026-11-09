"""The efficient frontier of optimal execution.

Every level of risk aversion ``lam`` gives one optimal schedule, and every schedule
has an expected cost and a variance. Plotting those pairs traces the *efficient
frontier*: the best expected cost achievable for each level of risk. No schedule can
sit below the curve, and any schedule above it is wasting either cost or risk.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from qml.execution.almgren_chriss import (
    ACParams,
    expected_cost_and_variance,
    optimal_trajectory,
)


@dataclass(frozen=True)
class FrontierPoint:
    """One schedule on (or off) the frontier.

    Attributes
    ----------
    lam : float
        Risk aversion that produced the schedule.
    expected_cost : float
        Expected implementation shortfall in dollars.
    variance : float
        Variance of the shortfall in dollars squared.
    """

    lam: float
    expected_cost: float
    variance: float

    @property
    def std(self) -> float:
        """Standard deviation of the shortfall in dollars."""
        return float(np.sqrt(self.variance))


def frontier_point(params: ACParams, lam: float) -> FrontierPoint:
    """Cost and risk of the optimal schedule for one risk aversion.

    Parameters
    ----------
    params : ACParams
        Market parameters.
    lam : float
        Risk aversion (0 gives the TWAP schedule).

    Returns
    -------
    FrontierPoint
        Expected cost and variance of the optimal schedule.
    """
    trajectory = optimal_trajectory(
        params.X, params.T, params.n_steps, params.sigma, params.eta, params.gamma, lam
    )
    cost, variance = expected_cost_and_variance(trajectory, params)
    return FrontierPoint(lam, cost, variance)


def efficient_frontier(params: ACParams, lambdas: ArrayLike) -> pd.DataFrame:
    """Sweep risk aversion and tabulate the frontier.

    Parameters
    ----------
    params : ACParams
        Market parameters.
    lambdas : array-like
        Risk-aversion values, usually a log-spaced grid.

    Returns
    -------
    pd.DataFrame
        Columns ``lam``, ``expected_cost``, ``variance`` and ``std``, one row per lambda.
    """
    # More risk aversion -> faster selling -> higher impact cost but lower variance,
    # so the table walks along the frontier from its low-cost end to its low-risk end.
    points = [frontier_point(params, float(lam)) for lam in np.asarray(lambdas, dtype=float)]
    frame = pd.DataFrame([astuple(p) for p in points], columns=["lam", "expected_cost", "variance"])
    frame["std"] = np.sqrt(frame["variance"])
    return frame
