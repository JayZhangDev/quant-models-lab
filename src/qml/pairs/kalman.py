"""A Kalman filter that tracks a moving hedge ratio between two assets.

Pairs trading bets that two related assets (say Coca-Cola and Pepsi) stay in a
stable relationship ``y = beta * x + alpha``. The catch is that ``beta`` drifts over
time. A Kalman filter treats ``beta`` and ``alpha`` as hidden values that wander
slowly, and updates its best guess of them every day from that day's prices.

The filter is written from scratch in NumPy so every step is visible:

* state (what we are estimating):  ``theta = [beta, alpha]``
* state model (how it moves):      ``theta_t = theta_{t-1} + w_t``,  ``w ~ N(0, Q)``
* observation (what we see):       ``y_t = beta_t * x_t + alpha_t + eps_t``,  ``eps ~ N(0, R)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KalmanStep:
    """What the filter reports after seeing one new ``(x, y)`` pair.

    Attributes
    ----------
    beta : float
        Updated hedge ratio.
    alpha : float
        Updated intercept.
    error : float
        Prediction error ``y - (beta * x + alpha)`` computed *before* the update,
        i.e. using only information from earlier days.
    error_std : float
        Standard deviation the filter expected for that error.
    """

    beta: float
    alpha: float
    error: float
    error_std: float

    @property
    def z(self) -> float:
        """The error in units of its expected size: the live z-score used for trading."""
        return self.error / self.error_std


class KalmanHedge:
    """Online estimator of a time-varying hedge ratio and intercept.

    Parameters
    ----------
    delta : float, optional
        Controls how fast the state is allowed to drift. The process noise is
        ``Q = delta / (1 - delta) * I``. Near 0 the filter behaves like one fixed
        regression over all history; larger values make it forget the past faster.
    obs_var : float, optional
        Observation noise variance ``R``: how noisy ``y`` is around the true line.
    initial_state : tuple of float, optional
        Starting guess for ``(beta, alpha)``.
    initial_cov : float, optional
        Starting uncertainty about that guess (variance on the diagonal of ``P``).
        A large value means "I know nothing yet", so the first data points count a lot.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_var: float = 1e-3,
        initial_state: tuple[float, float] = (0.0, 0.0),
        initial_cov: float = 1.0,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be between 0 and 1, got {delta}")
        if obs_var <= 0.0:
            raise ValueError(f"obs_var must be positive, got {obs_var}")
        self.delta = delta
        self.obs_var = obs_var
        self.state: FloatArray = np.array(initial_state, dtype=float)  # [beta, alpha]
        self.cov: FloatArray = np.eye(2) * initial_cov  # P: uncertainty about the state
        self.process_cov: FloatArray = np.eye(2) * delta / (1.0 - delta)  # Q

    def update(self, x: float, y: float) -> KalmanStep:
        """Take one new observation and update the estimate of ``(beta, alpha)``.

        Parameters
        ----------
        x : float
            Today's value of the hedging asset.
        y : float
            Today's value of the asset being hedged.

        Returns
        -------
        KalmanStep
            Updated beta and alpha, plus the prediction error and its expected std.
        """
        h = np.array([x, 1.0])  # observation vector, so that y = h @ [beta, alpha]

        # 1. Predict. The state is a random walk, so the best guess for today is
        #    yesterday's estimate - but we are a little less sure of it (add Q).
        cov_pred = self.cov + self.process_cov

        # 2. Compare the prediction with what actually happened.
        error = y - h @ self.state
        error_var = h @ cov_pred @ h + self.obs_var  # uncertainty from the state + noise

        # 3. Kalman gain: how much of the surprise to believe. When we are unsure of
        #    the state (big P) the gain is large; when y is noisy (big R) it is small.
        gain = cov_pred @ h / error_var

        # 4. Update the estimate and shrink its uncertainty.
        self.state = self.state + gain * error
        self.cov = cov_pred - np.outer(gain, h) @ cov_pred

        return KalmanStep(
            beta=float(self.state[0]),
            alpha=float(self.state[1]),
            error=float(error),
            error_std=float(np.sqrt(error_var)),
        )


def run_filter(x: ArrayLike, y: ArrayLike, delta: float, obs_var: float) -> pd.DataFrame:
    """Run a fresh :class:`KalmanHedge` over two whole series.

    Parameters
    ----------
    x, y : array-like, shape (T,)
        The two series (this project uses log prices). If they are pandas Series,
        the result keeps their index.
    delta : float
        Process-noise parameter (see :class:`KalmanHedge`).
    obs_var : float
        Observation noise variance ``R``.

    Returns
    -------
    pd.DataFrame
        Columns ``beta``, ``alpha``, ``error``, ``error_std`` and ``z``, one row per day.
    """
    index = x.index if isinstance(x, pd.Series) else None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    kf = KalmanHedge(delta=delta, obs_var=obs_var)
    steps = [kf.update(xi, yi) for xi, yi in zip(x_arr, y_arr, strict=True)]
    frame = pd.DataFrame(
        {
            "beta": [s.beta for s in steps],
            "alpha": [s.alpha for s in steps],
            "error": [s.error for s in steps],
            "error_std": [s.error_std for s in steps],
        },
        index=index,
    )
    frame["z"] = frame["error"] / frame["error_std"]
    return frame


def fit_obs_var(x: ArrayLike, y: ArrayLike, delta: float, burn_in: int = 60) -> float:
    """Choose the observation noise ``R`` that best explains the data, for a given delta.

    Parameters
    ----------
    x, y : array-like, shape (T,)
        Training data only - never the test period.
    delta : float
        Process-noise parameter held fixed during the fit.
    burn_in : int, optional
        Number of initial days ignored while the filter is still warming up.

    Returns
    -------
    float
        Maximum-likelihood estimate of ``R``.

    Notes
    -----
    If ``R`` is wrong the z-scores are mis-scaled: too small an ``R`` makes every day
    look like a 5-sigma event, too large an ``R`` makes nothing ever reach 2. Maximum
    likelihood picks the ``R`` under which the observed errors are "as unsurprising
    as possible", which is what makes ``|z| > 2`` mean roughly what it should.
    """

    def negative_log_likelihood(log10_r: float) -> float:
        out = run_filter(x, y, delta, 10.0**log10_r).iloc[burn_in:]
        variance = out["error_std"].to_numpy() ** 2
        # Gaussian log-likelihood of the prediction errors (constants dropped).
        return float(np.sum(np.log(variance) + out["error"].to_numpy() ** 2 / variance))

    # Search over orders of magnitude: R could plausibly be anywhere from 1e-8 to 1.
    result = minimize_scalar(
        negative_log_likelihood, bounds=(-8.0, 0.0), method="bounded", options={"xatol": 0.02}
    )
    return float(10.0**result.x)


def rolling_ols(x: pd.Series, y: pd.Series, window: int = 60) -> pd.DataFrame:
    """Rolling-window regression signals, the classic alternative to the Kalman filter.

    Parameters
    ----------
    x, y : pd.Series
        The two series on the same index.
    window : int, optional
        Number of past days in each regression.

    Returns
    -------
    pd.DataFrame
        Same columns as :func:`run_filter`: ``beta``, ``alpha``, ``error``,
        ``error_std`` and ``z``. The first ``window`` rows are NaN.

    Notes
    -----
    To make the comparison fair, each day's ``beta`` and ``alpha`` come from the
    regression over the *previous* ``window`` days, so - exactly like the Kalman
    filter's error - today's error is a genuine one-step-ahead prediction error.
    """
    # Regression of y on x from rolling moments: beta = cov(x, y) / var(x).
    var_x = x.rolling(window).var()
    beta = x.rolling(window).cov(y) / var_x
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    # Residual variance of that fit: var(y) - beta^2 var(x), with the usual
    # (n - 1) / (n - 2) correction for having estimated two parameters.
    resid_var = (y.rolling(window).var() - beta**2 * var_x) * (window - 1) / (window - 2)

    beta, alpha = beta.shift(1), alpha.shift(1)  # only use information up to yesterday
    error = y - (beta * x + alpha)
    error_std = np.sqrt(resid_var.clip(lower=0.0)).shift(1)
    return pd.DataFrame(
        {
            "beta": beta,
            "alpha": alpha,
            "error": error,
            "error_std": error_std,
            "z": error / error_std,
        }
    )
