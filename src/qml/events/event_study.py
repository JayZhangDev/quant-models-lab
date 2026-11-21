"""Event studies: how much did one piece of news move a stock?

The idea, in three steps:

1. **Learn what "normal" looks like.** On a quiet *estimation window* well before the
   news, fit a regression of the stock's daily return on one or more *factors* (the
   market, the stock's sector). This says how the stock usually moves with them.
2. **Predict the event days.** Use that regression and the factors' actual returns on
   the event days to predict what the stock "should" have done.
3. **Compare.** The *abnormal return* (AR) is actual minus predicted: the part of the
   move the market and sector cannot explain. Adding ARs over several days gives the
   *cumulative abnormal return* (CAR).

    AR_t  = R_t - (alpha + beta_1 * F1_t + beta_2 * F2_t + ...)
    CAR   = sum of AR_t over the window
    t     = CAR / (sigma * sqrt(L))     sigma = residual std on the estimation window

All positions here are *trading-day offsets* from day 0, the first trading day that
could react to the news (a weekend announcement is day 0 on Monday).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FactorModel:
    """A fitted "normal return" model: ``R = alpha + sum(beta_i * F_i) + noise``.

    Attributes
    ----------
    alpha : float
        Average daily return not explained by the factors.
    betas : pd.Series
        Sensitivity to each factor, indexed by factor name.
    resid_std : float
        Standard deviation of the estimation-window residuals: how big an
        "unexplained" day normally is.
    r_squared : float
        Share of the stock's daily variance the factors explain.
    n_obs : int
        Number of days in the estimation window.
    """

    alpha: float
    betas: pd.Series
    resid_std: float
    r_squared: float
    n_obs: int

    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """Predicted ("normal") stock returns given the factors' actual returns."""
        return self.alpha + factors[self.betas.index] @ self.betas


def fit_factor_model(stock: pd.Series, factors: pd.DataFrame) -> FactorModel:
    """Fit ordinary least squares of stock returns on factor returns.

    Parameters
    ----------
    stock : pd.Series
        Daily returns of the stock over the estimation window.
    factors : pd.DataFrame
        Daily returns of the factors (one column each) on the same dates.

    Returns
    -------
    FactorModel
        Intercept, slopes and residual volatility.
    """
    # Adding a column of ones lets one least-squares solve find alpha and the betas.
    x = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
    y = stock.to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    # Divide by (n - k) rather than n: we used k parameters, which "spends" k data points.
    n, k = x.shape
    resid_std = float(np.sqrt(resid @ resid / (n - k)))
    r_squared = float(1.0 - resid.var() / y.var()) if y.var() > 0 else 0.0
    return FactorModel(
        alpha=float(coef[0]),
        betas=pd.Series(coef[1:], index=factors.columns),
        resid_std=resid_std,
        r_squared=r_squared,
        n_obs=n,
    )


def event_position(index: pd.DatetimeIndex, date: str | pd.Timestamp) -> int:
    """Position of day 0: the first trading day on or after ``date``.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Trading days, sorted.
    date : str or pd.Timestamp
        Calendar date of the news (a weekend date maps to the next Monday).

    Returns
    -------
    int
        Integer position in ``index``.
    """
    pos = int(index.searchsorted(pd.Timestamp(date)))
    if pos >= len(index):
        raise ValueError(f"No trading day on or after {date} in the data.")
    return pos


def estimation_slice(event_pos: int, length: int = 250, gap: int = 30) -> slice:
    """Rows used to fit the normal-return model.

    Parameters
    ----------
    event_pos : int
        Position of day 0.
    length : int, optional
        Number of trading days in the estimation window (about one year).
    gap : int, optional
        Trading days left out between the window's end and day 0, so that any
        build-up to the news cannot leak into "normal".

    Returns
    -------
    slice
        ``slice(event_pos - gap - length, event_pos - gap)``.
    """
    start = event_pos - gap - length
    if start < 0:
        raise ValueError("Not enough history before the event for the estimation window.")
    return slice(start, event_pos - gap)


@dataclass(frozen=True)
class EventStudyResult:
    """Everything one event study produces.

    Attributes
    ----------
    model : FactorModel
        The fitted normal-return model.
    abnormal : pd.DataFrame
        Columns ``actual``, ``predicted``, ``ar`` and ``offset`` (trading days from
        day 0), from ``-pre_days`` up to the last day available.
    event_date : pd.Timestamp
        Day 0.
    """

    model: FactorModel
    abnormal: pd.DataFrame
    event_date: pd.Timestamp

    def car(self, start: int, end: int) -> float:
        """Cumulative abnormal return from offset ``start`` to ``end`` (inclusive)."""
        return cumulative_abnormal_return(self.abnormal, start, end)

    def t_stat(self, start: int, end: int) -> float:
        """t-statistic of the CAR, using the estimation-window residual volatility."""
        length = end - start + 1
        return self.car(start, end) / (self.model.resid_std * np.sqrt(length))


def cumulative_abnormal_return(abnormal: pd.DataFrame, start: int, end: int) -> float:
    """Sum of abnormal returns over trading-day offsets ``start`` to ``end`` inclusive.

    Parameters
    ----------
    abnormal : pd.DataFrame
        Must contain ``offset`` and ``ar`` columns.
    start, end : int
        Window bounds as offsets from day 0.

    Returns
    -------
    float
        The CAR. Raises if any day of the window is missing from the data.
    """
    window = abnormal[(abnormal["offset"] >= start) & (abnormal["offset"] <= end)]
    if len(window) != end - start + 1:
        raise ValueError(f"Window [{start}, {end}] is not fully covered by the data.")
    # Adding simple returns is the standard event-study convention. It is a close
    # approximation to compounding when daily moves are a few percent.
    return float(window["ar"].sum())


def run_event_study(
    stock: pd.Series,
    factors: pd.DataFrame,
    event_date: str | pd.Timestamp,
    est_length: int = 250,
    gap: int = 30,
    pre_days: int = 20,
    last_date: str | pd.Timestamp | None = None,
) -> EventStudyResult:
    """Fit the normal-return model before the event and measure abnormal returns.

    Parameters
    ----------
    stock : pd.Series
        Daily returns of the stock, indexed by trading day.
    factors : pd.DataFrame
        Daily factor returns on the same index (e.g. ``SPY`` and ``SMH``).
    event_date : str or pd.Timestamp
        Calendar date of the news; day 0 is the first trading day on or after it.
    est_length, gap : int, optional
        Estimation window length and the gap before day 0 (see :func:`estimation_slice`).
    pre_days : int, optional
        Trading days before day 0 to include in the abnormal-return table.
    last_date : str or pd.Timestamp, optional
        Last date to include, e.g. the day before a confounding earnings release.

    Returns
    -------
    EventStudyResult
        The model, the abnormal-return table and day 0.
    """
    data = pd.concat([stock.rename("__stock__"), factors], axis=1).dropna()
    if last_date is not None:
        data = data.loc[: pd.Timestamp(last_date)]
    pos = event_position(data.index, event_date)
    est = data.iloc[estimation_slice(pos, est_length, gap)]
    model = fit_factor_model(est["__stock__"], est[factors.columns])

    window = data.iloc[max(pos - pre_days, 0) :]
    predicted = model.predict(window[factors.columns])
    abnormal = pd.DataFrame(
        {
            "actual": window["__stock__"],
            "predicted": predicted,
            "ar": window["__stock__"] - predicted,
            "offset": np.arange(len(window)) - min(pre_days, pos),
        }
    )
    return EventStudyResult(model, abnormal, data.index[pos])


def car_table(result: EventStudyResult, windows: Mapping[str, tuple[int, int]]) -> pd.DataFrame:
    """CAR, t-statistic and actual/predicted totals for several windows.

    Parameters
    ----------
    result : EventStudyResult
        Output of :func:`run_event_study`.
    windows : mapping of str to (int, int)
        Named windows as (start, end) offsets.

    Returns
    -------
    pd.DataFrame
        One row per window: ``window``, ``days``, ``actual``, ``predicted``, ``car``,
        ``t_stat``. Windows not covered by the data are skipped.
    """
    rows = []
    ab = result.abnormal
    for name, (start, end) in windows.items():
        sel = ab[(ab["offset"] >= start) & (ab["offset"] <= end)]
        if len(sel) != end - start + 1:
            continue
        rows.append(
            {
                "window": name,
                "days": end - start + 1,
                "actual": float(sel["actual"].sum()),
                "predicted": float(sel["predicted"].sum()),
                "car": result.car(start, end),
                "t_stat": result.t_stat(start, end),
            }
        )
    return pd.DataFrame(rows)


def placebo_cars(
    stock: pd.Series,
    factors: pd.DataFrame,
    window: tuple[int, int],
    n_draws: int = 1000,
    exclude: Sequence[str | pd.Timestamp] = (),
    exclude_radius: int = 20,
    est_length: int = 250,
    gap: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the same CAR calculation on random "fake event" dates.

    Parameters
    ----------
    stock : pd.Series
        Daily stock returns (the full history).
    factors : pd.DataFrame
        Daily factor returns on the same index.
    window : (int, int)
        CAR window, as offsets from each fake day 0.
    n_draws : int, optional
        Number of random dates.
    exclude : sequence of dates, optional
        Real event dates. Fake dates within ``exclude_radius`` trading days of any of
        them are never drawn, so the real news cannot sneak into the "normal" sample.
    exclude_radius : int, optional
        See ``exclude``.
    est_length, gap : int, optional
        Same estimation settings as the real study.
    seed : int, optional
        Seed for the random choice of dates. Same seed, same dates.

    Returns
    -------
    pd.DataFrame
        One row per fake date: ``date``, ``car`` and ``scar`` (the CAR divided by
        ``sigma * sqrt(L)``, i.e. its t-statistic).

    Notes
    -----
    Why both? A raw CAR of -10% is more unusual for a calm stock than for a wild one.
    NVDA's volatility has changed a lot over the years, so the *standardized* CAR
    (``scar``) gives the fairer comparison.
    """
    data = pd.concat([stock.rename("__stock__"), factors], axis=1).dropna()
    start, end = window
    length = end - start + 1
    first = est_length + gap
    last = len(data) - 1 - max(end, 0)
    candidates = np.arange(first, last + 1)
    for date in exclude:
        pos = event_position(data.index, date)
        candidates = candidates[np.abs(candidates - pos) > exclude_radius]
    if candidates.size < n_draws:
        raise ValueError("Not enough non-event history for the requested number of draws.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(candidates, size=n_draws, replace=False))
    y_all = data["__stock__"].to_numpy(dtype=float)
    x_all = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy(dtype=float)])

    rows = []
    for pos in chosen:
        est = slice(pos - gap - est_length, pos - gap)
        coef, *_ = np.linalg.lstsq(x_all[est], y_all[est], rcond=None)
        resid = y_all[est] - x_all[est] @ coef
        sigma = np.sqrt(resid @ resid / (est_length - x_all.shape[1]))
        win = slice(pos + start, pos + end + 1)
        car = float(np.sum(y_all[win] - x_all[win] @ coef))
        rows.append({"date": data.index[pos], "car": car, "scar": car / (sigma * np.sqrt(length))})
    return pd.DataFrame(rows)


def percentile_of(value: float, sample: Sequence[float] | FloatArray) -> float:
    """Percent of ``sample`` at or below ``value`` (0 = lower than every draw)."""
    arr = np.asarray(sample, dtype=float)
    return float(100.0 * np.mean(arr <= value))


def abnormal_volume(
    volume: pd.Series,
    event_date: str | pd.Timestamp,
    days: Sequence[int],
    baseline: int = 60,
) -> pd.DataFrame:
    """Trading volume on event days relative to its normal level.

    Parameters
    ----------
    volume : pd.Series
        Daily share volume.
    event_date : str or pd.Timestamp
        Calendar date of the news.
    days : sequence of int
        Offsets from day 0 to report.
    baseline : int, optional
        Number of trading days, ending the day before day 0, that define "normal".

    Returns
    -------
    pd.DataFrame
        ``offset``, ``date``, ``volume`` and ``ratio`` (volume / baseline average).
    """
    vol = volume.dropna()
    pos = event_position(vol.index, event_date)
    # The baseline stops the day before day 0, so the event cannot inflate its own yardstick.
    normal = float(vol.iloc[pos - baseline : pos].mean())
    rows = []
    for d in days:
        if 0 <= pos + d < len(vol):
            rows.append(
                {
                    "offset": d,
                    "date": vol.index[pos + d],
                    "volume": float(vol.iloc[pos + d]),
                    "ratio": float(vol.iloc[pos + d]) / normal,
                }
            )
    return pd.DataFrame(rows)
