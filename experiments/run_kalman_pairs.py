"""Experiment 3: pairs trading with a Kalman-filter hedge ratio vs a rolling regression.

Run with ``make run-all`` or ``uv run python experiments/run_kalman_pairs.py``.

For three economically linked pairs the script:

1. tests cointegration on the training window (first 60% of the data),
2. picks the Kalman ``delta`` with the best *training* Sharpe ratio (among deltas that
   traded at least 20 times),
3. trades the z-score rule on the untouched last 40%, after 5 bps costs per trade,
4. compares against a 60-day rolling regression on the same test period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qml.common.data import get_prices
from qml.common.plotting import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    SURFACE,
    apply_style,
    save_fig,
    save_table,
)
from qml.pairs.backtest import (
    BacktestConfig,
    BacktestResult,
    choose_delta,
    engle_granger_pvalue,
    run_backtest,
    sweep_delta,
)
from qml.pairs.kalman import rolling_ols, run_filter

logger = logging.getLogger("run_kalman_pairs")

# Each pair is (y, x): we model y = beta * x + alpha on log prices.
PAIRS = (("KO", "PEP"), ("XOM", "CVX"), ("GLD", "GDX"))
START, END = "2014-01-01", "2026-10-31"
TRAIN_FRACTION = 0.6  # first 60% chooses delta, last 40% is the out-of-sample test
DELTAS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
MIN_TRAIN_TRADES = 20  # ignore deltas whose training Sharpe rests on a handful of trades
OLS_WINDOW = 60
BURN_IN = 60  # days ignored at the very start while the filter warms up
CONFIG = BacktestConfig(entry_z=2.0, exit_z=0.5, cost_bps=5.0)
PAIR_COLORS = (BLUE, ORANGE, AQUA)


@dataclass(frozen=True)
class PairRun:
    """Everything computed for one pair, kept together for plotting."""

    name: str
    split_date: pd.Timestamp
    sweep: pd.DataFrame
    best_delta: float
    best_train_sharpe: float
    kalman: pd.DataFrame  # full-sample signals
    ols: pd.DataFrame
    kalman_test: BacktestResult
    ols_test: BacktestResult
    p_train: float
    p_test: float


def analyse_pair(y_ticker: str, x_ticker: str) -> PairRun:
    """Run the full train/test procedure for one pair."""
    prices = get_prices([y_ticker, x_ticker], START, END).dropna()
    log_y, log_x = np.log(prices[y_ticker]), np.log(prices[x_ticker])
    ret_y = prices[y_ticker].pct_change().fillna(0.0)
    ret_x = prices[x_ticker].pct_change().fillna(0.0)
    split = int(len(prices) * TRAIN_FRACTION)
    train, test = slice(0, split), slice(split, None)

    # Step 1: is there a stable long-run relationship at all? (training data only)
    p_train = engle_granger_pvalue(log_y.iloc[train], log_x.iloc[train])
    p_test = engle_granger_pvalue(log_y.iloc[test], log_x.iloc[test])

    # Step 2: choose delta on the training window. For each candidate, the noise
    # level R is fitted by maximum likelihood, also on training data only.
    sweep = sweep_delta(
        log_x.iloc[train],
        log_y.iloc[train],
        ret_x.iloc[train],
        ret_y.iloc[train],
        DELTAS,
        BURN_IN,
        CONFIG,
    )
    best = choose_delta(sweep, MIN_TRAIN_TRADES)

    # Step 3: run the filter over the whole history (it only ever looks backwards),
    # then trade the test period with the parameters frozen from training.
    kalman = run_filter(log_x, log_y, best["delta"], best["obs_var"])
    ols = rolling_ols(log_x, log_y, OLS_WINDOW)
    kalman_test = run_backtest(kalman.iloc[test], ret_x.iloc[test], ret_y.iloc[test], CONFIG)
    ols_test = run_backtest(ols.iloc[test], ret_x.iloc[test], ret_y.iloc[test], CONFIG)

    return PairRun(
        name=f"{y_ticker}/{x_ticker}",
        split_date=prices.index[split],
        sweep=sweep,
        best_delta=float(best["delta"]),
        best_train_sharpe=float(best["sharpe"]),
        kalman=kalman,
        ols=ols,
        kalman_test=kalman_test,
        ols_test=ols_test,
        p_train=p_train,
        p_test=p_test,
    )


def results_table(runs: list[PairRun]) -> pd.DataFrame:
    """Out-of-sample metrics: one row per pair and method."""
    rows = []
    for run in runs:
        for method, result in (
            (f"Kalman (delta={run.best_delta:.0e})", run.kalman_test),
            (f"Rolling OLS ({OLS_WINDOW}d)", run.ols_test),
        ):
            metrics = result.metrics()
            rows.append(
                {
                    "pair": run.name,
                    "method": method,
                    "sharpe": metrics["sharpe"],
                    "annual_return": metrics["annual_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "n_trades": int(metrics["n_trades"]),
                }
            )
    return pd.DataFrame(rows)


def cointegration_table(runs: list[PairRun]) -> pd.DataFrame:
    """Engle-Granger p-values and the delta chosen for each pair."""
    return pd.DataFrame(
        {
            "pair": [r.name for r in runs],
            "eg_pvalue_train": [r.p_train for r in runs],
            "eg_pvalue_test": [r.p_test for r in runs],
            "chosen_delta": [f"{r.best_delta:.0e}" for r in runs],
            "train_sharpe_at_chosen_delta": [r.best_train_sharpe for r in runs],
        }
    )


def mark_split(ax: plt.Axes, run: PairRun, label: bool) -> None:
    """Draw the train/test boundary on a time-series panel."""
    ax.axvline(run.split_date, color=MUTED, linewidth=1.0, linestyle="--")
    if label:
        ax.annotate(
            " test period starts",
            xy=(run.split_date, 1.0),
            xycoords=("data", "axes fraction"),
            va="top",
            fontsize=9,
            color=INK_SECONDARY,
        )


def plot_betas(runs: list[PairRun]) -> None:
    """Figure: hedge ratio over time, Kalman vs rolling OLS, one panel per pair."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(8.5, 8.4), sharex=True)
    for i, (ax, run) in enumerate(zip(axes, runs, strict=True)):
        ols_beta = run.ols["beta"].iloc[BURN_IN:]
        kalman_beta = run.kalman["beta"].iloc[BURN_IN:]
        ax.plot(ols_beta.index, ols_beta, color=ORANGE, linewidth=1.0, label="Rolling OLS (60d)")
        ax.plot(
            kalman_beta.index,
            kalman_beta,
            color=BLUE,
            label=f"Kalman (delta = {run.best_delta:.0e})",
        )
        # Rolling OLS occasionally spikes; keep the axis focused on the typical range.
        low, high = np.nanpercentile(ols_beta, [1, 99])
        pad = 0.15 * (high - low)
        ax.set_ylim(min(low, kalman_beta.min()) - pad, max(high, kalman_beta.max()) + pad)
        ax.set_ylabel("Hedge ratio (beta)")
        ax.set_title(f"{run.name}")
        ax.legend(loc="upper left", ncols=2)
        mark_split(ax, run, label=(i == 0))
    axes[-1].set_xlabel("Date")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.95))
    fig.text(
        0.01,
        0.99,
        "Hedge ratios over time: the Kalman estimate moves smoothly, rolling OLS jumps around",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "kalman_beta")


def plot_zscores(runs: list[PairRun]) -> None:
    """Figure: Kalman z-score in the test period with the entry and exit bands."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(8.5, 8.0), sharex=True)
    for i, (ax, run) in enumerate(zip(axes, runs, strict=True)):
        z = run.kalman["z"].loc[run.split_date :]
        ax.plot(z.index, z, color=BLUE, linewidth=0.9, label="Kalman z-score")
        for level, color, name in (
            (CONFIG.entry_z, ORANGE, f"Entry: |z| > {CONFIG.entry_z:g}"),
            (CONFIG.exit_z, AQUA, f"Exit: |z| < {CONFIG.exit_z:g}"),
        ):
            ax.axhline(level, color=color, linewidth=1.3, linestyle="--", label=name)
            ax.axhline(-level, color=color, linewidth=1.3, linestyle="--")
        ax.set_ylim(-5, 6.5)  # extra room on top for the legend
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_ylabel("z-score")
        ax.set_title(f"{run.name}: {run.kalman_test.n_trades} trades in the test period")
        if i == 0:
            ax.legend(loc="upper right", ncols=3)
    axes[-1].set_xlabel("Date")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.95))
    fig.text(
        0.01,
        0.99,
        "Trading signal: prediction error divided by its expected size (test period only)",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "kalman_zscore")


def plot_equity(runs: list[PairRun]) -> None:
    """Figure: growth of $1 in the test period, after costs, Kalman vs rolling OLS."""
    fig, axes = plt.subplots(1, len(runs), figsize=(12.0, 4.6), sharey=True)
    for ax, run in zip(axes, runs, strict=True):
        for result, color, label in (
            (run.kalman_test, BLUE, "Kalman"),
            (run.ols_test, ORANGE, "Rolling OLS (60d)"),
        ):
            equity = (1.0 + result.returns).cumprod()
            sharpe = result.metrics()["sharpe"]
            ax.plot(equity.index, equity, color=color, label=f"{label} (Sharpe {sharpe:.2f})")
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.set_title(run.name)
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.tick_params(axis="x", labelrotation=30)
    # The y-axis is shared: add headroom so the legends sit above every curve.
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low, high + 0.28 * (high - low))
    axes[0].set_ylabel("Growth of $1 (after 5 bps costs)")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))
    fig.text(
        0.01,
        0.985,
        "Out-of-sample equity curves: last 40% of the data, never used to choose parameters",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "kalman_equity")


def plot_delta_sweep(runs: list[PairRun]) -> None:
    """Figure: training Sharpe ratio as a function of delta."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for run, color in zip(runs, PAIR_COLORS, strict=True):
        sweep = run.sweep
        ax.plot(sweep["delta"], sweep["sharpe"], color=color, label=run.name)
        # Filled dot = traded often enough to trust; hollow dot = too few trades.
        enough = sweep["n_trades"] >= MIN_TRAIN_TRADES
        ax.scatter(sweep["delta"][enough], sweep["sharpe"][enough], s=36, color=color, zorder=3)
        ax.scatter(
            sweep["delta"][~enough],
            sweep["sharpe"][~enough],
            s=36,
            facecolors=SURFACE,
            edgecolors=color,
            linewidths=1.5,
            zorder=3,
        )
        ax.scatter(
            [run.best_delta],
            [run.best_train_sharpe],
            s=190,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(
        "delta (process noise).  Small: beta nearly fixed, few slow trades.  "
        "Large: beta chases noise, no signal left."
    )
    ax.set_ylabel("Sharpe ratio on training data (after costs)")
    ax.set_title(
        f"Choosing delta on training data. Hollow dots: under {MIN_TRAIN_TRADES} trades "
        "(not eligible). Rings: chosen."
    )
    ax.legend(loc="lower left")
    save_fig(fig, "kalman_delta_sweep")


def main() -> None:
    """Run the experiment for every pair and write tables and figures to results/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()

    runs = [analyse_pair(y, x) for y, x in PAIRS]

    cointegration = cointegration_table(runs)
    save_table(cointegration, "kalman_cointegration", float_format="{:.3f}")
    logger.info("Cointegration and chosen delta:\n%s", cointegration.to_string(index=False))

    sweep = pd.concat([r.sweep.assign(pair=r.name) for r in runs], ignore_index=True)
    sweep["delta"] = sweep["delta"].map("{:.0e}".format)
    sweep["obs_var"] = sweep["obs_var"].map("{:.2e}".format)
    save_table(sweep[["pair", "delta", "obs_var", "sharpe", "n_trades"]], "kalman_delta_sweep")

    table = results_table(runs)
    save_table(table, "kalman_results", float_format="{:.3f}")
    logger.info("Out-of-sample results:\n%s", table.round(3).to_string(index=False))

    plot_betas(runs)
    plot_zscores(runs)
    plot_equity(runs)
    plot_delta_sweep(runs)


if __name__ == "__main__":
    main()
