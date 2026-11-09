"""Experiment 2: Avellaneda-Stoikov market making vs a naive symmetric strategy.

Run with ``make run-all`` or ``uv run python experiments/run_avellaneda_stoikov.py``.

Reproduces the simulation study from the 2008 paper: 1,000 trading sessions for each
strategy at three levels of risk aversion (gamma), then compares profit and inventory.
No market data is needed - everything is simulated from a fixed seed.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qml.common.plotting import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    ORANGE,
    apply_style,
    save_fig,
    save_table,
)
from qml.market_making.simulator import ASParams, SimulationResult, simulate
from qml.market_making.strategies import InventoryStrategy, SymmetricStrategy, reservation_price

logger = logging.getLogger("run_avellaneda_stoikov")

GAMMAS = (0.01, 0.1, 0.5)
N_PATHS = 1000
SEED = 42
STRATEGY_COLORS = {"Inventory": BLUE, "Symmetric": ORANGE}


def run_all() -> dict[float, dict[str, SimulationResult]]:
    """Simulate both strategies for every gamma, using the same seed for a fair comparison."""
    results: dict[float, dict[str, SimulationResult]] = {}
    for gamma in GAMMAS:
        params = replace(ASParams(), gamma=gamma)
        results[gamma] = {
            strategy.name: simulate(strategy, params, N_PATHS, SEED)
            for strategy in (InventoryStrategy(), SymmetricStrategy())
        }
    return results


def summary_table(results: dict[float, dict[str, SimulationResult]]) -> pd.DataFrame:
    """Build the table in the paper's format: one row per (gamma, strategy)."""
    rows = []
    for gamma, by_strategy in results.items():
        for name, result in by_strategy.items():
            rows.append(
                {
                    "gamma": gamma,
                    "strategy": name,
                    "mean_spread": result.mean_spread,
                    "mean_profit": float(result.pnl.mean()),
                    "std_profit": float(result.pnl.std(ddof=1)),
                    "mean_final_q": float(result.final_inventory.mean()),
                    "std_final_q": float(result.final_inventory.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def plot_sample_path(result: SimulationResult, params: ASParams, path: int = 0) -> None:
    """Figure: one session of the inventory strategy - prices and quotes, then inventory."""
    quote_times = result.times[:-1]
    reservation = np.array(
        [
            reservation_price(result.mid[i, path], result.inventory[i, path], t, params)
            for i, t in enumerate(quote_times)
        ]
    )
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 6.4), sharex=True, height_ratios=[2.2, 1.0]
    )
    top.plot(quote_times, result.ask[:, path], color=ORANGE, linewidth=1.0, label="Ask quote")
    top.plot(result.times, result.mid[:, path], color=INK_SECONDARY, label="Mid-price")
    top.plot(quote_times, reservation, color=BLUE, label="Reservation price")
    top.plot(quote_times, result.bid[:, path], color=AQUA, linewidth=1.0, label="Bid quote")
    top.set_ylabel("Price")
    top.set_title(
        f"One simulated session (gamma = {params.gamma}): quotes follow the reservation "
        "price, not the mid"
    )
    top.legend(loc="upper left", ncols=4)

    bottom.step(result.times, result.inventory[:, path], where="post", color=BLUE)
    bottom.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    bottom.set_ylabel("Inventory (shares)")
    bottom.set_xlabel("Time (fraction of the session)")
    bottom.set_title("Inventory: when it is positive the reservation price sits below the mid")
    save_fig(fig, "as_sample_path")


def plot_histograms(
    results: dict[float, dict[str, SimulationResult]],
    attribute: str,
    xlabel: str,
    title: str,
    name: str,
    integer_bins: bool = False,
) -> None:
    """Figure: overlaid histograms of both strategies, one panel per gamma."""
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(12.0, 4.4), sharey=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))  # leave a strip for the headline
    for ax, gamma in zip(axes, GAMMAS, strict=True):
        values = {n: getattr(r, attribute) for n, r in results[gamma].items()}
        low = min(v.min() for v in values.values())
        high = max(v.max() for v in values.values())
        # Shared bins so the two histograms in a panel are directly comparable.
        bins = np.arange(np.floor(low) - 0.5, np.ceil(high) + 1.5) if integer_bins else 40
        for strategy_name, v in values.items():
            ax.hist(
                v,
                bins=bins,
                range=None if integer_bins else (low, high),
                color=STRATEGY_COLORS[strategy_name],
                alpha=0.55,
                label=f"{strategy_name} (std {v.std(ddof=1):.1f})",
            )
        ax.set_title(f"gamma = {gamma}")
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(f"Number of sessions (out of {N_PATHS})")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.25)  # headroom so legends clear the bars
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold")
    save_fig(fig, name)


def main() -> None:
    """Run the simulations and write the table and figures to results/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()

    results = run_all()
    table = summary_table(results)
    save_table(table, "as_results", float_format="{:.2f}")
    logger.info("Results (%d sessions each):\n%s", N_PATHS, table.round(2).to_string(index=False))

    base = replace(ASParams(), gamma=0.1)
    plot_sample_path(results[0.1]["Inventory"], base)
    plot_histograms(
        results,
        attribute="pnl",
        xlabel="Profit at the end of the session",
        title="Profit: the inventory strategy is more consistent; it pays with a lower mean, "
        "and pays more as gamma grows",
        name="as_pnl_histograms",
    )
    plot_histograms(
        results,
        attribute="final_inventory",
        xlabel="Shares held at the end of the session",
        title="Final inventory: the inventory strategy ends each session close to flat",
        name="as_inventory_histograms",
        integer_bins=True,
    )


if __name__ == "__main__":
    main()
