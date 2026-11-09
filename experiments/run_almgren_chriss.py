"""Experiment 4: Almgren-Chriss optimal execution.

Run with ``make run-all`` or ``uv run python experiments/run_almgren_chriss.py``.

The task: sell 1,000,000 shares of a $50 stock within 5 trading days. The script
draws the optimal schedule for several risk aversions, traces the efficient frontier,
compares linear with square-root market impact, and checks the cost formulas against
a 10,000-path Monte Carlo simulation. No market data is needed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qml.common.plotting import (
    BLUE,
    INK,
    INK_SECONDARY,
    ORANGE,
    SEQUENTIAL,
    apply_style,
    save_fig,
    save_table,
)
from qml.execution.almgren_chriss import (
    ACParams,
    expected_cost,
    expected_cost_and_variance,
    kappa_continuous,
    optimal_trajectory,
    optimize_trajectory,
    simulate_costs,
    sqrt_impact_coefficient,
    twap_trajectory,
)
from qml.execution.frontier import efficient_frontier, frontier_point

logger = logging.getLogger("run_almgren_chriss")

PARAMS = ACParams()
TRAJECTORY_LAMBDAS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)  # plus lambda = 0 (TWAP)
FRONTIER_LAMBDAS = np.logspace(-8.5, -4.0, 60)
MODERATE_LAMBDA = 1e-6  # the value used in the paper's example
NERVOUS_LAMBDA = 1e-5  # a trader who badly wants to be done early
N_PATHS = 10_000
SEED = 2024
MILLION = 1e6


def closed_form(lam: float) -> np.ndarray:
    """Optimal linear-impact trajectory for the default parameters."""
    p = PARAMS
    return optimal_trajectory(p.X, p.T, p.n_steps, p.sigma, p.eta, p.gamma, lam)


def plot_trajectories() -> None:
    """Figure: optimal holdings over time for several risk aversions."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    times = PARAMS.times
    ax.plot(
        times,
        twap_trajectory(PARAMS.X, PARAMS.n_steps) / MILLION,
        color=INK_SECONDARY,
        linestyle="--",
        label="lambda = 0 (TWAP: constant speed)",
    )
    # Risk aversion is an ordered quantity, so the lines share one hue and get darker
    # as lambda grows instead of each getting an unrelated colour.
    shades = SEQUENTIAL(np.linspace(0.3, 1.0, len(TRAJECTORY_LAMBDAS)))
    for lam, shade in zip(TRAJECTORY_LAMBDAS, shades, strict=True):
        time_scale = 1.0 / kappa_continuous(PARAMS.sigma, PARAMS.eta, lam)
        ax.plot(
            times,
            closed_form(lam) / MILLION,
            color=shade,
            label=f"lambda = {lam:.0e}  (time scale 1/kappa = {time_scale:.1f} days)",
        )
    ax.set_xlabel("Time (trading days)")
    ax.set_ylabel("Shares still held (millions)")
    ax.set_title("More risk aversion means selling faster at the start")
    ax.legend(loc="upper right", title="Risk aversion (darker = more nervous)")
    ax.set_xlim(0.0, PARAMS.T)
    ax.set_ylim(0.0, None)
    save_fig(fig, "ac_trajectories")


def plot_frontier(frontier: pd.DataFrame) -> None:
    """Figure: efficient frontier with the TWAP and nervous-trader points marked."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        frontier["variance"] / MILLION**2,
        frontier["expected_cost"] / MILLION,
        color=BLUE,
        label="Efficient frontier (one point per lambda)",
    )
    marks = (
        (0.0, "TWAP (lambda -> 0):\ncheapest on average, riskiest", (-12, 75), "right"),
        (MODERATE_LAMBDA, f"lambda = {MODERATE_LAMBDA:.0e}", (14, 16), "left"),
        (
            NERVOUS_LAMBDA,
            f"Nervous trader (lambda = {NERVOUS_LAMBDA:.0e}):\npays more to cut risk",
            (16, 4),
            "left",
        ),
    )
    for lam, text, offset, align in marks:
        point = frontier_point(PARAMS, lam)
        x, y = point.variance / MILLION**2, point.expected_cost / MILLION
        ax.scatter([x], [y], s=70, color=ORANGE, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(
            f"{text}\ncost {y:.2f}M USD, std {point.std / MILLION:.2f}M USD",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("Variance of cost, (USD million) squared   (risk: lower is safer)")
    ax.set_ylabel("Expected cost, USD million")
    ax.set_title("Efficient frontier: every extra dollar of expected cost buys less risk")
    ax.set_xlim(left=-0.05)
    ax.legend(loc="upper right")
    save_fig(fig, "ac_efficient_frontier")


def plot_linear_vs_sqrt(lam: float) -> pd.DataFrame:
    """Figure: optimal schedules under linear and square-root impact at the same lambda."""
    linear = closed_form(lam)
    sqrt_schedule = optimize_trajectory(PARAMS, lam, impact="sqrt")
    times = PARAMS.times
    bucket_starts = times[:-1]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True)
    for schedule, color, label in (
        (linear, BLUE, "Linear impact (closed form)"),
        (sqrt_schedule, ORANGE, "Square-root impact (numerical)"),
    ):
        top.plot(times, schedule / MILLION, color=color, label=label)
        bottom.step(
            bucket_starts, -np.diff(schedule) / 1000.0, where="post", color=color, label=label
        )
    top.set_ylabel("Shares still held (millions)")
    top.set_title(
        f"Same risk aversion (lambda = {lam:.0e}), different impact model: "
        "square-root impact sells harder early"
    )
    top.legend(loc="upper right")
    bottom.set_ylabel("Shares sold per half-hour (thousands)")
    bottom.set_xlabel("Time (trading days)")
    bottom.set_title("Trade sizes: square-root impact opens with a much larger trade, then tapers")
    bottom.set_xlim(0.0, PARAMS.T)
    save_fig(fig, "ac_linear_vs_sqrt")

    # How long does each schedule take to get 99% done?
    def days_to_99(schedule: np.ndarray) -> float:
        return float(times[np.argmax(schedule <= 0.01 * PARAMS.X)])

    return pd.DataFrame(
        {
            "impact_model": ["linear", "square-root"],
            "first_trade_shares": [linear[0] - linear[1], sqrt_schedule[0] - sqrt_schedule[1]],
            "sold_after_day_1_pct": [
                100 * (1 - linear[13] / PARAMS.X),
                100 * (1 - sqrt_schedule[13] / PARAMS.X),
            ],
            "days_to_99pct_done": [days_to_99(linear), days_to_99(sqrt_schedule)],
            "expected_cost_usd": [
                expected_cost(-np.diff(linear), PARAMS, "linear"),
                expected_cost(-np.diff(sqrt_schedule), PARAMS, "sqrt"),
            ],
        }
    )


def monte_carlo_check() -> pd.DataFrame:
    """Simulate each schedule on 10,000 price paths and compare with the formulas."""
    schedules = {
        "TWAP (lambda = 0)": closed_form(0.0),
        f"lambda = {MODERATE_LAMBDA:.0e}": closed_form(MODERATE_LAMBDA),
        f"lambda = {NERVOUS_LAMBDA:.0e}": closed_form(NERVOUS_LAMBDA),
    }
    rows = []
    for name, trajectory in schedules.items():
        cost, variance = expected_cost_and_variance(trajectory, PARAMS)
        simulated = simulate_costs(trajectory, PARAMS, N_PATHS, SEED)
        rows.append(
            {
                "schedule": name,
                "analytic_mean_usd": cost,
                "simulated_mean_usd": float(simulated.mean()),
                "analytic_std_usd": float(np.sqrt(variance)),
                "simulated_std_usd": float(simulated.std(ddof=1)),
                # How many standard errors apart the two means are (|z| < 2 is fine).
                "mean_gap_in_std_errors": float(
                    (simulated.mean() - cost) / np.sqrt(variance / N_PATHS)
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_monte_carlo(table: pd.DataFrame) -> None:
    """Figure: analytic vs simulated mean and standard deviation of cost."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(table))
    panels = (
        ("analytic_mean_usd", "simulated_mean_usd", "Expected cost, USD million"),
        ("analytic_std_usd", "simulated_std_usd", "Standard deviation of cost, USD million"),
    )
    for ax, (analytic, simulated, ylabel) in zip(axes, panels, strict=True):
        for offset, column, color, label in (
            (-0.2, analytic, BLUE, "Closed-form formula"),
            (0.2, simulated, ORANGE, f"Monte Carlo ({N_PATHS:,} paths)"),
        ):
            heights = table[column].to_numpy() / MILLION
            ax.bar(x + offset, heights, width=0.36, color=color, label=label)
            for xi, h in zip(x + offset, heights, strict=True):
                ax.text(xi, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, table["schedule"])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(",")[0])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
        ax.legend(loc="upper right")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))
    fig.text(
        0.01,
        0.985,
        "Monte Carlo check: simulated costs agree with the closed-form mean and risk",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "ac_monte_carlo_check")


def main() -> None:
    """Run every part of the experiment and write figures and tables to results/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()

    parameters = pd.DataFrame(
        {"parameter": list(asdict(PARAMS)), "value": [f"{v:g}" for v in asdict(PARAMS).values()]}
    )
    save_table(parameters, "ac_parameters")

    plot_trajectories()

    frontier = efficient_frontier(PARAMS, FRONTIER_LAMBDAS)
    save_table(frontier, "ac_frontier", float_format="{:.4g}")
    plot_frontier(frontier)

    key_points = pd.DataFrame(
        [
            {
                "schedule": name,
                "lambda": lam,
                "expected_cost_usd": point.expected_cost,
                "std_usd": point.std,
                "cost_bps_of_notional": 1e4 * point.expected_cost / (PARAMS.X * PARAMS.s0),
            }
            for name, lam in (
                ("TWAP", 0.0),
                ("Moderate", MODERATE_LAMBDA),
                ("Nervous trader", NERVOUS_LAMBDA),
            )
            for point in [frontier_point(PARAMS, lam)]
        ]
    )
    save_table(key_points, "ac_key_points", float_format="{:.4g}")
    logger.info("Key schedules:\n%s", key_points.to_string(index=False))

    comparison = plot_linear_vs_sqrt(MODERATE_LAMBDA)
    save_table(comparison, "ac_linear_vs_sqrt", float_format="{:.4g}")
    logger.info(
        "Linear vs square-root impact (c = %.2f):\n%s",
        sqrt_impact_coefficient(PARAMS),
        comparison.to_string(index=False),
    )

    mc = monte_carlo_check()
    save_table(mc, "ac_monte_carlo_check", float_format="{:.4g}")
    plot_monte_carlo(mc)
    logger.info("Monte Carlo check:\n%s", mc.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
