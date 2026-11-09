"""Experiment 1: Marchenko-Pastur cleaning of a real stock correlation matrix.

Run with ``make run-all`` or ``uv run python experiments/run_rmt.py``.

Part A looks at ~300 US stocks over four years and asks: how much of the correlation
matrix is real, and how much is noise? Part B checks whether removing the noise makes
a better portfolio, using a rolling test where the portfolio never sees the future.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qml.common import DATA_DIR
from qml.common.data import clean_returns, get_prices, load_universe
from qml.common.metrics import (
    annualized_return,
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from qml.common.plotting import (
    AQUA,
    BLUE,
    DIVERGING,
    INK,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    apply_style,
    save_fig,
    save_table,
)
from qml.rmt.marchenko_pastur import (
    clean_correlation,
    correlation_matrix,
    count_signal_eigenvalues,
    fit_sigma2,
    mp_bounds,
    mp_pdf,
    sorted_eigh,
)
from qml.rmt.portfolio import (
    CovEstimator,
    RollingResult,
    cleaned_covariance,
    min_variance_weights,
    min_variance_weights_long_only,
    rolling_min_variance,
    sample_covariance,
)

logger = logging.getLogger("run_rmt")

# Fixed dates so the experiment gives the same answer every time it is run.
START, END = "2022-10-31", "2026-10-31"
N_UNIVERSE = 300  # stocks in the full-sample analysis (Part A)
# Part B estimates on one year (252 days). The theory needs q = N/T < 1, i.e. fewer
# stocks than days, so the rolling test uses a random 200-stock subset: q = 0.79.
N_OOS = 200
EST_WINDOW = 252  # one year of history per estimate
HOLD_PERIOD = 21  # hold each portfolio for one month
SEED = 7
# Six pairs of close competitors, so real structure (bright 2x2 blocks) is easy to spot.
HEATMAP_TICKERS = [
    "AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "KO", "PEP", "HD", "LOW",
]  # fmt: skip


def load_returns() -> pd.DataFrame:
    """Download (or load cached) prices and return clean daily returns for the universe."""
    tickers = load_universe(DATA_DIR / "universe_sp500.txt")
    returns = clean_returns(get_prices(tickers, START, END), max_missing=0.05)
    # The universe file is ordered roughly by company size, so "the first 300 that
    # survive cleaning" means the 300 largest names with a complete history.
    return returns.iloc[:, :N_UNIVERSE]


def plot_spectrum(eigenvalues: np.ndarray, q: float, sigma2: float) -> None:
    """Figure (a): eigenvalue histogram against the fitted Marchenko-Pastur density."""
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    n_signal = count_signal_eigenvalues(eigenvalues, q, sigma2)
    x_max = 3.0 * lam_plus
    shown = eigenvalues[eigenvalues <= x_max]
    # Scale the bars so the eigenvalues inside the noise band have total area 1,
    # the same as the Marchenko-Pastur curve they are compared with.
    n_in_band = int(np.sum(eigenvalues <= lam_plus))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    counts, edges = np.histogram(shown, bins=70, range=(0.0, x_max))
    width = edges[1] - edges[0]
    ax.bar(
        edges[:-1],
        counts / (n_in_band * width),
        width=width * 0.9,
        align="edge",
        color=BLUE,
        alpha=0.55,
        label="Measured eigenvalues",
    )
    grid = np.linspace(lam_minus, lam_plus, 600)
    ax.plot(
        grid,
        mp_pdf(grid, q, sigma2),
        color=ORANGE,
        linewidth=2.2,
        label=f"Marchenko-Pastur noise density (fitted $\\sigma^2$ = {sigma2:.2f})",
    )
    top = ax.get_ylim()[1]
    for edge, name in ((lam_minus, "$\\lambda_-$"), (lam_plus, "$\\lambda_+$")):
        ax.axvline(edge, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
        ax.text(edge + 0.02, top * 1.01, f"{name} = {edge:.2f}", ha="left", va="bottom", fontsize=9)

    n_off_chart = int(np.sum(eigenvalues > x_max))
    ax.annotate(
        f"{n_signal} eigenvalues sit above $\\lambda_+$: real structure.\n"
        f"{n_off_chart} are off this chart; the largest is {eigenvalues[0]:.0f}\n"
        f"(the market factor, {eigenvalues[0] / eigenvalues.size:.0%} of all variance).",
        xy=(x_max, top * 0.08),
        xytext=(x_max * 0.98, top * 0.62),
        ha="right",
        va="center",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.0},
    )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, top * 1.08)
    ax.set_xlabel("Eigenvalue of the correlation matrix (1.0 = one average stock's variance)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Most of the correlation matrix is noise: {eigenvalues.size - n_signal} of "
        f"{eigenvalues.size} eigenvalues fall in the noise band",
        pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98))
    save_fig(fig, "rmt_eigenvalue_spectrum")


def plot_market_eigenvector(vector: np.ndarray, tickers: list[str]) -> None:
    """Figure (b): loadings of the largest eigenvector, sorted from high to low."""
    vector = vector if vector.mean() > 0 else -vector  # the overall sign is arbitrary
    order = np.argsort(vector)[::-1]
    loadings = vector[order]
    names = [tickers[i] for i in order]
    uniform = 1.0 / np.sqrt(vector.size)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(np.arange(vector.size), loadings, width=1.0, color=BLUE, alpha=0.75, linewidth=0)
    ax.axhline(uniform, color=ORANGE, linewidth=1.8)
    ax.text(
        vector.size * 0.995,
        uniform * 1.04,
        f"Perfectly equal weights: 1/$\\sqrt{{N}}$ = {uniform:.3f}",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.text(2, loadings[0], f" highest: {names[0]}", ha="left", va="center", fontsize=9)
    ax.annotate(
        f"lowest: {names[-1]}",
        xy=(vector.size - 1, max(loadings[-1], 0.0)),
        xytext=(vector.size - 6, uniform * 0.92),
        ha="right",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.0},
    )
    ax.set_xlim(-1, vector.size)
    ax.set_xlabel(f"The {vector.size} stocks, ranked by loading")
    ax.set_ylabel("Loading (weight in the eigenvector)")
    ax.set_title(
        f"The largest eigenvector is 'the market': all {int(np.sum(vector > 0))} of "
        f"{vector.size} loadings are positive"
    )
    ax.grid(axis="x", visible=False)
    save_fig(fig, "rmt_market_eigenvector")


def plot_heatmaps(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Figure (c): raw vs cleaned correlations for a 12-stock sample, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0))
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))  # leave a strip for the headline
    for ax, matrix, title in zip(
        axes,
        (raw, cleaned),
        ("Raw sample correlation", "Cleaned: noise eigenvalues flattened"),
        strict=True,
    ):
        image = ax.imshow(matrix.to_numpy(), cmap=DIVERGING, vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_title(title)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Print each off-diagonal value so small raw-vs-cleaned changes can be read off.
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i != j:
                    value = matrix.iat[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}".replace("0.", "."),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.6 else INK,
                    )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    colorbar.set_label("Correlation (-1 = opposite, 0 = unrelated, +1 = identical)")
    colorbar.outline.set_visible(False)
    fig.text(
        0.01,
        0.985,
        "Cleaning keeps the broad pattern but shrinks pair-specific links (HD-LOW, XOM-CVX)",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "rmt_correlation_heatmap")


def summarize(
    name: str, constraint: str, result: RollingResult, estimator: str
) -> dict[str, object]:
    """One row of the out-of-sample table for a given estimator."""
    daily = result.daily_returns[estimator].to_numpy()
    windows = result.windows[result.windows["estimator"] == estimator]
    # Average the predicted *variances* (not vols) across months, then take the root,
    # so the prediction is comparable with the full-period realized volatility.
    predicted = float(np.sqrt(np.mean(windows["predicted_vol"] ** 2)))
    realized = annualized_vol(daily)
    return {
        "covariance": name,
        "constraint": constraint,
        "predicted_vol": predicted,
        "realized_vol": realized,
        "realized_over_predicted": realized / predicted,
        "ann_return": annualized_return(daily),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(daily),
        "monthly_turnover": turnover(result.weights[estimator]),
    }


def plot_oos(daily: pd.DataFrame) -> None:
    """Figure (d): growth of $1 and rolling volatility, out of sample."""
    colors = {"Raw covariance": ORANGE, "Cleaned covariance": BLUE, "Equal weight": AQUA}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True)
    for name, color in colors.items():
        growth = (1.0 + daily[name]).cumprod()
        top.plot(growth.index, growth, color=color, label=name)
        rolling = daily[name].rolling(63).std() * np.sqrt(252) * 100
        bottom.plot(rolling.index, rolling, color=color, label=name)
    top.set_ylabel("Growth of $1")
    top.set_title("Out of sample: minimum-variance portfolios from raw vs cleaned covariance")
    top.legend(loc="upper left", ncols=3)
    bottom.set_ylabel("Volatility, % per year")
    bottom.set_xlabel("Date")
    bottom.set_title("Rolling 3-month realized volatility (lower is better for this portfolio)")
    save_fig(fig, "rmt_oos_performance")


def plot_predicted_vs_realized(table: pd.DataFrame) -> None:
    """Figure (e): the volatility each matrix promised vs what actually happened."""
    rows = table[table["covariance"] != "Equal weight"]
    labels = [f"{c}\n{k}" for c, k in zip(rows["covariance"], rows["constraint"], strict=True)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for offset, column, color, label in (
        (-0.2, "predicted_vol", BLUE, "Predicted by the covariance matrix"),
        (0.2, "realized_vol", ORANGE, "Realized out of sample"),
    ):
        heights = rows[column].to_numpy() * 100
        ax.bar(x + offset, heights, width=0.36, color=color, label=label)
        for xi, h in zip(x + offset, heights, strict=True):
            ax.text(xi, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Volatility, % per year")
    ax.set_title("The raw matrix promises far less risk than it delivers")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    save_fig(fig, "rmt_predicted_vs_realized")


def cleaned_with_rule(rule: str) -> CovEstimator:
    """Build a cleaned-covariance estimator that chooses sigma2 by a simpler, named rule.

    Used only for the robustness check: does the result depend on how the noise
    threshold is picked?
    """

    def estimator(window: np.ndarray) -> np.ndarray:
        n_obs, n_assets = window.shape
        corr = correlation_matrix(window)
        eigenvalues, _ = sorted_eigh(corr)
        # "textbook": pretend there is no market factor, so sigma2 = 1.
        # "market": Laloux et al.'s rule, subtract only the largest eigenvalue's share.
        sigma2 = 1.0 if rule == "textbook" else 1.0 - eigenvalues[0] / n_assets
        vols = window.std(axis=0, ddof=1)
        return clean_correlation(corr, n_assets / n_obs, sigma2) * np.outer(vols, vols)

    return estimator


def run_out_of_sample(returns: pd.DataFrame) -> pd.DataFrame:
    """Part B: rolling one-year-estimate / one-month-hold test on a 200-stock subset."""
    rng = np.random.default_rng(SEED)
    subset = sorted(rng.choice(returns.columns, size=N_OOS, replace=False))
    sample = returns[subset]
    estimators = {"raw": sample_covariance, "cleaned": cleaned_covariance}

    logger.info("Rolling test: %d stocks, q = %.2f", N_OOS, N_OOS / EST_WINDOW)
    free = rolling_min_variance(sample, estimators, EST_WINDOW, HOLD_PERIOD, min_variance_weights)
    logger.info("Rolling test again with the long-only constraint (slower)...")
    long_only = rolling_min_variance(
        sample, estimators, EST_WINDOW, HOLD_PERIOD, min_variance_weights_long_only
    )

    # Robustness: swap the fitted sigma2 for two cruder rules and see if it matters.
    rules = {"textbook": cleaned_with_rule("textbook"), "market": cleaned_with_rule("market")}
    alt = rolling_min_variance(sample, rules, EST_WINDOW, HOLD_PERIOD, min_variance_weights)
    robustness = pd.DataFrame(
        {
            "noise_threshold_rule": [
                "No cleaning (raw)",
                "sigma2 fitted to the histogram (used above)",
                "sigma2 = 1 - largest eigenvalue / N",
                "sigma2 = 1 (textbook, ignores the market factor)",
            ],
            "realized_vol": [
                annualized_vol(free.daily_returns["raw"]),
                annualized_vol(free.daily_returns["cleaned"]),
                annualized_vol(alt.daily_returns["market"]),
                annualized_vol(alt.daily_returns["textbook"]),
            ],
        }
    )
    save_table(robustness, "rmt_threshold_robustness", float_format="{:.3f}")

    equal_weight = sample.loc[free.daily_returns.index].mean(axis=1)
    rows = [
        summarize("Raw covariance", "shorting allowed", free, "raw"),
        summarize("Cleaned covariance", "shorting allowed", free, "cleaned"),
        summarize("Raw covariance", "long-only", long_only, "raw"),
        summarize("Cleaned covariance", "long-only", long_only, "cleaned"),
        {
            "covariance": "Equal weight",
            "constraint": "benchmark",
            "predicted_vol": float("nan"),
            "realized_vol": annualized_vol(equal_weight),
            "realized_over_predicted": float("nan"),
            "ann_return": annualized_return(equal_weight),
            "sharpe": sharpe_ratio(equal_weight),
            "max_drawdown": max_drawdown(equal_weight),
            "monthly_turnover": 0.0,
        },
    ]
    table = pd.DataFrame(rows)
    save_table(table, "rmt_oos", float_format="{:.3f}")

    daily = pd.DataFrame(
        {
            "Raw covariance": free.daily_returns["raw"],
            "Cleaned covariance": free.daily_returns["cleaned"],
            "Equal weight": equal_weight,
        }
    )
    plot_oos(daily)
    plot_predicted_vs_realized(table)
    return table


def main() -> None:
    """Run the whole experiment and write figures and tables to results/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()

    returns = load_returns()
    n_obs, n_assets = returns.shape
    q = n_assets / n_obs
    logger.info("Universe: %d stocks x %d days, q = N/T = %.3f", n_assets, n_obs, q)

    # Part A: full-sample spectrum.
    corr = correlation_matrix(returns.to_numpy())
    eigenvalues, eigenvectors = sorted_eigh(corr)
    sigma2 = fit_sigma2(eigenvalues, q)
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    n_signal = count_signal_eigenvalues(eigenvalues, q, sigma2)
    cleaned = clean_correlation(corr, q, sigma2)
    logger.info(
        "sigma2 = %.3f, noise band = [%.3f, %.3f], %d signal eigenvalues, largest = %.1f",
        sigma2,
        lam_minus,
        lam_plus,
        n_signal,
        eigenvalues[0],
    )

    tickers = list(returns.columns)
    plot_spectrum(eigenvalues, q, sigma2)
    plot_market_eigenvector(eigenvectors[:, 0], tickers)
    sample = [t for t in HEATMAP_TICKERS if t in tickers]
    raw_df = pd.DataFrame(corr, index=tickers, columns=tickers).loc[sample, sample]
    cleaned_df = pd.DataFrame(cleaned, index=tickers, columns=tickers).loc[sample, sample]
    plot_heatmaps(raw_df, cleaned_df)

    spectrum = pd.DataFrame(
        {
            "n_stocks": [n_assets],
            "n_days": [n_obs],
            "q": [q],
            "sigma2": [sigma2],
            "lambda_minus": [lam_minus],
            "lambda_plus": [lam_plus],
            "largest_eigenvalue": [float(eigenvalues[0])],
            "market_share_of_variance": [float(eigenvalues[0] / n_assets)],
            "signal_eigenvalues": [n_signal],
            "noise_eigenvalues": [n_assets - n_signal],
        }
    )
    save_table(spectrum, "rmt_spectrum", float_format="{:.3f}")

    # Part B: does cleaning help a real portfolio, out of sample?
    table = run_out_of_sample(returns)
    logger.info("Out-of-sample results:\n%s", table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
