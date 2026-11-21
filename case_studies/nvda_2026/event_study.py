"""Phase 1 of the NVDA case study: an event study of the November 2026 news.

Run with ``uv run python case_studies/nvda_2026/run_all.py`` (or this file directly).

Question: how much of NVDA's fall after the board announced its investigation on
Sunday, Nov 8, 2026 was caused by that news, and how much was the market, the chip
sector, or a slide that had already started? Event dates and sources are in
``events.yaml``. Figures and tables go to ``case_studies/nvda_2026/results/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qml.common.data import get_prices
from qml.common.plotting import (
    AQUA,
    BLUE,
    INK,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    apply_style,
    save_fig,
    save_table,
)
from qml.events.event_study import (
    EventStudyResult,
    abnormal_volume,
    car_table,
    percentile_of,
    placebo_cars,
    run_event_study,
)

logger = logging.getLogger("nvda_event_study")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVENTS_FILE = HERE / "events.yaml"

STOCK = "NVDA"
MARKET, SECTOR = "SPY", "SMH"
PEERS = ("AMD", "AVGO", "TSM", "MU")
START, END = "2014-01-01", "2026-11-21"  # END is exclusive: data runs to Fri Nov 20
MODELS = {"Market model (SPY)": [MARKET], "Two-factor (SPY + SMH)": [MARKET, SECTOR]}
TWO_FACTOR = "Two-factor (SPY + SMH)"
EST_LENGTH, GAP = 250, 30  # one year of history, ending 30 trading days before day 0
PRE_DAYS = 25  # days before day 0 kept in the abnormal-return table (covers Oct 29)
PLACEBO_DRAWS, PLACEBO_SEED = 1000, 2026
HEADLINE_WINDOW = (0, 2)
# Robustness: (estimation length, gap before day 0) pairs, and NVDA's assumed weight in
# SMH for the "sector without NVDA" factor. SMH's NVDA weight was about 20.6% on Oct 24,
# 2026 (stockanalysis.com/etf/SMH/holdings); 15% and 25% bracket how it has drifted.
ROBUST_WINDOWS = ((250, 30), (120, 30), (250, 60), (500, 30))
NVDA_WEIGHTS_IN_SMH = (0.15, 0.20, 0.25)


@dataclass(frozen=True)
class Event:
    """One row of ``events.yaml``."""

    id: str
    date: pd.Timestamp
    trading_day: pd.Timestamp
    role: str
    label: str


def load_events(path: Path = EVENTS_FILE) -> dict[str, Event]:
    """Read the verified event dates."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["events"]
    return {
        e["id"]: Event(
            e["id"], pd.Timestamp(e["date"]), pd.Timestamp(e["trading_day"]), e["role"], e["label"]
        )
        for e in raw
    }


def load_returns() -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns for NVDA, the factors and peers, plus NVDA's daily volume."""
    tickers = [STOCK, MARKET, SECTOR, *PEERS]
    prices = get_prices(tickers, START, END)
    volume = get_prices([STOCK], START, END, field="Volume")[STOCK]
    return prices.pct_change().iloc[1:], volume


def event_windows(result: EventStudyResult, last_offset: int) -> dict[str, tuple[int, int]]:
    """The windows asked for, with the post-event window stopping before earnings."""
    return {
        "[0]": (0, 0),
        "[0,+1]": (0, 1),
        "[0,+2]": (0, 2),
        "[-5,-1] (leak check)": (-5, -1),
        f"[+1,+{last_offset}] (to last day before earnings)": (1, last_offset),
        f"[0,+{last_offset}] (whole post-news week)": (0, last_offset),
    }


def run_main_study(
    returns: pd.DataFrame, events: dict[str, Event], ticker: str = STOCK
) -> dict[str, EventStudyResult]:
    """Run both normal-return models for one ticker around the main event."""
    day0 = events["board_investigation"].trading_day
    # Stop the data the trading day before the earnings date, so earnings never mix in.
    last_date = events["earnings"].date - pd.Timedelta(days=1)
    return {
        name: run_event_study(
            returns[ticker],
            returns[factors],
            day0,
            est_length=EST_LENGTH,
            gap=GAP,
            pre_days=PRE_DAYS,
            last_date=last_date,
        )
        for name, factors in MODELS.items()
    }


def last_post_offset(result: EventStudyResult) -> int:
    """Largest offset available in the data (before earnings)."""
    return int(result.abnormal["offset"].max())


def offset_of(result: EventStudyResult, date: pd.Timestamp) -> int:
    """Offset (trading days from day 0) of a calendar trading day."""
    return int(result.abnormal.loc[pd.Timestamp(date), "offset"])


def event_markers(events: dict[str, Event]) -> list[Event]:
    """Events drawn on the charts, in date order."""
    return sorted(events.values(), key=lambda e: e.trading_day)


def draw_event_lines(ax: plt.Axes, events: dict[str, Event], x_of=None) -> None:
    """Vertical dashed lines with short labels for each event."""
    top = ax.get_ylim()[1]
    # Events that share a trading day get one line with their labels joined.
    grouped: dict[pd.Timestamp, list[Event]] = {}
    for event in event_markers(events):
        grouped.setdefault(event.trading_day, []).append(event)
    for i, same_day in enumerate(grouped.values()):
        x = same_day[0].trading_day if x_of is None else x_of(same_day[0])
        if x is None:
            continue
        main = any(e.role == "main_event" for e in same_day)
        label = " + ".join(e.label for e in same_day)
        ax.axvline(
            x,
            color=ORANGE if main else MUTED,
            linestyle="--",
            linewidth=1.4 if main else 0.9,
            zorder=1,
        )
        ax.text(
            x,
            top,
            f" {label}",
            rotation=90,
            va="top",
            ha="left" if i % 2 else "right",
            fontsize=7.5,
            color=INK if main else INK_SECONDARY,
        )


# --- Charts -----------------------------------------------------------------------


def plot_price(prices: pd.DataFrame, events: dict[str, Event], peak: pd.Timestamp) -> None:
    """Figure: NVDA, SMH and SPY rebased to 100 at NVDA's October peak, events marked."""
    window = prices.loc["2026-09-01":]
    rebased = 100 * window / window.loc[peak]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for ticker, color, label in (
        (STOCK, BLUE, "NVDA"),
        (SECTOR, AQUA, "SMH (semiconductor ETF)"),
        (MARKET, INK_SECONDARY, "SPY (S&P 500)"),
    ):
        ax.plot(rebased.index, rebased[ticker], color=color, label=label)
    ax.axhline(100, color=MUTED, linewidth=0.8)
    ax.set_xlim(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-11-24"))
    ax.set_ylim(rebased.min().min() - 3, rebased.max().max() + 12)
    draw_event_lines(ax, events)
    ax.set_ylabel(f"Price, rebased to 100 at NVDA's peak ({peak:%b %d})")
    ax.set_xlabel("Date (2026)")
    ax.set_title("NVDA was already falling with the chip sector before the Nov 8 news")
    ax.legend(loc="lower left")
    save_fig(fig, "nvda_price_events", RESULTS)


def plot_daily_ar(result: EventStudyResult, events: dict[str, Event]) -> None:
    """Figure: daily abnormal returns (two-factor) with +/-2 sigma bands."""
    ab = result.abnormal[result.abnormal["offset"] >= -15]
    sigma = result.model.resid_std
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    colors = [ORANGE if abs(v) > 2 * sigma else BLUE for v in ab["ar"]]
    ax.bar(ab["offset"], 100 * ab["ar"], color=colors, width=0.75)
    ax.axhspan(-200 * sigma, 200 * sigma, color=MUTED, alpha=0.08, label="+/-2 sigma (normal day)")
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_ylim(min(100 * ab["ar"].min(), -200 * sigma) * 1.25, 100 * ab["ar"].max() * 1.9 + 1)

    def x_of(e: Event) -> int | None:
        return offset_of(result, e.trading_day) if e.trading_day in ab.index else None

    draw_event_lines(ax, events, x_of)
    ax.set_xlabel("Trading days from day 0 (Mon Nov 9, 2026)")
    ax.set_ylabel("Abnormal return, %")
    ax.set_title(
        f"NVDA daily abnormal returns, two-factor model (sigma = {100 * sigma:.1f}% a day). "
        "Orange: beyond 2 sigma"
    )
    ax.legend(loc="lower left")
    save_fig(fig, "nvda_daily_abnormal_returns", RESULTS)


def plot_car_paths(results: dict[str, EventStudyResult], events: dict[str, Event]) -> None:
    """Figure: CAR over time for both models, cumulated from day -10."""
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for (name, result), color in zip(results.items(), (INK_SECONDARY, BLUE), strict=True):
        ab = result.abnormal[result.abnormal["offset"] >= -10]
        path = 100 * ab["ar"].cumsum()
        ax.plot(ab["offset"], path, color=color, marker="o", markersize=4, label=name)
        ax.annotate(
            f"{path.iloc[-1]:.1f}%",
            xy=(ab["offset"].iloc[-1], path.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    first = next(iter(results.values()))
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo, max(ax.get_ylim()[1], 0) + 0.45 * (ax.get_ylim()[1] - lo))

    def x_of(e: Event) -> int | None:
        ab = first.abnormal
        if e.trading_day in ab.index and ab.loc[e.trading_day, "offset"] >= -10:
            return offset_of(first, e.trading_day)
        return None

    draw_event_lines(ax, events, x_of)
    ax.set_xlabel("Trading days from day 0 (Mon Nov 9, 2026)")
    ax.set_ylabel("Cumulative abnormal return since day -10, %")
    ax.set_title("Cumulative abnormal return: both models agree the news days were abnormal")
    ax.legend(loc="lower left")
    save_fig(fig, "nvda_car_paths", RESULTS)


def plot_peers(peer_table: pd.DataFrame, last: int) -> None:
    """Figure: CAR of NVDA vs its peers in the same windows (two-factor model)."""
    tickers = list(dict.fromkeys(peer_table["ticker"]))
    windows = ["[0]", "[0,+2]", f"[0,+{last}] (whole post-news week)"]
    labels = ["Day 0 (Nov 9)", "Days 0 to +2", f"Days 0 to +{last} (to day before earnings)"]
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for i, (window, label, color) in enumerate(
        zip(windows, labels, (BLUE, ORANGE, AQUA), strict=True)
    ):
        sub = peer_table[peer_table["window"] == window].set_index("ticker").loc[tickers]
        heights = 100 * sub["car"].to_numpy()
        bars = ax.bar(x + (i - 1) * 0.26, heights, width=0.24, color=color, label=label)
        for bar, t in zip(bars, sub["t_stat"], strict=True):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.4 if h >= 0 else -0.4),
                f"t={t:.1f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=7,
            )
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xticks(x, tickers)
    ax.set_ylabel("Cumulative abnormal return, %")
    ax.set_xlabel("Stock (two-factor model: SPY + SMH, fitted before the news)")
    ax.set_title("Only NVDA had a large negative abnormal return; peers did not fall with it")
    ax.grid(axis="x", visible=False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 1.5, hi + 0.3 * (hi - lo))
    ax.legend(loc="upper right", ncols=3)
    save_fig(fig, "nvda_peer_car", RESULTS)


def plot_placebo(placebo: pd.DataFrame, real_car: float, real_t: float) -> None:
    """Figure: placebo CARs and standardized CARs with the real values marked."""
    start, end = HEADLINE_WINDOW
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    panels = (
        ("car", 100 * real_car, "CAR over days [0,+2], %", 100.0, "%"),
        ("scar", real_t, "Standardized CAR (t-statistic)", 1.0, ""),
    )
    for ax, (column, real, xlabel, scale, unit) in zip(axes, panels, strict=True):
        values = scale * placebo[column]
        ax.hist(values, bins=50, color=BLUE, alpha=0.7, label=f"{len(values):,} random dates")
        pct = percentile_of(real / scale, placebo[column])
        ax.axvline(real, color=ORANGE, linewidth=2.2, label=f"Nov 9-11, 2026: {real:.1f}{unit}")
        ax.text(
            real,
            ax.get_ylim()[1] * 0.6,
            f"  {pct:.1f}% of random\n  dates were lower",
            ha="left",
            fontsize=9,
            color=INK,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of random dates")
        ax.legend(loc="upper right")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))
    fig.text(
        0.01,
        0.985,
        f"Placebo test: the same [{start},+{end}] CAR on 1,000 random NVDA dates since 2015",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "nvda_placebo", RESULTS)


def plot_drawdown(
    decomposition: pd.DataFrame, events: dict[str, Event], peak: pd.Timestamp
) -> None:
    """Figure: from the October peak, actual return split into normal and abnormal parts."""
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    d = 100 * decomposition
    ax.plot(d.index, d["actual"], color=INK, label="Actual NVDA return (cumulative)")
    ax.plot(d.index, d["predicted"], color=AQUA, label="Explained by SPY + SMH (normal)")
    ax.plot(d.index, d["abnormal"], color=BLUE, label="Abnormal (NVDA-specific)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.5 * (hi - lo))
    shown = {k: e for k, e in events.items() if e.trading_day <= d.index[-1]}
    draw_event_lines(ax, shown)
    ax.set_xlabel("Date (2026)")
    ax.set_ylabel(f"Cumulative return since {peak:%b %d} peak, % (sum of daily)")
    ax.set_title("From the October peak: how much of NVDA's drop was sector vs NVDA-specific")
    ax.legend(loc="lower left")
    save_fig(fig, "nvda_drawdown_decomposition", RESULTS)


# --- Analysis ---------------------------------------------------------------------


def drawdown_decomposition(
    result: EventStudyResult, prices: pd.Series, peak: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Split NVDA's return from its peak into the normal and abnormal parts.

    Daily returns are summed (not compounded) so that actual = predicted + abnormal
    holds exactly; the compounded drop is reported alongside for reference.
    """
    ab = result.abnormal.loc[result.abnormal.index > peak]
    cumulative = ab[["actual", "predicted", "ar"]].cumsum().rename(columns={"ar": "abnormal"})
    trough = prices.loc[peak:].idxmin()
    upto = ab.loc[:trough]
    event_part = upto[upto["offset"] >= 0]["ar"].sum()
    summary = {
        "peak_date": peak,
        "trough_date": trough,
        "compounded_drop": float(prices.loc[trough] / prices.loc[peak] - 1),
        "sum_of_daily_returns": float(upto["actual"].sum()),
        "explained_by_spy_smh": float(upto["predicted"].sum()),
        "abnormal_before_day0": float(upto[upto["offset"] < 0]["ar"].sum()),
        "abnormal_day0_onward": float(event_part),
    }
    summary["share_of_drop_from_event_car"] = (
        summary["abnormal_day0_onward"] / summary["sum_of_daily_returns"]
    )
    summary["share_of_drop_from_sector_market"] = (
        summary["explained_by_spy_smh"] / summary["sum_of_daily_returns"]
    )
    return cumulative, summary


# --- Earnings and robustness ---------------------------------------------------------


def sector_ex_nvda(returns: pd.DataFrame, weight: float) -> pd.Series:
    """Approximate SMH return with NVDA taken out: (SMH - w * NVDA) / (1 - w).

    NVDA is about a fifth of SMH, so on Nov 9 SMH's +5.1% already contains NVDA's own
    -4.3%. Using SMH as the "normal" benchmark therefore lets some of the news leak into
    the prediction. Removing NVDA's share gives a cleaner sector benchmark. The weight
    is held fixed over time, which is an approximation.
    """
    return (returns[SECTOR] - weight * returns[STOCK]) / (1.0 - weight)


def robustness_table(returns: pd.DataFrame, events: dict[str, Event]) -> pd.DataFrame:
    """CARs under different estimation windows and sector benchmarks."""
    day0 = events["board_investigation"].trading_day
    last_date = events["earnings"].date - pd.Timedelta(days=1)
    data = returns.copy()
    specs: list[tuple[str, list[str]]] = [
        ("Market model (SPY)", [MARKET]),
        ("Two-factor (SPY + SMH)", [MARKET, SECTOR]),
    ]
    for w in NVDA_WEIGHTS_IN_SMH:
        col = f"SMH_ex_NVDA_{int(100 * w)}"
        data[col] = sector_ex_nvda(returns, w)
        specs.append((f"Two-factor, SMH without NVDA (w = {w:.0%})", [MARKET, col]))

    rows = []
    for est_length, gap in ROBUST_WINDOWS:
        for name, factors in specs:
            # The NVDA-weight sensitivity is shown for the baseline window only.
            baseline = (est_length, gap) == ROBUST_WINDOWS[0]
            if "w = " in name and not baseline and "20%" not in name:
                continue
            res = run_event_study(data[STOCK], data[factors], day0, est_length, gap,
                                  pre_days=PRE_DAYS, last_date=last_date)  # fmt: skip
            last = last_post_offset(res)
            rows.append(
                {
                    "estimation": f"{est_length} days, ending {gap} days before",
                    "model": name,
                    "car_day0": res.car(0, 0),
                    "t_day0": res.t_stat(0, 0),
                    "car_0_2": res.car(0, 2),
                    "t_0_2": res.t_stat(0, 2),
                    f"car_0_{last}": res.car(0, last),
                    f"t_0_{last}": res.t_stat(0, last),
                }
            )
    return pd.DataFrame(rows)


def earnings_study(returns: pd.DataFrame, events: dict[str, Event]) -> pd.DataFrame:
    """Earnings as a separate event: day 0 = Nov 18, the first day after the release.

    The estimation window is kept identical to the main study (same end date), so the
    scandal days never leak into "normal". Only day 0 is reported as clean: Nov 20 was a
    market-wide AI/chip sell-off (see events.yaml), so longer windows are shown only as
    a caution.
    """
    day0 = events["board_investigation"].trading_day
    earn = events["earnings"].trading_day
    index = returns.index
    extra = int(index.searchsorted(earn) - index.searchsorted(day0))
    rows = []
    for name, factors in MODELS.items():
        res = run_event_study(returns[STOCK], returns[factors], earn, EST_LENGTH, GAP + extra,
                              pre_days=5)  # fmt: skip
        last = last_post_offset(res)
        windows = {"[0] Nov 18 (clean)": (0, 0), f"[0,+{last}] incl. Nov 20 sell-off": (0, last)}
        for window, (a, b) in windows.items():
            if b > last:
                continue
            sel = res.abnormal[(res.abnormal["offset"] >= a) & (res.abnormal["offset"] <= b)]
            rows.append(
                {
                    "model": name,
                    "window": window,
                    "actual": float(sel["actual"].sum()),
                    "predicted": float(sel["predicted"].sum()),
                    "car": res.car(a, b),
                    "t_stat": res.t_stat(a, b),
                }
            )
    return pd.DataFrame(rows)


def main() -> dict[str, object]:
    """Run Phase 1 and return the headline numbers."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()
    events = load_events()
    returns, volume = load_returns()
    prices = get_prices([STOCK, MARKET, SECTOR, *PEERS], START, END)

    # 1. Main event study, both models.
    results = run_main_study(returns, events)
    last = last_post_offset(results[TWO_FACTOR])
    windows = event_windows(results[TWO_FACTOR], last)
    car = pd.concat(
        [car_table(r, windows).assign(model=name) for name, r in results.items()],
        ignore_index=True,
    )[["model", "window", "days", "actual", "predicted", "car", "t_stat"]]
    save_table(car, "nvda_car", float_format="{:.4f}", results_dir=RESULTS)

    fit = pd.DataFrame(
        [
            {
                "model": name,
                "alpha_daily": r.model.alpha,
                **{f"beta_{k}": v for k, v in r.model.betas.items()},
                "resid_std_daily": r.model.resid_std,
                "r_squared": r.model.r_squared,
                "n_obs": r.model.n_obs,
            }
            for name, r in results.items()
        ]
    )
    save_table(fit, "nvda_model_fit", float_format="{:.4f}", results_dir=RESULTS)

    # Daily ARs for the in-window events (each news day on its own).
    two = results[TWO_FACTOR]
    daily = two.abnormal[two.abnormal["offset"] >= -5].copy()
    daily["date"] = daily.index.date
    # Several events can share a trading day (Nov 9 has two), so join their labels.
    labels: dict[object, list[str]] = {}
    for e in event_markers(events):
        labels.setdefault(e.trading_day.date(), []).append(e.label)
    daily["news"] = ["; ".join(labels.get(d, [])) for d in daily["date"]]
    daily["t_stat"] = daily["ar"] / two.model.resid_std
    save_table(
        daily[["date", "offset", "actual", "predicted", "ar", "t_stat", "news"]],
        "nvda_daily_ar",
        results_dir=RESULTS,
    )

    # 2. The earlier NYT report (Oct 29), as its own small event study.
    prior = {
        name: run_event_study(
            returns[STOCK],
            returns[f],
            events["nyt_report"].trading_day,
            est_length=EST_LENGTH,
            gap=GAP,
            pre_days=5,
            last_date=events["house_letter"].trading_day - pd.Timedelta(days=1),
        )
        for name, f in MODELS.items()
    }
    prior_table = pd.concat(
        [
            car_table(r, {"[0]": (0, 0), "[0,+1]": (0, 1)}).assign(model=name)
            for name, r in prior.items()
        ],
        ignore_index=True,
    )[["model", "window", "actual", "predicted", "car", "t_stat"]]
    save_table(prior_table, "nvda_prior_event_car", float_format="{:.4f}", results_dir=RESULTS)

    # 3. Placebo: the same CAR on 1,000 random non-event dates.
    exclude = [e.trading_day for e in events.values() if e.trading_day <= returns.index[-1]]
    placebo = placebo_cars(
        returns[STOCK],
        returns[MODELS[TWO_FACTOR]],
        HEADLINE_WINDOW,
        n_draws=PLACEBO_DRAWS,
        exclude=exclude,
        est_length=EST_LENGTH,
        gap=GAP,
        seed=PLACEBO_SEED,
    )
    real_car = two.car(*HEADLINE_WINDOW)
    real_t = two.t_stat(*HEADLINE_WINDOW)
    placebo_summary = pd.DataFrame(
        {
            "statistic": ["CAR [0,+2]", "Standardized CAR [0,+2]"],
            "real_value": [real_car, real_t],
            "placebo_mean": [placebo["car"].mean(), placebo["scar"].mean()],
            "placebo_1st_percentile": [
                placebo["car"].quantile(0.01),
                placebo["scar"].quantile(0.01),
            ],
            "percent_of_placebo_at_or_below_real": [
                percentile_of(real_car, placebo["car"]),
                percentile_of(real_t, placebo["scar"]),
            ],
        }
    )
    save_table(placebo_summary, "nvda_placebo_summary", float_format="{:.4f}", results_dir=RESULTS)

    # 4. Peers: same models, same windows.
    peer_rows = []
    for ticker in [STOCK, *PEERS]:
        res = run_main_study(returns, events, ticker)[TWO_FACTOR]
        peer_rows.append(car_table(res, windows).assign(ticker=ticker))
    peer_table = pd.concat(peer_rows, ignore_index=True)[
        ["ticker", "window", "actual", "predicted", "car", "t_stat"]
    ]
    save_table(peer_table, "nvda_peer_car", float_format="{:.4f}", results_dir=RESULTS)

    # 5. Volume.
    vol = abnormal_volume(volume, events["board_investigation"].trading_day, range(-5, last + 1))
    vol["date"] = vol["date"].dt.date
    save_table(vol, "nvda_abnormal_volume", float_format="{:.2f}", results_dir=RESULTS)

    # 6. The slide that started before the news.
    peak = prices[STOCK].loc["2026-10-01":"2026-11-08"].idxmax()
    # Stop at the last pre-earnings day so earnings never mix into the decomposition.
    pre_earnings = prices[STOCK].loc[: two.abnormal.index[-1]]
    cumulative, drawdown = drawdown_decomposition(two, pre_earnings, peak)
    save_table(
        pd.DataFrame([drawdown]).astype({"peak_date": str, "trough_date": str}),
        "nvda_drawdown",
        float_format="{:.4f}",
        results_dir=RESULTS,
    )

    # 7. Robustness and the separate earnings event.
    robust = robustness_table(returns, events)
    save_table(robust, "nvda_robustness", float_format="{:.4f}", results_dir=RESULTS)
    earnings = earnings_study(returns, events)
    save_table(earnings, "nvda_earnings_car", float_format="{:.4f}", results_dir=RESULTS)
    logger.info("Robustness:\n%s", robust.round(4).to_string(index=False))
    logger.info("Earnings (separate event):\n%s", earnings.round(4).to_string(index=False))

    # Charts.
    plot_price(prices, events, peak)
    plot_daily_ar(two, events)
    plot_car_paths(results, events)
    plot_peers(peer_table, last)
    plot_placebo(placebo, real_car, real_t)
    plot_drawdown(cumulative, events, peak)

    logger.info("Model fit:\n%s", fit.round(4).to_string(index=False))
    logger.info("CAR table:\n%s", car.round(4).to_string(index=False))
    logger.info("Prior event (Oct 29 NYT report):\n%s", prior_table.round(4).to_string(index=False))
    logger.info("Placebo:\n%s", placebo_summary.round(4).to_string(index=False))
    logger.info("Peers:\n%s", peer_table.round(4).to_string(index=False))
    logger.info("Volume:\n%s", vol.round(2).to_string(index=False))
    logger.info("Drawdown:\n%s", pd.Series(drawdown).to_string())
    logger.info(
        "Daily ARs:\n%s",
        daily[["offset", "actual", "predicted", "ar", "t_stat", "news"]].round(4).to_string(),
    )
    return {"car": car, "placebo": placebo_summary, "peers": peer_table, "drawdown": drawdown}


if __name__ == "__main__":
    main()
