"""Phase 2 of the NVDA case study: the repo's four models applied to the event.

Each section reuses a tested model from ``src/qml`` and answers one plain question:

1. Marchenko-Pastur: did the market's correlation structure change?
2. Kalman filter: did NVDA decouple from its sector?
3. Almgren-Chriss: what did it cost to sell a large block of NVDA?
4. Avellaneda-Stoikov: what happened to a market maker quoting NVDA?

Every number that is an assumption rather than a measurement is a named constant
below, with the reason for its value.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from event_study import (
    END,
    RESULTS,
    SECTOR,
    STOCK,
    draw_event_lines,
    load_events,
)

from qml.common.data import clean_returns, get_prices
from qml.common.plotting import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    apply_style,
    save_fig,
    save_table,
)
from qml.execution.almgren_chriss import (
    ACParams,
    expected_cost_and_variance,
    kappa_continuous,
    optimal_trajectory,
)
from qml.market_making.simulator import ASParams, simulate
from qml.market_making.strategies import InventoryStrategy, SymmetricStrategy, optimal_spread
from qml.pairs.kalman import fit_obs_var, rolling_ols, run_filter
from qml.rmt.marchenko_pastur import (
    correlation_matrix,
    count_signal_eigenvalues,
    fit_sigma2,
    mp_bounds,
    sorted_eigh,
)

logger = logging.getLogger("nvda_models")

DAY0 = pd.Timestamp("2026-11-09")  # first trading day after the Nov 8 announcement
PEAK = pd.Timestamp("2026-10-06")  # NVDA's October closing high (found in Phase 1)
CALM_END = pd.Timestamp("2026-10-05")  # last day before the slide: "calm" windows end here
LAST = pd.Timestamp("2026-11-13")  # last trading day before earnings in the data

# --- 1. Marchenko-Pastur ------------------------------------------------------------
# About 130 large US-listed tech, internet and semiconductor names. Symbols that no
# longer trade, or have more than 5% missing data, are dropped automatically.
TECH_UNIVERSE = (  # noqa: SIM905  (a word list reads better than 130 quoted strings)
    "NVDA AMD AVGO TSM MU INTC QCOM TXN ADI AMAT LRCX KLAC ASML MRVL NXPI MCHP ON MPWR "
    "SWKS QRVO TER ENTG ARM SMCI ALAB COHR LSCC WOLF AMKR MTSI CRUS SYNA SITM FORM ONTO "
    "AAPL MSFT GOOGL META AMZN ORCL CRM ADBE NOW INTU IBM CSCO ACN ANET DELL HPQ HPE "
    "PANW CRWD FTNT ZS SNOW DDOG PLTR MDB NET TEAM WDAY ADSK CDNS SNPS KEYS GLW APH TEL "
    "MSI ZBRA NTAP STX WDC PSTG UBER ABNB SHOP SPOT NFLX TTWO ROP FICO IT CDW AKAM FFIV "
    "GDDY TYL PTC GEN EPAM COIN HOOD APP AXON TTD DOCU ZM OKTA TWLO PINS SNAP DASH RBLX "
    "U MSTR VRT CIEN JBL FLEX TSLA NOW GRMN TRMB VRSN CHKP CYBR MANH GWRE DT ESTC PATH "
    "S BILL HUBS CFLT GTLB IOT NTNX SMAR"
).split()
N_RMT_STOCKS = 100  # first 100 that survive cleaning: q = 100 / 126 = 0.79
RMT_START = "2025-01-01"
CALM_DAYS = 126  # six months of trading days, ending CALM_END
ROLLING_DAYS = 25  # short window for the rolling view (about five weeks)

# --- 2. Kalman ----------------------------------------------------------------------
KALMAN_START = "2023-01-02"
KALMAN_TRAIN_END = pd.Timestamp("2026-09-25")  # ~30 trading days before day 0, like Phase 1
DELTA_GRID = (1e-7, 1e-6, 1e-5, 1e-4)
OLS_WINDOW = 60

# --- 3. Almgren-Chriss --------------------------------------------------------------
SHARES_TO_SELL = 2_000_000
DAYS_TO_SELL = 3
BUCKETS_PER_DAY = 13  # half-hour slices of a 6.5-hour session
# Square-root impact rule of thumb: selling at speed v (shares/day) into a market that
# trades V shares/day costs about Y * sigma_daily * sqrt(v / V) per share. Y = 1 is the
# usual order of magnitude; it is an assumption here, not something we measured.
IMPACT_Y = 1.0
# The repo's closed-form model needs *linear* impact, so we pick eta so that the linear
# model charges exactly the square-root cost when selling at the constant (TWAP) speed.
PERMANENT_SHARE = 0.5  # assumption: half of the TWAP-speed impact is permanent
HALF_SPREAD = 0.005  # assumption: NVDA's quoted spread is about one cent
AC_LAMBDAS = (3e-9, 1e-8, 3e-8)  # risk aversion per dollar; 0 would be plain TWAP

# --- 4. Avellaneda-Stoikov ----------------------------------------------------------
# The simulator works in its own units: price 100 and one "session" = one trading day.
# Rescaling NVDA to a $100 price makes sigma = daily volatility in percent, so a 2%
# daily vol becomes sigma = 2 (the paper's default). k, A and gamma stay at the
# paper's values: we have no NVDA order-book data to calibrate them.
AS_PATHS, AS_SEED = 1000, 7


# =====================================================================================
# 1. Marchenko-Pastur: did the market's structure change?
# =====================================================================================


def top_eigen_stats(returns: pd.DataFrame) -> dict[str, float]:
    """Largest eigenvalue, its share of variance and NVDA's loading for one window."""
    corr = correlation_matrix(returns.to_numpy())
    values, vectors = sorted_eigh(corr)
    nvda = returns.columns.get_loc(STOCK)
    n_obs, n_assets = returns.shape
    # For standardized returns, lambda_1 * v_i^2 is the share of stock i's own variance
    # explained by the top factor; averaged over all stocks it equals lambda_1 / N.
    explained = values[0] * vectors[:, 0] ** 2
    stats = {
        "days": n_obs,
        "stocks": n_assets,
        "q": n_assets / n_obs,
        "largest_eigenvalue": float(values[0]),
        "top_share_of_variance": float(values[0] / n_assets),
        "nvda_variance_explained_by_top_factor": float(explained[nvda]),
    }
    # The noise band only exists when there are more days than stocks (q < 1).
    if n_assets < n_obs:
        sigma2 = fit_sigma2(values, n_assets / n_obs)
        stats["sigma2"] = sigma2
        stats["lambda_plus"] = mp_bounds(n_assets / n_obs, sigma2)[1]
        stats["eigenvalues_above_lambda_plus"] = count_signal_eigenvalues(
            values, n_assets / n_obs, sigma2
        )
    return stats


def run_rmt(events: dict) -> pd.DataFrame:
    """Compare the correlation structure before the slide and around the news."""
    prices = get_prices(TECH_UNIVERSE, RMT_START, END)
    returns = clean_returns(prices, max_missing=0.05).iloc[:, :N_RMT_STOCKS].loc[:LAST]
    logger.info("RMT universe: %d stocks", returns.shape[1])

    calm = returns.loc[:CALM_END].iloc[-CALM_DAYS:]
    stressed = returns.loc[PEAK:LAST]
    rows = [
        {"window": f"Calm: {CALM_DAYS} days to {CALM_END:%b %d}", **top_eigen_stats(calm)},
        {"window": f"Slide + news: {PEAK:%b %d} to {LAST:%b %d}", **top_eigen_stats(stressed)},
    ]
    # A short window mechanically inflates the top eigenvalue (fewer days, more noise),
    # so the fair yardstick for the 29-day stressed window is the average 29-day window
    # inside the calm period, not the 126-day calm window itself.
    n = len(stressed)
    same_length = [top_eigen_stats(calm.iloc[i : i + n]) for i in range(len(calm) - n + 1)]
    rows.append(
        {"window": f"Calm, average {n}-day window", **pd.DataFrame(same_length).mean().to_dict()}
    )
    table = pd.DataFrame(rows)
    save_table(table, "rmt_structure", float_format="{:.3f}", results_dir=RESULTS)

    # Rolling view: the same two statistics in every 25-day window through 2026.
    rows = []
    for end in range(ROLLING_DAYS, len(returns) + 1):
        win = returns.iloc[end - ROLLING_DAYS : end]
        if win.index[-1] < pd.Timestamp("2026-01-02"):
            continue
        s = top_eigen_stats(win)
        rows.append({"date": win.index[-1], **s})
    rolling = pd.DataFrame(rows).set_index("date")
    nvda_share = rolling["nvda_variance_explained_by_top_factor"]
    summary = pd.DataFrame(
        {
            "statistic": [
                "NVDA share, highest 25-day window before day 0",
                "NVDA share, 25-day window ending the day before day 0",
                f"NVDA share, 25-day window ending {LAST:%b %d}",
                f"Average stock share, 25-day window ending {LAST:%b %d}",
            ],
            "value": [
                float(nvda_share.loc[: DAY0 - pd.Timedelta(days=1)].max()),
                float(nvda_share.loc[: DAY0 - pd.Timedelta(days=1)].iloc[-1]),
                float(nvda_share.iloc[-1]),
                float(rolling["top_share_of_variance"].iloc[-1]),
            ],
        }
    )
    save_table(summary, "rmt_rolling_summary", float_format="{:.3f}", results_dir=RESULTS)
    logger.info("Rolling RMT summary:\n%s", summary.to_string(index=False))
    plot_rmt(rolling, events)
    return table


def plot_rmt(rolling: pd.DataFrame, events: dict) -> None:
    """Figure: rolling top-eigenvalue share and NVDA's loading, events marked."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.0, 6.6), sharex=True)
    top.plot(rolling.index, 100 * rolling["top_share_of_variance"], color=BLUE)
    top.set_ylabel("Variance explained by\nthe top eigenvalue, %")
    top.set_title(
        f"Rolling {ROLLING_DAYS}-day correlation structure of {int(rolling['stocks'].iloc[-1])} "
        "tech and chip stocks"
    )
    lo, hi = top.get_ylim()
    top.set_ylim(lo, hi + 0.35 * (hi - lo))
    main_only = {k: e for k, e in events.items() if e.role == "main_event"}
    draw_event_lines(top, main_only)

    bottom.plot(rolling.index, 100 * rolling["nvda_variance_explained_by_top_factor"],
                color=ORANGE, label="NVDA")  # fmt: skip
    bottom.plot(rolling.index, 100 * rolling["top_share_of_variance"], color=AQUA,
                label="Average stock")  # fmt: skip
    bottom.set_ylabel("Own variance explained\nby the top factor, %")
    bottom.set_xlabel(f"End date of each {ROLLING_DAYS}-trading-day window")
    bottom.set_title("How much a stock moves with the common factor (lower = moving on its own)")
    bottom.legend(loc="upper left")
    for e in events.values():
        if e.role == "main_event":
            bottom.axvline(e.trading_day, color=ORANGE, linestyle="--", linewidth=1.2)
    save_fig(fig, "rmt_rolling_structure", RESULTS)


# =====================================================================================
# 2. Kalman filter: did NVDA decouple from its sector?
# =====================================================================================


def kalman_log_likelihood(x: pd.Series, y: pd.Series, delta: float, obs_var: float) -> float:
    """Gaussian log-likelihood of the filter's one-step prediction errors (after burn-in)."""
    out = run_filter(x, y, delta, obs_var).iloc[60:]
    var = out["error_std"].to_numpy() ** 2
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + out["error"].to_numpy() ** 2 / var))


def choose_kalman_params(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Pick delta and R by maximum likelihood on the training window only."""
    best = None
    for delta in DELTA_GRID:
        obs_var = fit_obs_var(x, y, delta)
        ll = kalman_log_likelihood(x, y, delta, obs_var)
        if best is None or ll > best[2]:
            best = (delta, obs_var, ll)
    assert best is not None
    return best[0], best[1]


def reaction(frame: pd.DataFrame) -> dict[str, float]:
    """How each method's hedge ratio and z-score behaved over the news days."""
    before = float(frame["beta"].loc[:DAY0].iloc[-2])  # close of the last pre-news day
    news = frame.loc[DAY0:LAST]
    return {
        "beta_before": before,
        "beta_nov13": float(news["beta"].iloc[-1]),
        "beta_change": float(news["beta"].iloc[-1] - before),
        "z_on_nov9": float(news["z"].iloc[0]),
        "min_z_nov9_to_13": float(news["z"].min()),
        # A method that keeps flagging |z| > 2 is still "surprised"; one that drops back
        # inside the band has absorbed the jump into its estimate of the relationship.
        "days_beyond_2_sigma_nov9_to_13": int((news["z"].abs() > 2).sum()),
    }


def run_kalman(events: dict) -> pd.DataFrame:
    """Kalman and rolling-OLS hedge ratios of NVDA against SMH and against AMD."""
    prices = get_prices([STOCK, SECTOR, "AMD"], KALMAN_START, END).dropna()
    log_p = np.log(prices)
    rows, panels = [], []
    for x_name in (SECTOR, "AMD"):
        x, y = log_p[x_name], log_p[STOCK]
        delta, obs_var = choose_kalman_params(x.loc[:KALMAN_TRAIN_END], y.loc[:KALMAN_TRAIN_END])
        kalman = run_filter(x, y, delta, obs_var)
        ols = rolling_ols(x, y, OLS_WINDOW)
        for method, frame in (("Kalman", kalman), (f"Rolling OLS ({OLS_WINDOW}d)", ols)):
            rows.append(
                {
                    "pair": f"NVDA vs {x_name}",
                    "method": method,
                    "delta": delta if method == "Kalman" else np.nan,
                    **reaction(frame),
                }
            )
        panels.append((x_name, kalman, ols, delta))
    table = pd.DataFrame(rows)
    save_table(table, "kalman_reaction", float_format="{:.4g}", results_dir=RESULTS)
    plot_kalman(panels, events)
    return table


def plot_kalman(panels: list, events: dict) -> None:
    """Figure: hedge ratio and live z-score around the event, both pairs."""
    shown = {k: e for k, e in events.items() if e.trading_day <= LAST}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    start = pd.Timestamp("2026-08-01")
    for col, (x_name, kalman, ols, delta) in enumerate(panels):
        ax_b, ax_z = axes[0, col], axes[1, col]
        k, o = kalman.loc[start:], ols.loc[start:]
        ax_b.plot(o.index, o["beta"], color=ORANGE, linewidth=1.2, label="Rolling OLS (60d)")
        ax_b.plot(k.index, k["beta"], color=BLUE, label=f"Kalman (delta = {delta:.0e})")
        ax_b.set_title(f"NVDA vs {x_name}: hedge ratio (log prices)")
        ax_b.set_ylabel("Hedge ratio beta")
        ax_b.legend(loc="upper left")
        lo, hi = ax_b.get_ylim()
        ax_b.set_ylim(lo, hi + 0.4 * (hi - lo))
        draw_event_lines(ax_b, {k: e for k, e in shown.items() if e.role == "main_event"})

        ax_z.plot(o.index, o["z"], color=ORANGE, linewidth=1.2, label="Rolling OLS z-score")
        ax_z.plot(k.index, k["z"], color=BLUE, label="Kalman z-score")
        for level in (-2, 2):
            ax_z.axhline(level, color=MUTED, linestyle="--", linewidth=0.9)
        ax_z.set_title(f"NVDA vs {x_name}: live z-score of the spread")
        ax_z.set_ylabel("z-score (error / expected size)")
        ax_z.set_xlabel("Date (2026)")
        ax_z.legend(loc="lower left")
        for e in shown.values():
            if e.role == "main_event":
                ax_z.axvline(e.trading_day, color=ORANGE, linestyle="--", linewidth=1.2)
        ax_z.tick_params(axis="x", labelrotation=30)
    save_fig(fig, "kalman_decoupling", RESULTS)


# =====================================================================================
# 3. Almgren-Chriss: what did it cost to sell 2 million shares?
# =====================================================================================


def market_snapshot(close: pd.Series, volume: pd.Series, start, end) -> dict[str, float]:
    """Price, daily volatility and average volume over a window."""
    rets = close.pct_change().loc[start:end]
    return {
        "price": float(close.loc[:end].iloc[-1]),
        # Root-mean-square return: with only a few days, assuming a zero average move is
        # more stable than estimating the mean too.
        "daily_vol": float(np.sqrt(np.mean(rets**2))),
        "adv": float(volume.loc[start:end].mean()),
    }


def ac_params(snap: dict[str, float]) -> ACParams:
    """Translate a market snapshot into Almgren-Chriss parameters (assumptions above)."""
    sigma_dollars = snap["daily_vol"] * snap["price"]
    twap_speed = SHARES_TO_SELL / DAYS_TO_SELL
    impact_at_twap = IMPACT_Y * sigma_dollars * np.sqrt(twap_speed / snap["adv"])
    eta = (1 - PERMANENT_SHARE) * impact_at_twap / twap_speed
    gamma = PERMANENT_SHARE * impact_at_twap / SHARES_TO_SELL
    return ACParams(
        X=SHARES_TO_SELL,
        T=DAYS_TO_SELL,
        n_steps=DAYS_TO_SELL * BUCKETS_PER_DAY,
        s0=snap["price"],
        sigma=sigma_dollars,
        eta=eta,
        gamma=gamma,
        epsilon=HALF_SPREAD,
        daily_volume=snap["adv"],
    )


def run_almgren_chriss() -> pd.DataFrame:
    """Optimal 3-day schedules and costs, calm market vs event week."""
    data = get_prices([STOCK], "2026-01-01", END)[STOCK]
    volume = get_prices([STOCK], "2026-01-01", END, field="Volume")[STOCK]
    calm_start = data.loc[:CALM_END].index[-60]
    regimes = {
        "Calm (60 days to Oct 5)": market_snapshot(data, volume, calm_start, CALM_END),
        "Event week (Nov 9-13)": market_snapshot(data, volume, DAY0, LAST),
    }
    rows, schedules = [], {}
    for regime, snap in regimes.items():
        p = ac_params(snap)
        for lam in (0.0, *AC_LAMBDAS):
            traj = optimal_trajectory(p.X, p.T, p.n_steps, p.sigma, p.eta, p.gamma, lam)
            cost, var = expected_cost_and_variance(traj, p)
            notional = p.X * p.s0
            rows.append(
                {
                    "regime": regime,
                    "lambda": lam,
                    "price": p.s0,
                    "daily_vol_pct": 100 * snap["daily_vol"],
                    "adv_millions": snap["adv"] / 1e6,
                    "kappa_per_day": kappa_continuous(p.sigma, p.eta, lam) if lam else 0.0,
                    "sold_after_day_1_pct": 100 * (1 - traj[BUCKETS_PER_DAY] / p.X),
                    "expected_cost_bps": 1e4 * cost / notional,
                    "std_cost_bps": 1e4 * np.sqrt(var) / notional,
                }
            )
            schedules[(regime, lam)] = (p.times, traj)
    table = pd.DataFrame(rows)
    save_table(table, "almgren_chriss_event", float_format="{:.4g}", results_dir=RESULTS)
    params = pd.DataFrame(
        [{"regime": r, **{k: f"{v:.4g}" for k, v in vars(ac_params(s)).items()}}
         for r, s in regimes.items()]
    )  # fmt: skip
    save_table(params, "almgren_chriss_params", results_dir=RESULTS)
    plot_almgren_chriss(schedules, table)
    return table


def plot_almgren_chriss(schedules: dict, table: pd.DataFrame) -> None:
    """Figure: schedules calm vs event, and expected cost vs risk."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    regimes = list(dict.fromkeys(table["regime"]))
    shades = dict(zip(AC_LAMBDAS, ("#86b6ef", "#2a78d6", "#0d366b"), strict=True))
    for regime, style in zip(regimes, ("--", "-"), strict=True):
        for lam in AC_LAMBDAS:
            t, x = schedules[(regime, lam)]
            label = f"lambda {lam:.0e}, {regime.split(' (')[0].lower()}"
            left.plot(t, x / 1e6, color=shades[lam], linestyle=style, label=label)
    t, x = schedules[(regimes[0], 0.0)]
    left.plot(t, x / 1e6, color=INK_SECONDARY, linestyle=":", label="TWAP (lambda = 0)")
    left.set_xlabel("Time (trading days)")
    left.set_ylabel("Shares still held (millions)")
    left.set_title("Same risk aversion, event-week volatility: sell faster")
    left.legend(loc="upper right", fontsize=7.5)

    for regime, color in zip(regimes, (AQUA, ORANGE), strict=True):
        sub = table[table["regime"] == regime]
        right.plot(sub["std_cost_bps"], sub["expected_cost_bps"], color=color, marker="o",
                   label=regime)  # fmt: skip
        for _, row in sub.iterrows():
            tag = "TWAP" if row["lambda"] == 0 else f"{row['lambda']:.0e}"
            right.annotate(tag, (row["std_cost_bps"], row["expected_cost_bps"]),
                           xytext=(5, 4), textcoords="offset points", fontsize=8)  # fmt: skip
    right.set_xlabel("Risk: std of cost, basis points of the sale")
    right.set_ylabel("Expected cost, basis points of the sale")
    right.set_title("Cost vs risk of each schedule (labels = lambda)")
    right.legend(loc="upper right")
    save_fig(fig, "almgren_chriss_event", RESULTS)


# =====================================================================================
# 4. Avellaneda-Stoikov: what happened to a market maker?
# =====================================================================================


def run_avellaneda_stoikov(daily_vol: dict[str, float]) -> pd.DataFrame:
    """Simulate 1,000 sessions per regime and strategy, NVDA volatility rescaled."""
    rows, results = [], {}
    for regime, vol in daily_vol.items():
        params = replace(ASParams(), sigma=100 * vol)  # price 100 -> sigma = vol in %
        for strategy in (InventoryStrategy(), SymmetricStrategy()):
            res = simulate(strategy, params, AS_PATHS, AS_SEED)
            results[(regime, strategy.name)] = res
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy.name,
                    "sigma_sim_units": params.sigma,
                    "mean_spread": res.mean_spread,
                    "spread_at_open": optimal_spread(0.0, params),
                    "mean_profit": float(res.pnl.mean()),
                    "std_profit": float(res.pnl.std(ddof=1)),
                    "std_final_inventory": float(res.final_inventory.std(ddof=1)),
                    "pct_sessions_losing": float(100 * np.mean(res.pnl < 0)),
                }
            )
    table = pd.DataFrame(rows)
    save_table(table, "avellaneda_stoikov_event", float_format="{:.2f}", results_dir=RESULTS)
    plot_avellaneda_stoikov(results, daily_vol)
    return table


def plot_avellaneda_stoikov(results: dict, daily_vol: dict[str, float]) -> None:
    """Figure: quoted spread through the session, and the profit distribution."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for (regime, vol), color in zip(daily_vol.items(), (AQUA, ORANGE), strict=True):
        params = replace(ASParams(), sigma=100 * vol)
        t = np.linspace(0, params.T, 101)
        left.plot(t, [optimal_spread(x, params) for x in t], color=color,
                  label=f"{regime} (daily vol {100 * vol:.1f}%)")  # fmt: skip
        pnl = results[(regime, "Inventory")].pnl
        right.hist(pnl, bins=45, color=color, alpha=0.55,
                   label=f"{regime}: mean {pnl.mean():.0f}, std {pnl.std(ddof=1):.0f}")  # fmt: skip
    left.set_xlabel("Time through the trading day (fraction)")
    left.set_ylabel("Optimal bid-ask spread (price 100 scale)")
    left.set_title("Higher volatility: the model quotes wider, most of all early")
    left.legend(loc="upper right")
    right.axvline(0, color=INK_SECONDARY, linewidth=0.9)
    right.set_xlabel("Profit per session (simulator units)")
    right.set_ylabel(f"Number of sessions (of {AS_PATHS:,})")
    right.set_title("Inventory strategy: lower profit, similar spread of outcomes")
    right.legend(loc="upper left")
    lo, hi = right.get_ylim()
    right.set_ylim(lo, hi * 1.3)
    save_fig(fig, "avellaneda_stoikov_event", RESULTS)


def main() -> dict[str, pd.DataFrame]:
    """Run all four sections and write tables and figures to results/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply_style()
    events = load_events()

    rmt = run_rmt(events)
    kalman = run_kalman(events)
    ac = run_almgren_chriss()
    # Feed the same calm and event-week volatilities into the market-making simulator.
    vols = (ac.groupby("regime", sort=False)["daily_vol_pct"].first() / 100).to_dict()
    as_table = run_avellaneda_stoikov(vols)

    for name, table in (("RMT", rmt), ("Kalman", kalman), ("Almgren-Chriss", ac),
                        ("Avellaneda-Stoikov", as_table)):  # fmt: skip
        logger.info("%s:\n%s", name, table.round(3).to_string(index=False))
    return {"rmt": rmt, "kalman": kalman, "almgren_chriss": ac, "avellaneda_stoikov": as_table}


if __name__ == "__main__":
    main()
