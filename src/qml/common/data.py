"""Price data: download from Yahoo Finance once, then read from a local cache.

Why cache? Downloading 300 tickers takes a while and Yahoo rate-limits heavy users.
Saving the answer to ``data/cache/`` makes re-runs instant and keeps results stable
from one run to the next.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from qml.common import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR: Path = DATA_DIR / "cache"
_BATCH_SIZE = 100  # tickers per Yahoo request; smaller batches fail less often


def load_universe(path: Path | str) -> list[str]:
    """Read a list of tickers from a text file.

    Parameters
    ----------
    path : Path or str
        Text file with one ticker per line. Blank lines and lines starting
        with ``#`` are ignored.

    Returns
    -------
    list of str
        Tickers in file order, upper-cased, without duplicates.
    """
    tickers: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        ticker = line.strip().upper()
        if ticker and not ticker.startswith("#") and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _cache_path(tickers: Sequence[str], start: str, end: str, cache_dir: Path) -> Path:
    """Build a cache file name that is unique to this exact request."""
    # A short hash of the request means "same tickers + same dates" always maps to the
    # same file, while any change (one extra ticker, a new end date) gets a new file.
    key = ",".join(sorted(tickers)) + f"|{start}|{end}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return cache_dir / f"prices_{start}_{end}_{len(tickers)}tickers_{digest}.csv"


def _download(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance, a batch at a time."""
    # Imported here (not at the top) so the unit tests, which never touch the
    # network, do not need yfinance to be importable.
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = list(tickers[i : i + _BATCH_SIZE])
        # auto_adjust=True folds dividends and splits into "Close", so a 2-for-1 split
        # does not look like a 50% crash in our returns.
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            continue
        close = raw["Close"]
        if isinstance(close, pd.Series):  # a single ticker can come back as a Series
            close = close.to_frame(batch[0])
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices.sort_index()


def get_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_dir: Path | str = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Get daily adjusted closing prices, using the local cache when possible.

    Parameters
    ----------
    tickers : sequence of str
        Yahoo Finance symbols, e.g. ``["AAPL", "MSFT"]``.
    start, end : str
        Date range as ``"YYYY-MM-DD"``. ``end`` is exclusive (Yahoo's convention).
    cache_dir : Path or str, optional
        Folder for cached CSV files. Defaults to ``data/cache/``.
    refresh : bool, optional
        If True, ignore any cached file and download again.

    Returns
    -------
    pd.DataFrame
        One row per trading day, one column per ticker that returned data.
        Tickers that failed to download are dropped and reported in a warning.

    Raises
    ------
    RuntimeError
        If no ticker at all returned data (usually a network problem).
    """
    tickers = list(dict.fromkeys(t.upper() for t in tickers))  # de-duplicate, keep order
    cache_dir = Path(cache_dir)
    path = _cache_path(tickers, start, end, cache_dir)

    if path.exists() and not refresh:
        logger.info("Loading prices from cache: %s", path.name)
        return pd.read_csv(path, index_col="date", parse_dates=True)

    logger.info("Downloading %d tickers from Yahoo Finance (%s to %s)", len(tickers), start, end)
    prices = _download(tickers, start, end)

    # A delisted or misspelled ticker comes back as a column of NaN (or not at all).
    # Dropping it with a warning beats crashing a 300-stock run over one bad symbol.
    prices = prices.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("Dropped %d tickers with no data: %s", len(missing), ", ".join(missing))
    if prices.empty:
        raise RuntimeError("No price data downloaded. Check your internet connection.")

    prices = prices[[t for t in tickers if t in prices.columns]]  # keep the requested order
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


def clean_returns(prices: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Turn prices into daily simple returns, dropping tickers with too many gaps.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices, one column per ticker.
    max_missing : float, optional
        Drop any ticker whose share of missing prices is above this (default 5%).

    Returns
    -------
    pd.DataFrame
        Daily returns ``p_t / p_{t-1} - 1`` with no missing values.
    """
    # Stocks that listed mid-sample have long runs of NaN. Keeping them would force us
    # to throw away every date before their IPO, so we drop the stock instead.
    missing_share = prices.isna().mean()
    kept = prices.loc[:, missing_share <= max_missing]
    dropped = sorted(set(prices.columns) - set(kept.columns))
    if dropped:
        logger.warning(
            "Dropped %d tickers with >%.0f%% missing data: %s",
            len(dropped),
            100 * max_missing,
            ", ".join(dropped),
        )
    # Forward-fill the few remaining one-day gaps (a halted stock keeps its last price),
    # then drop any leading rows that are still incomplete.
    returns = kept.ffill().pct_change().iloc[1:]
    return returns.dropna(how="any")
