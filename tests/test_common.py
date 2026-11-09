"""Tests for the shared helpers: price cache and performance metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qml.common import data, metrics
from qml.common.plotting import to_markdown


def _fake_prices(tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5, name="date")
    return pd.DataFrame({t: np.linspace(100.0, 104.0, 5) for t in tickers}, index=dates)


def test_get_prices_uses_cache_on_second_call(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_download(tickers, start, end):
        calls.append(1)
        return _fake_prices(list(tickers))

    monkeypatch.setattr(data, "_download", fake_download)
    first = data.get_prices(["AAA", "BBB"], "2024-01-01", "2024-01-08", cache_dir=tmp_path)
    second = data.get_prices(["AAA", "BBB"], "2024-01-01", "2024-01-08", cache_dir=tmp_path)

    assert len(calls) == 1  # the second call never touched the "network"
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_get_prices_drops_failed_tickers(tmp_path: Path, monkeypatch, caplog) -> None:
    def fake_download(tickers, start, end):
        prices = _fake_prices(list(tickers))
        prices["DEAD"] = np.nan  # what Yahoo returns for a delisted symbol
        return prices

    monkeypatch.setattr(data, "_download", fake_download)
    with caplog.at_level("WARNING"):
        prices = data.get_prices(["AAA", "DEAD"], "2024-01-01", "2024-01-08", cache_dir=tmp_path)

    assert list(prices.columns) == ["AAA"]
    assert "DEAD" in caplog.text


def test_get_prices_raises_when_nothing_downloads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(data, "_download", lambda tickers, start, end: pd.DataFrame())
    with pytest.raises(RuntimeError):
        data.get_prices(["AAA"], "2024-01-01", "2024-01-08", cache_dir=tmp_path)


def test_load_universe_skips_comments_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "universe.txt"
    path.write_text("# header\naapl\n\nMSFT\nAAPL\n", encoding="utf-8")
    assert data.load_universe(path) == ["AAPL", "MSFT"]


def test_clean_returns_drops_gappy_tickers() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = pd.DataFrame(
        {"GOOD": np.linspace(100, 110, 100), "GAPPY": np.linspace(50, 60, 100)}, index=dates
    )
    prices.iloc[:20, 1] = np.nan  # 20% missing, above the 5% limit
    returns = data.clean_returns(prices)
    assert list(returns.columns) == ["GOOD"]
    assert not returns.isna().any().any()


def test_annualized_vol_matches_hand_calculation() -> None:
    returns = np.array([0.01, -0.01, 0.01, -0.01])
    expected = np.std(returns, ddof=1) * np.sqrt(252)
    assert metrics.annualized_vol(returns) == pytest.approx(expected)


def test_sharpe_is_zero_without_volatility() -> None:
    assert metrics.sharpe_ratio(np.zeros(10)) == 0.0


def test_sharpe_sign_follows_mean_return() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, 1000)
    assert metrics.sharpe_ratio(noise + 0.002) > 0
    assert metrics.sharpe_ratio(noise - 0.002) < 0


def test_max_drawdown_simple_case() -> None:
    # Up 10%, then down 50%, then up 10%: the worst fall from a peak is 50%.
    assert metrics.max_drawdown([0.10, -0.50, 0.10]) == pytest.approx(-0.50)
    assert metrics.max_drawdown([0.01, 0.02]) == 0.0


def test_annualized_return_of_constant_growth() -> None:
    daily = 1.10 ** (1 / 252) - 1  # grows exactly 10% over 252 days
    assert metrics.annualized_return(np.full(252, daily)) == pytest.approx(0.10)


def test_turnover_counts_one_way_trading() -> None:
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # First rebalance swaps the whole portfolio (turnover 1), second does nothing (0).
    assert metrics.turnover(weights) == pytest.approx(0.5)


def test_to_markdown_has_header_divider_and_rows() -> None:
    table = to_markdown(pd.DataFrame({"name": ["a"], "value": [1.23456]}), "{:.2f}")
    assert table.splitlines() == ["| name | value |", "| --- | --- |", "| a | 1.23 |"]
