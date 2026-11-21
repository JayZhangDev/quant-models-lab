"""Tests for the event-study module, on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qml.events.event_study import (
    abnormal_volume,
    car_table,
    estimation_slice,
    event_position,
    fit_factor_model,
    percentile_of,
    placebo_cars,
    run_event_study,
)

N_DAYS = 800
SHOCK_DAY = 600
SHOCK = -0.05


@pytest.fixture(scope="module")
def market() -> tuple[pd.Series, pd.DataFrame]:
    """Stock = 0.0002 + 1.2 * market + 0.5 * sector + small noise, with a -5% shock."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    factors = pd.DataFrame(
        {"MKT": rng.normal(0.0, 0.01, N_DAYS), "SEC": rng.normal(0.0, 0.015, N_DAYS)},
        index=dates,
    )
    noise = rng.normal(0.0, 0.002, N_DAYS)
    stock = 0.0002 + 1.2 * factors["MKT"] + 0.5 * factors["SEC"] + noise
    stock.iloc[SHOCK_DAY] += SHOCK
    return stock, factors


def test_fit_recovers_the_true_coefficients(market) -> None:
    stock, factors = market
    model = fit_factor_model(stock.iloc[:500], factors.iloc[:500])
    assert model.betas["MKT"] == pytest.approx(1.2, abs=0.02)
    assert model.betas["SEC"] == pytest.approx(0.5, abs=0.02)
    assert model.resid_std == pytest.approx(0.002, rel=0.1)
    assert model.r_squared > 0.95


def test_planted_shock_is_found_on_the_right_day_and_nowhere_else(market) -> None:
    stock, factors = market
    result = run_event_study(stock, factors, stock.index[SHOCK_DAY], pre_days=10)
    ab = result.abnormal.set_index("offset")
    assert result.event_date == stock.index[SHOCK_DAY]
    assert ab.loc[0, "ar"] == pytest.approx(SHOCK, abs=0.006)
    others = ab.drop(index=0)["ar"]
    assert others.abs().max() < 0.01  # about 0 everywhere else (noise std is 0.2%)
    assert result.t_stat(0, 0) < -15


def test_car_equals_the_sum_of_abnormal_returns(market) -> None:
    stock, factors = market
    result = run_event_study(stock, factors, stock.index[SHOCK_DAY], pre_days=10)
    ab = result.abnormal
    expected = ab[(ab["offset"] >= -3) & (ab["offset"] <= 5)]["ar"].sum()
    assert result.car(-3, 5) == pytest.approx(expected)
    table = car_table(result, {"[0]": (0, 0), "[-3,+5]": (-3, 5)})
    assert table.loc[1, "car"] == pytest.approx(expected)
    assert table.loc[1, "actual"] - table.loc[1, "predicted"] == pytest.approx(expected)


def test_market_model_blames_sector_moves_on_the_news_but_two_factor_does_not() -> None:
    # The sector jumps 8% on the event day. A market-only model cannot see that, so it
    # calls the stock's sector-driven gain "abnormal"; the two-factor model does not.
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2021-01-01", periods=500)
    factors = pd.DataFrame(
        {"MKT": rng.normal(0, 0.01, 500), "SEC": rng.normal(0, 0.01, 500)}, index=dates
    )
    factors.iloc[450, 1] = 0.08
    stock = factors["MKT"] + factors["SEC"] + rng.normal(0, 0.001, 500)
    one = run_event_study(stock, factors[["MKT"]], dates[450])
    two = run_event_study(stock, factors, dates[450])
    assert one.car(0, 0) > 0.06
    assert abs(two.car(0, 0)) < 0.005


def test_weekend_news_maps_to_monday() -> None:
    index = pd.bdate_range("2026-11-02", "2026-11-13")
    assert index[event_position(index, "2026-11-08")] == pd.Timestamp("2026-11-09")


def test_estimation_window_ends_before_the_gap() -> None:
    est = estimation_slice(event_pos=400, length=250, gap=30)
    assert (est.start, est.stop) == (120, 370)
    with pytest.raises(ValueError):
        estimation_slice(event_pos=100, length=250, gap=30)


def test_windows_past_the_data_are_rejected_or_skipped(market) -> None:
    stock, factors = market
    result = run_event_study(stock, factors, stock.index[SHOCK_DAY], last_date=stock.index[602])
    with pytest.raises(ValueError):
        result.car(0, 5)
    table = car_table(result, {"[0,+2]": (0, 2), "[0,+5]": (0, 5)})
    assert table["window"].tolist() == ["[0,+2]"]


def test_placebo_is_reproducible_and_excludes_the_real_event(market) -> None:
    stock, factors = market
    first = placebo_cars(
        stock, factors, (0, 2), n_draws=100, exclude=[stock.index[SHOCK_DAY]], seed=5
    )
    second = placebo_cars(
        stock, factors, (0, 2), n_draws=100, exclude=[stock.index[SHOCK_DAY]], seed=5
    )
    pd.testing.assert_frame_equal(first, second)
    positions = stock.index.get_indexer(first["date"])
    assert np.all(np.abs(positions - SHOCK_DAY) > 20)
    # Fake events in pure noise: CARs centred on zero, standardized CARs about N(0, 1).
    assert abs(first["car"].mean()) < 0.002
    assert first["scar"].std() == pytest.approx(1.0, abs=0.25)


def test_real_shock_is_extreme_against_the_placebo(market) -> None:
    stock, factors = market
    placebo = placebo_cars(stock, factors, (0, 0), n_draws=200, exclude=[stock.index[SHOCK_DAY]])
    real = run_event_study(stock, factors, stock.index[SHOCK_DAY]).car(0, 0)
    assert percentile_of(real, placebo["car"]) == 0.0


def test_abnormal_volume_ratio() -> None:
    dates = pd.bdate_range("2026-01-01", periods=100)
    volume = pd.Series(1_000_000.0, index=dates)
    volume.iloc[80] = 3_000_000.0
    out = abnormal_volume(volume, dates[80], days=[-1, 0, 1], baseline=60)
    assert out["ratio"].tolist() == pytest.approx([1.0, 3.0, 1.0])
