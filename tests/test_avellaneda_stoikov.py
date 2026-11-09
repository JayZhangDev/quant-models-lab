"""Tests for the Avellaneda-Stoikov formulas and the market simulator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qml.market_making.simulator import ASParams, fill_intensity, simulate
from qml.market_making.strategies import (
    InventoryStrategy,
    SymmetricStrategy,
    optimal_spread,
    reservation_price,
)

PARAMS = ASParams()


def test_reservation_price_equals_mid_with_no_inventory() -> None:
    mid = np.array([99.0, 100.0, 101.5])
    assert np.array_equal(reservation_price(mid, np.zeros(3), t=0.3, params=PARAMS), mid)


def test_reservation_price_leans_against_inventory() -> None:
    long_r = reservation_price(100.0, 5.0, t=0.0, params=PARAMS)
    short_r = reservation_price(100.0, -5.0, t=0.0, params=PARAMS)
    assert long_r < 100.0 < short_r
    # s - q * gamma * sigma^2 * (T - t) = 100 - 5 * 0.1 * 4 * 1
    assert long_r == pytest.approx(98.0)


def test_reservation_price_equals_mid_with_zero_volatility() -> None:
    calm = replace(PARAMS, sigma=0.0)
    inventory = np.array([-10.0, 0.0, 7.0])
    r = reservation_price(100.0, inventory, t=0.2, params=calm)
    assert np.array_equal(r, np.full(3, 100.0))


def test_reservation_price_converges_to_mid_at_the_close() -> None:
    assert reservation_price(100.0, 8.0, t=PARAMS.T, params=PARAMS) == pytest.approx(100.0)


def test_spread_shrinks_as_time_runs_out() -> None:
    spreads = [optimal_spread(t, PARAMS) for t in np.linspace(0.0, PARAMS.T, 11)]
    assert all(later < earlier for earlier, later in zip(spreads, spreads[1:], strict=False))
    # At the close only the constant term (2 / gamma) * ln(1 + gamma / k) is left.
    assert spreads[-1] == pytest.approx(2 / 0.1 * np.log(1 + 0.1 / 1.5))


def test_spread_matches_the_paper_average() -> None:
    # Average of the spread formula over the session at gamma = 0.1 (hand-checked: 1.49).
    mean_spread = np.mean([optimal_spread(t, PARAMS) for t in np.arange(200) * PARAMS.dt])
    assert mean_spread == pytest.approx(1.49, abs=0.01)


def test_fill_intensity_decays_with_distance() -> None:
    rates = fill_intensity([0.0, 0.5, 1.0], PARAMS)
    assert rates[0] == pytest.approx(PARAMS.A)
    assert rates[0] > rates[1] > rates[2]


def test_same_seed_gives_identical_results() -> None:
    first = simulate(InventoryStrategy(), PARAMS, n_paths=50, seed=123)
    second = simulate(InventoryStrategy(), PARAMS, n_paths=50, seed=123)
    assert np.array_equal(first.pnl, second.pnl)
    assert np.array_equal(first.inventory, second.inventory)


def test_different_seeds_give_different_results() -> None:
    first = simulate(InventoryStrategy(), PARAMS, n_paths=50, seed=1)
    second = simulate(InventoryStrategy(), PARAMS, n_paths=50, seed=2)
    assert not np.array_equal(first.pnl, second.pnl)


def test_both_strategies_face_the_same_price_paths() -> None:
    inventory = simulate(InventoryStrategy(), PARAMS, n_paths=20, seed=5)
    symmetric = simulate(SymmetricStrategy(), PARAMS, n_paths=20, seed=5)
    assert np.array_equal(inventory.mid, symmetric.mid)


def test_bookkeeping_is_consistent() -> None:
    result = simulate(InventoryStrategy(), PARAMS, n_paths=30, seed=9)
    assert result.mid.shape == (PARAMS.n_steps + 1, 30)
    # Inventory changes by at most one share per side per step.
    assert np.abs(np.diff(result.inventory, axis=0)).max() <= 1.0
    # Every cash change is explained by a buy at the bid or a sale at the ask.
    cash_change = np.diff(result.cash, axis=0)
    inventory_change = np.diff(result.inventory, axis=0)
    bought_only = inventory_change == 1.0
    sold_only = inventory_change == -1.0
    assert np.allclose(cash_change[bought_only], -result.bid[bought_only])
    assert np.allclose(cash_change[sold_only], result.ask[sold_only])


def test_quotes_never_cross() -> None:
    result = simulate(InventoryStrategy(), PARAMS, n_paths=100, seed=3)
    assert np.all(result.ask > result.bid)


def test_inventory_strategy_cuts_risk_and_gives_up_a_little_profit() -> None:
    inventory = simulate(InventoryStrategy(), PARAMS, n_paths=1000, seed=0)
    symmetric = simulate(SymmetricStrategy(), PARAMS, n_paths=1000, seed=0)

    # The paper's headline result: much less risk for slightly less profit.
    assert inventory.pnl.std() < 0.7 * symmetric.pnl.std()
    assert inventory.final_inventory.std() < 0.6 * symmetric.final_inventory.std()
    assert inventory.pnl.mean() < symmetric.pnl.mean()
    assert inventory.pnl.mean() > 0.85 * symmetric.pnl.mean()
