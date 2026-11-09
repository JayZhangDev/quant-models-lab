"""Tests for Almgren-Chriss optimal execution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qml.execution.almgren_chriss import (
    ACParams,
    continuous_trajectory,
    expected_cost,
    expected_cost_and_variance,
    kappa_continuous,
    kappa_discrete,
    optimal_trajectory,
    optimize_trajectory,
    simulate_costs,
    sqrt_impact_coefficient,
    twap_trajectory,
)
from qml.execution.frontier import efficient_frontier, frontier_point

PARAMS = ACParams()


def _optimal(lam: float, params: ACParams = PARAMS) -> np.ndarray:
    return optimal_trajectory(
        params.X, params.T, params.n_steps, params.sigma, params.eta, params.gamma, lam
    )


def test_zero_risk_aversion_gives_twap() -> None:
    assert np.allclose(_optimal(0.0), twap_trajectory(PARAMS.X, PARAMS.n_steps))


def test_tiny_risk_aversion_is_almost_a_straight_line() -> None:
    twap = twap_trajectory(PARAMS.X, PARAMS.n_steps)
    assert np.abs(_optimal(1e-12) - twap).max() < 1e-3 * PARAMS.X


@pytest.mark.parametrize("lam", [0.0, 1e-7, 1e-6, 1e-5, 1e-3])
def test_trajectory_starts_at_x_and_ends_at_zero(lam: float) -> None:
    trajectory = _optimal(lam)
    assert trajectory[0] == pytest.approx(PARAMS.X)
    assert trajectory[-1] == 0.0
    assert np.all(np.diff(trajectory) <= 1e-9)  # never buys back
    assert np.all(np.isfinite(trajectory))


def test_higher_risk_aversion_sells_faster_early() -> None:
    patient, medium, nervous = _optimal(1e-7), _optimal(1e-6), _optimal(1e-5)
    interior = slice(1, -1)
    assert np.all(nervous[interior] < medium[interior])
    assert np.all(medium[interior] < patient[interior])
    # The first trade of the nervous trader is the largest of the three.
    assert (nervous[0] - nervous[1]) > (medium[0] - medium[1]) > (patient[0] - patient[1])


def test_discrete_kappa_converges_to_the_continuous_formula() -> None:
    lam = 1e-6
    target = kappa_continuous(PARAMS.sigma, PARAMS.eta, lam)
    errors = [
        abs(kappa_discrete(PARAMS.T / n, PARAMS.sigma, PARAMS.eta, PARAMS.gamma, lam) - target)
        for n in (5, 50, 500, 5000)
    ]
    assert errors == sorted(errors, reverse=True)  # shrinks as the grid gets finer
    assert errors[-1] < 1e-4 * target


def test_discrete_trajectory_converges_to_the_continuous_one() -> None:
    lam = 1e-6
    fine = replace(PARAMS, n_steps=5000)
    discrete = _optimal(lam, fine)
    continuous = continuous_trajectory(fine.X, fine.T, fine.times, fine.sigma, fine.eta, lam)
    assert np.abs(discrete - continuous).max() < 1e-4 * fine.X


def test_twap_cost_matches_hand_calculation() -> None:
    cost, variance = expected_cost_and_variance(twap_trajectory(PARAMS.X, PARAMS.n_steps), PARAMS)
    tau, n = PARAMS.tau, PARAMS.X / PARAMS.n_steps
    by_hand = (
        0.5 * PARAMS.gamma * PARAMS.X**2  # permanent impact
        + PARAMS.epsilon * PARAMS.X  # half-spread on every share
        + (PARAMS.eta - 0.5 * PARAMS.gamma * tau) / tau * PARAMS.n_steps * n**2  # temporary
    )
    assert cost == pytest.approx(by_hand)
    assert variance > 0.0


def test_two_cost_functions_agree_for_linear_impact() -> None:
    trajectory = _optimal(1e-6)
    closed_form, _ = expected_cost_and_variance(trajectory, PARAMS)
    assert expected_cost(-np.diff(trajectory), PARAMS, "linear") == pytest.approx(closed_form)


def test_optimal_schedule_beats_twap_on_its_own_objective() -> None:
    lam = 1e-6
    cost_opt, var_opt = expected_cost_and_variance(_optimal(lam), PARAMS)
    cost_twap, var_twap = expected_cost_and_variance(
        twap_trajectory(PARAMS.X, PARAMS.n_steps), PARAMS
    )
    assert cost_opt + lam * var_opt < cost_twap + lam * var_twap
    assert cost_opt > cost_twap  # pays more impact...
    assert var_opt < var_twap  # ...to take less risk


@pytest.mark.parametrize("lam", [1e-7, 1e-6, 5e-6])
def test_numerical_optimizer_matches_closed_form_for_linear_impact(lam: float) -> None:
    numeric = optimize_trajectory(PARAMS, lam, impact="linear")
    assert np.abs(numeric - _optimal(lam)).max() < 2e-3 * PARAMS.X


def test_sqrt_impact_is_calibrated_to_match_linear_at_twap() -> None:
    twap_trades = np.full(PARAMS.n_steps, PARAMS.X / PARAMS.n_steps)
    linear = expected_cost(twap_trades, PARAMS, "linear")
    sqrt = expected_cost(twap_trades, PARAMS, "sqrt", sqrt_impact_coefficient(PARAMS))
    assert sqrt == pytest.approx(linear)


def test_sqrt_impact_schedule_is_valid_and_more_front_loaded() -> None:
    lam = 1e-6
    sqrt_schedule = optimize_trajectory(PARAMS, lam, impact="sqrt")
    linear_schedule = _optimal(lam)
    assert sqrt_schedule[0] == pytest.approx(PARAMS.X)
    assert sqrt_schedule[-1] == 0.0
    assert np.all(np.diff(sqrt_schedule) <= 1e-6)
    # Big trades are relatively cheaper under square-root impact, so it sells more early.
    assert (sqrt_schedule[0] - sqrt_schedule[1]) > (linear_schedule[0] - linear_schedule[1])


def test_unknown_impact_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        expected_cost(np.ones(PARAMS.n_steps), PARAMS, impact="cubic")


@pytest.mark.parametrize("lam", [0.0, 1e-6])
def test_monte_carlo_matches_closed_form(lam: float) -> None:
    trajectory = _optimal(lam)
    cost, variance = expected_cost_and_variance(trajectory, PARAMS)
    n_paths = 10_000
    simulated = simulate_costs(trajectory, PARAMS, n_paths=n_paths, seed=11)

    standard_error = np.sqrt(variance / n_paths)
    assert abs(simulated.mean() - cost) < 4.0 * standard_error
    assert simulated.var(ddof=1) == pytest.approx(variance, rel=0.06)


def test_monte_carlo_is_reproducible() -> None:
    trajectory = _optimal(1e-6)
    first = simulate_costs(trajectory, PARAMS, n_paths=100, seed=5)
    second = simulate_costs(trajectory, PARAMS, n_paths=100, seed=5)
    assert np.array_equal(first, second)


def test_frontier_trades_cost_for_risk() -> None:
    frontier = efficient_frontier(PARAMS, np.logspace(-8, -4, 20))
    assert np.all(np.diff(frontier["expected_cost"]) > 0)  # more aversion costs more
    assert np.all(np.diff(frontier["variance"]) < 0)  # and buys less risk
    twap = frontier_point(PARAMS, 0.0)
    assert twap.expected_cost < frontier["expected_cost"].min()  # TWAP is the cheapest
    assert twap.variance > frontier["variance"].max()  # and the riskiest
