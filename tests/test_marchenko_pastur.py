"""Tests for the Marchenko-Pastur law, correlation cleaning and min-variance portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

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
    cleaned_covariance,
    min_variance_weights,
    min_variance_weights_long_only,
    rolling_min_variance,
    sample_covariance,
)

N_ASSETS, N_OBS = 200, 1000
Q = N_ASSETS / N_OBS


@pytest.fixture(scope="module")
def noise_returns() -> np.ndarray:
    """Pure noise: 200 unrelated 'stocks' observed for 1000 'days'."""
    return np.random.default_rng(42).standard_normal((N_OBS, N_ASSETS))


@pytest.fixture(scope="module")
def factor_returns() -> np.ndarray:
    """One strong common factor plus noise, like a market that moves all stocks together."""
    rng = np.random.default_rng(7)
    market = rng.standard_normal((N_OBS, 1))
    return 0.6 * market + 0.8 * rng.standard_normal((N_OBS, N_ASSETS))


def test_bounds_formula() -> None:
    lam_minus, lam_plus = mp_bounds(0.25, sigma2=1.0)
    assert lam_minus == pytest.approx(0.25)  # (1 - 0.5)^2
    assert lam_plus == pytest.approx(2.25)  # (1 + 0.5)^2


def test_bounds_reject_q_at_or_above_one() -> None:
    with pytest.raises(ValueError):
        mp_bounds(1.0)
    with pytest.raises(ValueError):
        mp_bounds(1.5)


@pytest.mark.parametrize(("q", "sigma2"), [(0.2, 1.0), (0.5, 1.0), (0.3, 0.6)])
def test_pdf_integrates_to_one(q: float, sigma2: float) -> None:
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    area, _ = quad(lambda x: float(mp_pdf(x, q, sigma2)[0]), lam_minus, lam_plus)
    assert area == pytest.approx(1.0, abs=1e-6)


def test_pdf_is_zero_outside_the_band() -> None:
    lam_minus, lam_plus = mp_bounds(Q)
    assert np.all(mp_pdf([lam_minus - 0.1, lam_plus + 0.1], Q) == 0.0)


def test_pure_noise_eigenvalues_fall_inside_the_band(noise_returns: np.ndarray) -> None:
    eigenvalues, _ = sorted_eigh(correlation_matrix(noise_returns))
    lam_minus, lam_plus = mp_bounds(Q)
    tolerance = 0.05  # finite N blurs the edges slightly
    inside = (eigenvalues > lam_minus - tolerance) & (eigenvalues < lam_plus + tolerance)
    assert inside.mean() >= 0.99


def test_fit_recovers_sigma2_of_pure_noise(noise_returns: np.ndarray) -> None:
    eigenvalues, _ = sorted_eigh(correlation_matrix(noise_returns))
    assert fit_sigma2(eigenvalues, Q) == pytest.approx(1.0, abs=0.08)


def test_fit_gives_sigma2_below_one_with_a_market_factor(factor_returns: np.ndarray) -> None:
    eigenvalues, _ = sorted_eigh(correlation_matrix(factor_returns))
    sigma2 = fit_sigma2(eigenvalues, Q)
    # The factor explains 0.36 of each stock's variance, leaving 0.64 as noise.
    assert sigma2 == pytest.approx(0.64, abs=0.08)
    assert count_signal_eigenvalues(eigenvalues, Q, sigma2) == 1


def test_cleaned_matrix_is_a_valid_correlation_matrix(factor_returns: np.ndarray) -> None:
    corr = correlation_matrix(factor_returns)
    cleaned = clean_correlation(corr, Q)

    assert np.allclose(cleaned, cleaned.T)  # symmetric
    assert np.allclose(np.diag(cleaned), 1.0)  # unit diagonal
    assert np.trace(cleaned) == pytest.approx(np.trace(corr))  # trace preserved
    assert np.linalg.eigvalsh(cleaned).min() > -1e-10  # positive semi-definite


def test_cleaning_keeps_the_market_factor(factor_returns: np.ndarray) -> None:
    corr = correlation_matrix(factor_returns)
    top_before = sorted_eigh(corr)[0][0]
    top_after = sorted_eigh(clean_correlation(corr, Q))[0][0]
    assert top_after == pytest.approx(top_before, rel=0.02)


def test_cleaning_pure_noise_gives_nearly_the_identity(noise_returns: np.ndarray) -> None:
    cleaned = clean_correlation(correlation_matrix(noise_returns), Q, sigma2=1.0)
    off_diagonal = cleaned - np.eye(N_ASSETS)
    assert np.abs(off_diagonal).max() < 0.05


def test_min_variance_weights_sum_to_one_and_beat_equal_weight() -> None:
    cov = np.diag([0.04, 0.01, 0.09])
    weights = min_variance_weights(cov)
    assert weights.sum() == pytest.approx(1.0)
    # With no correlations the answer is "inverse variance": the calmest asset gets most.
    assert weights == pytest.approx(np.array([9.0, 36.0, 4.0]) / 49.0)
    equal = np.full(3, 1 / 3)
    assert weights @ cov @ weights < equal @ cov @ equal


def test_long_only_weights_are_valid_and_match_when_unconstrained_is_positive() -> None:
    cov = np.diag([0.04, 0.01, 0.09])
    weights = min_variance_weights_long_only(cov)
    assert weights.min() >= 0.0
    assert weights.sum() == pytest.approx(1.0)
    assert weights == pytest.approx(min_variance_weights(cov), abs=1e-4)


def test_long_only_removes_short_positions() -> None:
    # Two highly correlated assets: the unconstrained solution shorts the riskier one.
    cov = np.array([[0.010, 0.018], [0.018, 0.040]])
    assert min_variance_weights(cov).min() < 0.0
    assert min_variance_weights_long_only(cov) == pytest.approx([1.0, 0.0], abs=1e-6)


def test_rolling_backtest_is_out_of_sample_and_cleaning_helps() -> None:
    rng = np.random.default_rng(3)
    n_obs, n_assets = 600, 60
    market = rng.standard_normal((n_obs, 1))
    values = 0.01 * (0.5 * market + rng.standard_normal((n_obs, n_assets)))
    returns = pd.DataFrame(values, index=pd.bdate_range("2020-01-01", periods=n_obs))

    result = rolling_min_variance(
        returns,
        {"raw": sample_covariance, "cleaned": cleaned_covariance},
        est_window=100,
        hold_period=20,
    )

    # First out-of-sample day comes right after the first estimation window.
    assert result.daily_returns.index[0] == returns.index[100]
    assert result.weights["raw"].shape == (25, n_assets)
    assert np.allclose(result.weights["cleaned"].sum(axis=1), 1.0)

    realized = result.daily_returns.std()
    assert realized["cleaned"] < realized["raw"]
    # The raw matrix is over-optimistic: it predicts far less risk than it delivers.
    by_estimator = result.windows.groupby("estimator")[["predicted_vol", "realized_vol"]].mean()
    assert by_estimator.loc["raw", "predicted_vol"] < 0.7 * by_estimator.loc["raw", "realized_vol"]
