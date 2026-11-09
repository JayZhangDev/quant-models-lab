"""Marchenko-Pastur law: telling real correlation structure apart from noise.

The setup: N stocks, T days of returns, and ``q = N / T``. Even if the stocks were
completely unrelated, the *measured* correlation matrix would not be the identity,
because T is finite. Random matrix theory says exactly what that noise looks like:
the eigenvalues of a pure-noise correlation matrix pile up inside a known interval
``[lambda_minus, lambda_plus]``. Any eigenvalue above ``lambda_plus`` is therefore
real structure (the market, a sector); everything inside the interval cannot be
told apart from noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur noise band.

    Parameters
    ----------
    q : float
        Ratio ``N / T`` (number of assets over number of observations), ``0 < q < 1``.
    sigma2 : float, optional
        Variance of the noise part of the data (1.0 for a pure-noise correlation matrix).

    Returns
    -------
    tuple of float
        ``(lambda_minus, lambda_plus) = sigma2 * (1 -/+ sqrt(q))**2``.
    """
    # The band widens as q grows: fewer days per stock means noisier estimates,
    # so noise eigenvalues spread further away from 1.
    if not 0.0 < q < 1.0:
        raise ValueError(f"q = N/T must be between 0 and 1, got {q:.3f}")
    if sigma2 <= 0.0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    root_q = np.sqrt(q)
    return float(sigma2 * (1.0 - root_q) ** 2), float(sigma2 * (1.0 + root_q) ** 2)


def mp_pdf(lam: ArrayLike, q: float, sigma2: float = 1.0) -> FloatArray:
    """Marchenko-Pastur probability density of noise eigenvalues.

    Parameters
    ----------
    lam : array-like
        Eigenvalue(s) at which to evaluate the density.
    q : float
        Ratio ``N / T``, ``0 < q < 1``.
    sigma2 : float, optional
        Variance of the noise.

    Returns
    -------
    np.ndarray
        ``sqrt((lambda_plus - lam) * (lam - lambda_minus)) / (2 * pi * q * sigma2 * lam)``
        inside the band and 0 outside it. Same shape as ``lam``.
    """
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    pdf = np.zeros_like(lam_arr)
    # Only evaluate the formula inside the band; outside it the square root would be
    # of a negative number, and the true density is simply zero there.
    inside = (lam_arr > lam_minus) & (lam_arr < lam_plus)
    x = lam_arr[inside]
    pdf[inside] = np.sqrt((lam_plus - x) * (x - lam_minus)) / (2.0 * np.pi * q * sigma2 * x)
    return pdf


def correlation_matrix(returns: ArrayLike) -> FloatArray:
    """Sample correlation matrix of a (T, N) array of returns.

    Parameters
    ----------
    returns : array-like, shape (T, N)
        One row per day, one column per asset.

    Returns
    -------
    np.ndarray, shape (N, N)
        Pearson correlations with an exact unit diagonal.
    """
    corr = np.corrcoef(np.asarray(returns, dtype=float), rowvar=False)
    np.fill_diagonal(corr, 1.0)  # remove floating-point dust like 0.9999999999
    return corr


def sorted_eigh(matrix: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Eigen-decomposition of a symmetric matrix, largest eigenvalue first.

    Parameters
    ----------
    matrix : array-like, shape (N, N)
        A symmetric matrix (e.g. a correlation matrix).

    Returns
    -------
    eigenvalues : np.ndarray, shape (N,)
        Sorted from largest to smallest.
    eigenvectors : np.ndarray, shape (N, N)
        Column ``i`` is the eigenvector for ``eigenvalues[i]``.
    """
    # eigh is the solver for symmetric matrices: faster than eig and guaranteed to
    # return real numbers. It sorts ascending, so we flip to get "biggest first".
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def fit_sigma2(eigenvalues: ArrayLike, q: float, n_bins: int | None = None) -> float:
    """Fit the noise variance ``sigma2`` so the MP density matches the eigenvalue histogram.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues of a sample correlation matrix.
    q : float
        Ratio ``N / T``.
    n_bins : int, optional
        Number of histogram bins. Defaults to about ``2 * sqrt(N)``.

    Returns
    -------
    float
        The ``sigma2`` in (0, 1] that minimizes the squared distance between the
        histogram of the bulk eigenvalues and the Marchenko-Pastur density.

    Notes
    -----
    Why is the fitted ``sigma2`` below 1 for real stocks? A correlation matrix has
    trace N, so its eigenvalues always average exactly 1. When one eigenvalue is huge
    (the "market factor" - everything moving together - can be 30% or more of the
    total), the remaining eigenvalues must average *less* than 1 to compensate. The
    noise band therefore sits lower and narrower than the textbook ``sigma2 = 1``
    band: ``sigma2`` is roughly the share of variance NOT explained by the big factors.
    """
    values = np.asarray(eigenvalues, dtype=float)

    # Step 1: set aside the obvious outliers. Since sigma2 can never exceed 1, an
    # eigenvalue above the sigma2 = 1 upper edge cannot be noise under any fit.
    upper = mp_bounds(q, 1.0)[1]
    bulk = values[values <= upper]
    if bulk.size < 5:
        raise ValueError("Too few bulk eigenvalues to fit a Marchenko-Pastur density.")

    # Step 2: histogram of the bulk, normalized to integrate to 1 like a density.
    if n_bins is None:
        n_bins = max(10, int(2 * np.sqrt(bulk.size)))
    density, edges = np.histogram(bulk, bins=n_bins, range=(0.0, upper), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def loss(sigma2: float) -> float:
        # The MP curve only describes the eigenvalues inside its own band, so we scale
        # it by the share of the bulk that actually lies there. Without this, a curve
        # with area 1 would be compared against a histogram whose in-band area is < 1.
        share_in_band = np.mean(bulk <= mp_bounds(q, sigma2)[1])
        return float(np.sum((density - share_in_band * mp_pdf(centers, q, sigma2)) ** 2))

    # Step 3: minimize. The loss can have small local dips (a histogram is bumpy), so
    # we first scan a coarse grid to find the right valley, then let scipy polish it.
    grid = np.linspace(0.05, 1.0, 96)
    best = int(np.argmin([loss(s) for s in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    result = minimize_scalar(loss, bounds=(lo, hi), method="bounded")
    return float(result.x)


def count_signal_eigenvalues(eigenvalues: ArrayLike, q: float, sigma2: float) -> int:
    """Number of eigenvalues above the noise band, i.e. carrying real information.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues of a sample correlation matrix.
    q : float
        Ratio ``N / T``.
    sigma2 : float
        Fitted noise variance.

    Returns
    -------
    int
        Count of eigenvalues strictly above ``lambda_plus``.
    """
    lam_plus = mp_bounds(q, sigma2)[1]
    return int(np.sum(np.asarray(eigenvalues, dtype=float) > lam_plus))


def clean_correlation(corr: ArrayLike, q: float, sigma2: float | None = None) -> FloatArray:
    """Denoise a correlation matrix by eigenvalue clipping.

    Parameters
    ----------
    corr : array-like, shape (N, N)
        Sample correlation matrix.
    q : float
        Ratio ``N / T`` of the sample it was estimated from.
    sigma2 : float, optional
        Noise variance. If omitted it is fitted with :func:`fit_sigma2`.

    Returns
    -------
    np.ndarray, shape (N, N)
        Cleaned correlation matrix: symmetric, positive semi-definite, unit diagonal.

    Notes
    -----
    The recipe (Laloux et al., 1999):

    1. Split the matrix into eigenvalues (how much variance) and eigenvectors (which
       combination of stocks).
    2. Keep every eigenvalue above ``lambda_plus`` - that is signal.
    3. Replace all the others with their average. They are noise, so their individual
       sizes are meaningless; averaging keeps the total variance (the trace) unchanged.
    4. Rebuild the matrix and rescale so the diagonal is exactly 1 again.
    """
    corr_arr = np.asarray(corr, dtype=float)
    values, vectors = sorted_eigh(corr_arr)
    if sigma2 is None:
        sigma2 = fit_sigma2(values, q)
    lam_plus = mp_bounds(q, sigma2)[1]

    is_noise = values <= lam_plus
    cleaned_values = values.copy()
    if is_noise.any():
        cleaned_values[is_noise] = values[is_noise].mean()

    # Rebuild: C = V * diag(eigenvalues) * V^T. Multiplying the columns of V by the
    # eigenvalues first avoids building a full diagonal matrix.
    cleaned = (vectors * cleaned_values) @ vectors.T
    cleaned = 0.5 * (cleaned + cleaned.T)  # enforce exact symmetry

    # Clipping changes the diagonal slightly; divide by sqrt(d_i * d_j) to restore the
    # "every stock has correlation 1 with itself" property.
    scale = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(scale, scale)
    np.fill_diagonal(cleaned, 1.0)
    return cleaned
