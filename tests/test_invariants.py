"""Property tests that hold regardless of the exact table contents.

Unlike the characterization tests, these do not depend on the golden file, so
they keep protecting the package even when the tables are regenerated or stored
at a different precision.

Several assertions carry an empirically measured tolerance rather than a
mathematically ideal one, and say so. Those are deliberate regression guards:
they lock in the *current* accuracy so it can only improve. The corresponding
ideal assertions live in ``test_known_bugs.py`` as strict xfails.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import levy
from _cases import ALPHAS, BETAS, CONVERT_POINTS, PARS, X

GRID = list(itertools.product(ALPHAS, BETAS))


# --------------------------------------------------------------------------
# Exact structural identities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,beta", [(1.5, 0.3), (0.7, -1.0), (1.9, 0.5), (1.0, 0.0)])
@pytest.mark.parametrize("mu,sigma", [(2.5, 0.5), (-1.0, 3.0), (100.0, 0.01)])
def test_location_scale_identity_is_exact(alpha, beta, mu, sigma):
    """pdf(x; mu, sigma) == pdf((x-mu)/sigma; 0, 1) / sigma, bit for bit.

    ``levy()`` implements location/scale by standardising ``x`` up front, so
    this identity is exact rather than approximate. Any future rewrite of that
    code path has to preserve it.
    """
    np.testing.assert_array_equal(
        levy.levy(X, alpha, beta, mu=mu, sigma=sigma),
        levy.levy((X - mu) / sigma, alpha, beta) / sigma,
    )
    np.testing.assert_array_equal(
        levy.levy(X, alpha, beta, mu=mu, sigma=sigma, cdf=True),
        levy.levy((X - mu) / sigma, alpha, beta, cdf=True),
    )


@pytest.mark.parametrize("alpha", ALPHAS)
def test_symmetric_case_is_symmetric(alpha):
    """For beta == 0 the distribution is symmetric about mu."""
    xs = np.linspace(-40.0, 40.0, 2001)
    pdf = levy.levy(xs, alpha, 0.0)
    np.testing.assert_allclose(pdf, pdf[::-1], rtol=0, atol=1e-8)

    cdf = levy.levy(xs, alpha, 0.0, cdf=True)
    np.testing.assert_allclose(cdf + cdf[::-1], 1.0, rtol=0, atol=1e-7)


@pytest.mark.parametrize("point_index", range(len(CONVERT_POINTS)))
@pytest.mark.parametrize("par_in,par_out", [(a, b) for a in PARS for b in PARS if a != b])
def test_parametrization_round_trip(point_index, par_in, par_out):
    """Converting there and back is the identity, for all 20 ordered pairs."""
    start = levy.Parameters.convert(np.array(CONVERT_POINTS[point_index]), "1", par_in)
    if not np.all(np.isfinite(start)):
        pytest.skip(f"point is not representable in parametrization {par_in!r}")

    there = levy.Parameters.convert(start, par_in, par_out)
    back = levy.Parameters.convert(there, par_out, par_in)
    np.testing.assert_allclose(back, start, rtol=1e-12, atol=1e-12)


def test_convert_is_identity_for_same_parametrization():
    arr = np.array([1.6, 0.5, 0.3, 1.2])
    for par in PARS:
        np.testing.assert_array_equal(levy.Parameters.convert(arr, par, par), arr)


# --------------------------------------------------------------------------
# Return-type contract
# --------------------------------------------------------------------------


def test_scalar_input_returns_python_float():
    result = levy.levy(1.0, 1.5, 0.0)
    assert isinstance(result, float)
    assert not isinstance(result, np.ndarray)


def test_array_input_returns_array_of_same_shape():
    for shape in [(17,), (17, 1), (3, 4)]:
        x = np.linspace(-5.0, 5.0, int(np.prod(shape))).reshape(shape)
        assert levy.levy(x, 1.5, 0.0).shape == shape


def test_random_returns_requested_shape():
    np.random.seed(0)
    assert levy.random(1.5, 0.0, shape=(7, 3)).shape == (7, 3)


# --------------------------------------------------------------------------
# Distributional sanity, with empirically measured tolerances
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,beta", GRID)
def test_cdf_stays_within_unit_interval(alpha, beta):
    """The CDF is in [0, 1], to within the interpolator's current accuracy.

    The 2e-6 slack is measured, not aspirational: a dense sweep over this grid
    reaches -1.66e-06 and 1.0000017. The zero-tolerance version of this
    assertion is a strict xfail in ``test_known_bugs.py``.
    """
    cdf = levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta, cdf=True)
    assert cdf.min() >= -2e-6
    assert cdf.max() <= 1.0 + 2e-6


@pytest.mark.parametrize("alpha,beta", GRID)
def test_pdf_is_essentially_non_negative(alpha, beta):
    """A density must not be negative; today it dips to -1.44e-05.

    ``neglog_levy`` masks this with ``np.maximum(1e-100, ...)``. The exact
    assertion is a strict xfail in ``test_known_bugs.py``.
    """
    pdf = levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta)
    assert pdf.min() >= -2e-5


@pytest.mark.parametrize("alpha,beta", GRID)
def test_cdf_is_essentially_monotone(alpha, beta):
    """The CDF is non-decreasing, to within the crossover discontinuity.

    Every worst-case violation sits exactly at the interpolation/power-law tail
    crossover read from ``upper_limit.npz``; the largest is -2.57e-03 at
    alpha=0.5, beta=0.0. Exact version: strict xfail in ``test_known_bugs.py``.
    """
    cdf = levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta, cdf=True)
    assert np.diff(cdf).min() >= -3e-3


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cdf_approaches_zero_and_one_in_the_tails(alpha):
    """The tolerance is 1e-3 because the tails are genuinely heavy.

    Stable laws have P(|X| > x) ~ x^-alpha, so at x=1e6 the remaining mass is
    3.99e-04 for alpha=0.5 and only 2.0e-10 for alpha=1.5. A single tight
    tolerance here would be asserting something false about alpha=0.5.
    """
    far = np.array([-1e6, 1e6])
    cdf = levy.levy(far, alpha, 0.0, cdf=True)
    assert cdf[0] == pytest.approx(0.0, abs=1e-3)
    assert cdf[1] == pytest.approx(1.0, abs=1e-3)


def test_tail_mass_shrinks_as_alpha_grows():
    """Heavier tails for smaller alpha -- the defining feature of these laws."""
    masses = [1.0 - levy.levy(np.array([1e6]), alpha, 0.0, cdf=True)[0] for alpha in ALPHAS]
    assert all(a > b for a, b in zip(masses, masses[1:])), masses


def test_neglog_levy_matches_minus_log_pdf():
    pdf = levy.levy(X, 1.5, 0.3, 0.0, 1.0)
    np.testing.assert_allclose(
        levy.neglog_levy(X, 1.5, 0.3, 0.0, 1.0),
        -np.log(np.maximum(1e-100, pdf)),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------
# Sampler statistics
# --------------------------------------------------------------------------


def test_alpha_2_sampler_is_gaussian_with_variance_two():
    """At alpha == 2 the stable law is Normal with scale sqrt(2)."""
    np.random.seed(0)
    sample = levy.random(2.0, 0.0, shape=(200000,))
    assert sample.mean() == pytest.approx(0.0, abs=0.02)
    assert sample.std() == pytest.approx(np.sqrt(2.0), rel=0.02)


def test_symmetric_sampler_has_symmetric_quantiles():
    np.random.seed(0)
    sample = levy.random(1.5, 0.0, shape=(200000,))
    lo, hi = np.percentile(sample, [10.0, 90.0])
    assert lo == pytest.approx(-hi, rel=0.05)


def test_location_shifts_the_sample():
    np.random.seed(0)
    base = levy.random(1.5, 0.0, mu=0.0, sigma=1.0, shape=(1000,))
    np.random.seed(0)
    shifted = levy.random(1.5, 0.0, mu=10.0, sigma=1.0, shape=(1000,))
    np.testing.assert_allclose(shifted - base, 10.0, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------
# The interpolator, exercised on a tiny synthetic grid
# --------------------------------------------------------------------------


def test_interpolator_reproduces_a_quadratic(tiny_grid):
    """Catmull-Rom is exact for quadratics on a uniform interior grid.

    This is a real oracle, not a smoke test, and it runs against a (20, 8, 11)
    synthetic array -- the shipped 12 MB tables are never touched.

    Quadratic, not cubic: Catmull-Rom takes its tangents from centred
    differences, which are exact for a quadratic but carry an O(h^2 f''')
    error for a cubic. Measured here: 1.3e-15 for a quadratic against 8.8e-04
    for a cubic.
    """
    grid, lower, upper, f = tiny_grid
    rng = np.random.RandomState(0)

    # Stay two cells away from every boundary, where the stencil is complete.
    margin = 2.5 * (upper - lower) / (np.array(grid.shape) - 1)
    points = rng.uniform(lower + margin, upper - margin, size=(500, 3))

    got = levy._interpolate(points, grid, lower, upper)
    want = f(points[:, 0], points[:, 1], points[:, 2])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_interpolator_converges_as_the_grid_refines():
    """Halving the spacing must reduce the error on a smooth non-polynomial.

    Guards against a regression that silently degrades the interpolator to
    something lower-order; a pure correctness bug would not show up in the
    quadratic oracle above.
    """
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])

    def f(x, y, z):
        return np.exp(0.5 * x) * np.cos(y) + np.sin(0.75 * z)

    rng = np.random.RandomState(0)
    errors = []
    for n in (12, 24, 48):
        axes = [np.linspace(lower[i], upper[i], n) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        margin = 2.5 * (upper - lower) / (n - 1)
        points = rng.uniform(lower + margin, upper - margin, size=(300, 3))
        got = levy._interpolate(points, f(gx, gy, gz), lower, upper)
        errors.append(np.abs(got - f(points[:, 0], points[:, 1], points[:, 2])).max())

    assert errors[1] < errors[0] / 4.0
    assert errors[2] < errors[1] / 4.0


def test_interpolator_recovers_grid_nodes(tiny_grid):
    """At a node, interpolation returns that node's value."""
    grid, lower, upper, _ = tiny_grid
    axes = [np.linspace(lower[i], upper[i], grid.shape[i]) for i in range(3)]
    idx = [(5, 3, 4), (9, 2, 7), (11, 6, 2)]
    points = np.array([[axes[d][i[d]] for d in range(3)] for i in idx])

    np.testing.assert_allclose(
        levy._interpolate(points, grid, lower, upper),
        [grid[i] for i in idx],
        rtol=1e-12,
        atol=1e-12,
    )


def test_interpolator_preserves_point_shape(tiny_grid):
    grid, lower, upper, _ = tiny_grid
    points = np.zeros((4, 6, 3))
    points[..., 1] = 1.0
    assert levy._interpolate(points, grid, lower, upper).shape == (4, 6)
