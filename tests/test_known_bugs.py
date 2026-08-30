"""Executable bug reports.

Every test here asserts the behaviour pylevy *should* have and is marked
``xfail(strict=True)``, so each one fails today and the suite turns red the
moment a bug is fixed without its report being updated. That makes this file
the checklist for the fix PRs that follow. When a bug is fixed, its test
moves to test_regressions.py and loses the marker.

Each bug was reproduced against numpy 2.5.2 / scipy 1.18.1 on Python 3.12.
Line references are to ``levy/__init__.py`` at the commit that added this file.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

import levy
from _cases import ALPHAS, BETAS

GRID = list(itertools.product(ALPHAS, BETAS))
xfail = pytest.mark.xfail(strict=True)

#: Resolve table paths from the package, not the working directory, so the
#: suite runs the same from any cwd.
PACKAGE_DIR = os.path.dirname(os.path.abspath(levy.__file__))


def _table(name):
    return np.load(os.path.join(PACKAGE_DIR, f"{name}.npz"))["arr_0"]


# --------------------------------------------------------------------------
# (a) random() drops mu and sigma at alpha == 2
# --------------------------------------------------------------------------


@xfail
def test_random_at_alpha_2_respects_location_and_scale():
    """levy/__init__.py:607-608 returns before mu and sigma are applied.

        if alpha == 2:
            return np.random.standard_normal(shape) * np.sqrt(2.0)

    Every other alpha ends at ``return mu + sigma * k`` (line 632). Observed:
    ``random(2.0, 0.0, mu=100.0, sigma=5.0)`` has mean 0.002, std 1.41.
    """
    np.random.seed(0)
    sample = levy.random(2.0, 0.0, mu=100.0, sigma=5.0, shape=(100000,))
    assert sample.mean() == pytest.approx(100.0, abs=0.1)
    assert sample.std() == pytest.approx(5.0 * np.sqrt(2.0), rel=0.02)


# --------------------------------------------------------------------------
# (b) alpha below the supported floor indexes the table from the wrong end
# --------------------------------------------------------------------------


@xfail
def test_alpha_below_supported_range_is_rejected():
    """The module docstring (line 43) says alpha < 0.5 is unsupported, but
    levy/__init__.py:469 computes ``int((alpha - 0.5) / 1.5 * 75)`` with no
    lower bound, so alpha=0.4 gives index -4 -- a *valid* negative index that
    silently reads the limits for alpha ~ 1.94. The ``except IndexError`` guard
    at line 471 only catches positive overflow.

    Observed: ``levy(x, 0.4, 0)`` returns the same values as ``levy(x, 0.5, 0)``
    instead of raising.
    """
    with pytest.raises(ValueError):
        levy.levy(np.array([1.0, 2.0]), 0.4, 0.0)


@xfail
def test_beta_outside_range_is_rejected():
    """Same missing validation on the beta axis (line 470)."""
    with pytest.raises(ValueError):
        levy.levy(np.array([1.0, 2.0]), 1.5, 1.5)


@xfail
def test_alpha_below_range_does_not_alias_to_the_boundary():
    """Distinct unsupported alphas must not silently produce identical output."""
    assert not np.allclose(
        levy.levy(np.array([1.0, 2.0]), 0.4, 0.0),
        levy.levy(np.array([1.0, 2.0]), 0.5, 0.0),
    )


# --------------------------------------------------------------------------
# (c) the tail-crossover cell is truncated instead of rounded
# --------------------------------------------------------------------------


@xfail
def test_tail_crossover_uses_the_nearest_grid_cell():
    """levy/__init__.py:469-470 truncate with ``int()`` where they should round.

    The consequence is observable, not cosmetic. At alpha=1.410182, beta=-0.5
    the truncated cell has an upper crossover of 71.30 while the nearest cell
    has 499.80. At x=285.55 the two branches disagree by 64%:

        levy() returns        1.919040e-07   (power-law tail branch)
        interpolated value    5.382755e-07   (what the nearest cell gives)

    ``sigma`` is unaffected; this is purely the choice of limit cell.
    """
    alpha, beta, x = 1.410182, -0.5, 285.55
    probe = np.array([x])

    # The nearest cell puts x inside the interpolated region.
    upper_limit = _table("upper_limit")
    k = (alpha - 0.5) / 1.5 * 75
    beta_index = int(round((beta + 1.0) / 2.0 * 100))
    assert int(k) != int(round(k)), "probe no longer straddles two cells"
    assert x < upper_limit[int(round(k)), beta_index]

    np.testing.assert_allclose(
        levy.levy(probe, alpha, beta),
        levy._int_levy(probe, alpha, beta),
        rtol=1e-9,
    )


# --------------------------------------------------------------------------
# (l) the alpha ~ 1 skewed sampler is ill-conditioned by construction
# --------------------------------------------------------------------------


@xfail
def test_alpha_1_skewed_sampler_is_well_conditioned():
    """levy/__init__.py:610-615 sidesteps alpha == 1 by nudging:

        radius = 1e-15
        if np.absolute(alpha - 1.0) < radius:
            alpha = 1.0 + radius

    The comment claims this "works fine for alpha infinitesimally greater or
    lower than 1.0". It does not. At alpha = 1 + 1e-15, ``tan(pi*alpha/2)``
    evaluates to -5.83e+14 -- the function is sitting on its pole -- so
    ``_phi(alpha, beta) = beta * tan(pi*alpha/2)`` is ~1.75e+14 for beta=0.3 and
    **a single ULP of alpha changes it by 11.4%**. ``1.0 - alpha`` is also pure
    cancellation (-1.11e-15).

    Consequences: samples for alpha ~ 1 with beta != 0 are not reproducible
    across libm builds (this was caught by CI disagreeing with a local run),
    and their accuracy is governed by the last bit of a tangent near infinity.
    beta == 0 is unaffected, because zero times the pole is zero.

    The fix is the closed form: Chambers, Mallows & Stuck (1976) give a
    separate alpha == 1 branch using log terms, which is what
    ``_calculate_levy`` already does for the density (levy/__init__.py:305).
    """
    alpha = 1.0 + 1e-15
    phi = levy._phi(alpha, 0.3)
    phi_one_ulp_away = levy._phi(np.nextafter(alpha, 2.0), 0.3)
    relative_swing = abs(phi_one_ulp_away - phi) / abs(phi)
    assert relative_swing < 1e-6, (
        f"one ULP in alpha moves _phi by {relative_swing:.1%}; "
        f"tan(pi*alpha/2) = {np.tan(np.pi * alpha / 2):.3e}"
    )


# --------------------------------------------------------------------------
# (h) four corrupt cells in the shipped CDF table
# --------------------------------------------------------------------------


@xfail
def test_cdf_table_has_no_corrupt_cells():
    """``cdf.npz`` holds 5.72e+307 at x-index {99,100}, alpha-index 4
    (alpha=0.58), beta-index {13,87} (beta=-+0.74) -- quadrature failures baked
    into the table in 2017.

    These four are also the only cells in either table that exceed the float32
    maximum, so a naive ``.astype(np.float32)`` would turn a wrong number into
    ``inf``. They must be repaired before the tables are down-converted.
    """
    cdf = _table("cdf")
    assert np.isfinite(cdf).all()
    assert cdf.max() <= 1.0 + 1e-6


@xfail
def test_cdf_is_usable_at_the_corrupt_cells():
    """User-visible symptom of the corrupt cells."""
    result = levy.levy(np.array([0.0]), 0.58, 0.74, cdf=True)
    assert 0.0 <= result[0] <= 1.0


# --------------------------------------------------------------------------
# Accuracy defects: the zero-tolerance forms of the guards in test_invariants
# --------------------------------------------------------------------------


@xfail
def test_pdf_is_never_negative():
    """A density must not be negative. The shipped ``pdf.npz`` has 1845
    negative entries (min -5.44e-08) and interpolation amplifies these to
    -1.44e-05. ``neglog_levy`` hides it behind ``np.maximum(1e-100, ...)``.
    """
    worst = min(
        levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta).min()
        for alpha, beta in GRID
    )
    assert worst >= 0.0


@xfail
def test_cdf_is_exactly_within_the_unit_interval():
    """Measured range over the grid: -1.66e-06 .. 1.0000017."""
    lo = min(
        levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta, cdf=True).min()
        for alpha, beta in GRID
    )
    hi = max(
        levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta, cdf=True).max()
        for alpha, beta in GRID
    )
    assert lo >= 0.0 and hi <= 1.0


@xfail
def test_cdf_is_monotone_across_the_tail_crossover():
    """The CDF steps *down* by up to 2.57e-03 (alpha=0.5, beta=0.0) exactly at
    the point where ``levy()`` switches from the interpolated table to the
    power-law asymptotic. Every worst case in the grid sits on that crossover,
    so the two branches do not meet continuously.
    """
    worst = min(
        np.diff(levy.levy(np.linspace(-500.0, 500.0, 4001), alpha, beta, cdf=True)).min()
        for alpha, beta in GRID
    )
    assert worst >= -1e-9
