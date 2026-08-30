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
# (c) the tail-crossover cell is truncated instead of rounded
# --------------------------------------------------------------------------


@xfail
def test_tail_crossover_is_accurate_off_the_grid():
    """The tail crossover is inaccurate for alpha/beta between grid points.

    ``_grid_index`` snaps (alpha, beta) to one cell of the 76x101 limit tables
    and uses that cell's crossover for every value in between. The consequence
    is large: at alpha=1.410182, beta=-0.5 the selected cell has an upper
    crossover of 71.30 while the neighbouring cell has 499.80, so at x=285.55
    the two disagree by 64% -- one takes the power-law branch, the other the
    interpolated branch.

    Note that simply rounding to the nearest cell does **not** fix this. Over
    300 sampled disagreement points, measured against ``_calculate_levy``:

        truncate  median 1.1164e-02   mean 3.5488e-01   p90 1.4560
        round     median 1.4552e-02   mean 4.3251e-01   p90 1.4656
        bilinear  median 1.0106e-02   mean 2.9459e-01   p90 0.9723

    Rounding is *worse* on the median and better on only 43% of points. The
    error is dominated by the seam itself: the interpolated branch and the
    power-law branch do not meet at the crossover (see
    ``test_cdf_is_monotone_across_the_tail_crossover``). A real fix means
    interpolating the limits and reconciling the two branches so they agree
    where they join, which is a numerical change worth its own PR and its own
    evidence -- not a one-line int()/round() swap.

    This test asserts the accuracy target, so it stays red until that lands.
    """
    rng = np.random.RandomState(0)
    errors = []
    for _ in range(40):
        alpha = rng.uniform(0.55, 1.95)
        beta = rng.uniform(-0.95, 0.95)
        x = rng.uniform(15.0, 700.0) * rng.choice([-1.0, 1.0])
        truth = levy._calculate_levy(float(x), alpha, beta, False)
        if not np.isfinite(truth) or abs(truth) < 1e-300:
            continue
        got = levy.levy(np.array([float(x)]), alpha, beta)[0]
        errors.append(abs(got - truth) / abs(truth))

    assert np.median(errors) < 1e-4, (
        "median relative error near the tail crossover is "
        f"{np.median(errors):.3e}"
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
