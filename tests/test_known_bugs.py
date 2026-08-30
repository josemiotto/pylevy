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
# (h) four corrupt cells in the shipped CDF table
# --------------------------------------------------------------------------


@xfail
def test_cdf_table_has_no_corrupt_cells():
    """The *shipped file* still holds 5.72e+307 in four cells.

    x-index {99,100}, alpha-index 4 (alpha=0.58), beta-index {13,87}
    (beta=-+0.74). ``levy()`` now repairs these when it loads the table, so the
    user-visible symptom is gone (see test_regressions.py), but the .npz on disk
    is unchanged and this stays red until the tables are regenerated.

    Note this is not a storage error: ``_calculate_levy`` still returns
    5.72e+307 for those exact arguments, so a regeneration with the current
    generator would reproduce them. Those grid points are the two closest to
    x = 0, where the oscillatory weight passed to ``integrate.quad`` degenerates.

    These four are also the only cells in either table that exceed the float32
    maximum, so a naive ``.astype(np.float32)`` would turn a wrong number into
    ``inf``. They must be fixed at the source before the tables are converted.
    """
    cdf = _table("cdf")
    assert np.isfinite(cdf).all()
    assert cdf.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------
# Accuracy defects: the zero-tolerance forms of the guards in test_invariants
# --------------------------------------------------------------------------


@xfail
def test_pdf_is_never_negative():
    """A density must not be negative; ``levy()`` returns down to -1.44e-05.

    The obvious culprit -- 1845 negative entries in ``pdf.npz`` (min -5.44e-08)
    -- is *not* the cause. Clamping every one of them to zero changes ``levy()``
    output by at most 1.642e-18 and leaves the most negative returned value
    exactly where it was, at -4.488e-08.

    The real source is the interpolator: Catmull-Rom weights have negative
    lobes, so a strictly non-negative grid can still interpolate below zero.
    Fixing this means clamping at the output of ``levy()`` or using a
    shape-preserving interpolant, not repairing the table. ``neglog_levy``
    currently hides it behind ``np.maximum(1e-100, ...)``.
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
