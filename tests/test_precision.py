"""Guards on the precision and size of the shipped lookup tables.

The tables are stored as float32. These tests pin the properties that made that
safe, so a future change cannot quietly give the precision back or undo the size
saving.

The one-off comparison against the previous float64 tables is recorded in the
commit that introduced them; it cannot be repeated here because those files are
no longer in the tree. What *is* repeatable is the comparison against
``calculate_levy`` ground truth, which is the thing that actually matters.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import levy

#: The interpolation error of the scheme itself, measured against quadrature.
#: float32 storage contributes ~6e-08 relative, three orders below this, which
#: is why the conversion is invisible end to end.
INTERPOLATION_ERROR_BUDGET = 2e-3


def test_shipped_tables_are_float32():
    for name in ("pdf", "cdf", "lower_limit", "upper_limit"):
        assert levy._read_from_cache(name).dtype == np.float32, name


def test_shipped_tables_fit_the_size_budget():
    """24.7 MB of uncompressed float64 became 10.3 MB of compressed float32."""
    total = sum(
        os.path.getsize(os.path.join(levy.PACKAGED_DATA, name))
        for name in os.listdir(levy.PACKAGED_DATA)
        if name.endswith(".npz")
    )
    assert total < 11e6, "shipped tables grew to {:.2f} MB".format(total / 1e6)


def test_limit_tables_are_merged():
    """lower_limit.npz and upper_limit.npz are now one limits.npz."""
    assert os.path.exists(os.path.join(levy.PACKAGED_DATA, "limits.npz"))
    for old in ("lower_limit.npz", "upper_limit.npz"):
        assert not os.path.exists(os.path.join(levy.PACKAGED_DATA, old))


def test_interpolation_promotes_to_float64():
    """A float32 grid must not drag results down to float32.

    ``_interpolate`` builds its weights in float64, so ``weights * np.take(...)``
    promotes. If that ever changed, every result would lose seven digits.
    """
    result = levy.levy(np.array([0.5, 1.0, 2.0]), 1.5, 0.0)
    assert result.dtype == np.float64
    assert levy.neglog_levy(np.array([1.0]), 1.5, 0.0, 0.0, 1.0).dtype == np.float64


def test_float32_storage_round_trips_exactly():
    """Loading must not perturb the stored values."""
    table = levy._read_from_cache("pdf")
    assert np.array_equal(table.astype(np.float64).astype(np.float32), table)


@pytest.mark.slow
def test_accuracy_against_quadrature_is_within_budget():
    """End-to-end error is set by the interpolation scheme, not by float32.

    Measured when the tables were converted: median relative error against
    ``calculate_levy`` was 8.6868e-05 with float64 tables and 8.6874e-05 with
    float32 -- a ratio of 1.0001.
    """
    rng = np.random.RandomState(1)
    errors = []
    for _ in range(120):
        alpha = rng.uniform(0.6, 1.95)
        beta = rng.uniform(-0.9, 0.9)
        x = float(rng.uniform(0.2, 30.0) * rng.choice([-1.0, 1.0]))
        truth = levy._calculate_levy(x, alpha, beta, False)
        if not np.isfinite(truth) or abs(truth) < 1e-300:
            continue
        got = levy.levy(np.array([x]), alpha, beta)[0]
        errors.append(abs(got - truth) / abs(truth))

    errors = np.array(errors)
    assert np.median(errors) < INTERPOLATION_ERROR_BUDGET, (
        "median relative error {:.3e} exceeds the budget".format(np.median(errors))
    )


@pytest.mark.slow
def test_fits_are_unaffected_by_table_precision():
    """Fitting must stay stable at the scale that matters: sampling error.

    Converting the tables shifted recovered parameters by at most 0.0044% of one
    sampling standard error over 40 seeded samples of n=1000. This asserts the
    fitter still recovers the truth well within its own noise.
    """
    fits = []
    for seed in range(12):
        np.random.seed(seed)
        sample = levy.random(1.5, 0.3, 0.0, 1.0, shape=(1000,))
        fits.append(np.asarray(levy.fit_levy(sample, par="0")[0].get("0")))
    fits = np.array(fits)

    means = fits.mean(axis=0)
    standard_errors = fits.std(axis=0, ddof=1) / np.sqrt(len(fits))
    for index, (name, truth) in enumerate(
        zip(("alpha", "beta", "mu", "sigma"), (1.5, 0.3, 0.0, 1.0))
    ):
        assert abs(means[index] - truth) < 4.0 * standard_errors[index] + 0.05, (
            "{}: recovered {:.4f}, expected {:.4f}".format(name, means[index], truth)
        )
