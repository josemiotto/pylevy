"""Regression tests for defects that have been fixed.

These started life in ``test_known_bugs.py`` as ``xfail(strict=True)`` bug
reports. When a fix lands, its report moves here and loses the marker, so the
file is a running record of what has been repaired and a guard against any of
it coming back. Anything still marked xfail over there is still broken.

Fixed in "Fix NumPy 2 incompatibility, remove dead code, replace prints with
logging" -- see that commit for the reasoning behind each.
"""

from __future__ import annotations

import gc
import math
import subprocess
import sys
import warnings

import numpy as np
import pytest

import levy


# --------------------------------------------------------------------------
# Parameters.x setter (was: UnboundLocalError for anything but
# OptimizeResult/ndarray, dispatching on __class__.__name__)
# --------------------------------------------------------------------------


def _free_parameters():
    return levy.Parameters(par="0", alpha=None, beta=None, mu=None, sigma=None)


def test_parameters_x_setter_accepts_a_list():
    parameters = _free_parameters()
    parameters.x = [1.5, 0.0, 0.0, 1.0]
    np.testing.assert_allclose(parameters.get("0"), [1.5, 0.0, 0.0, 1.0])


def test_parameters_x_setter_accepts_a_tuple():
    parameters = _free_parameters()
    parameters.x = (1.5, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(parameters.get("0"), [1.5, 0.0, 0.0, 1.0])


def test_parameters_x_setter_accepts_an_ndarray():
    parameters = _free_parameters()
    parameters.x = np.array([1.5, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(parameters.get("0"), [1.5, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("bad", ["not parameters", {"alpha": 1.5}, 1.5, None])
def test_parameters_x_setter_raises_type_error_for_unsupported_types(bad):
    """Was UnboundLocalError, which told the caller nothing."""
    parameters = _free_parameters()
    with pytest.raises(TypeError):
        parameters.x = bad


@pytest.mark.parametrize("bad", [[], [1.5, 0.0], [1.5, 0.0, 0.0, 1.0, 2.0]])
def test_parameters_x_setter_rejects_the_wrong_number_of_values(bad):
    """Was a bare IndexError from the assignment loop, naming neither the
    setter nor the length it wanted. A too-long sequence used to be accepted
    silently, with the surplus ignored.
    """
    parameters = _free_parameters()
    with pytest.raises(ValueError, match="free parameters"):
        parameters.x = bad


def test_parameters_x_setter_still_accepts_an_optimize_result():
    """The path fit_levy actually uses; dispatch is now isinstance-based."""
    from scipy.optimize import OptimizeResult

    parameters = _free_parameters()
    parameters.x = OptimizeResult(x=np.array([1.5, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(parameters.get("0"), [1.5, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# _reflect (was: unbounded `while 1:`)
# --------------------------------------------------------------------------


def test_reflect_terminates_for_far_out_of_bounds_input():
    """Used to need ~1e20 iterations; still running after 5 s in a subprocess."""
    completed = subprocess.run(
        [sys.executable, "-c", "import levy; print(levy._reflect(1e30, 1e-6, 1e10))"],
        timeout=30,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert 1e-6 <= float(completed.stdout) <= 1e10


@pytest.mark.parametrize("lower,upper", [(0.5, 2.0), (-1.0, 1.0), (1e-6, 1e10)])
def test_reflect_is_identity_inside_the_bounds(lower, upper):
    """The fast path must stay bit-exact: it is the case fitting always hits.

    If this ever regresses, every fit_levy golden moves with it.
    """
    rng = np.random.RandomState(0)
    for value in rng.uniform(lower, upper, 2000):
        result = levy._reflect(float(value), lower, upper)
        # `==` alone would accept -0.0 for 0.0, which is not the same
        # float; the docstring above claims bit-exactness, so assert it.
        assert result == float(value)
        assert math.copysign(1.0, result) == math.copysign(1.0, float(value))


@pytest.mark.parametrize("lower,upper", [(0.5, 2.0), (-1.0, 1.0), (-3.5, 7.25)])
def test_reflect_always_lands_inside_the_bounds(lower, upper):
    span = upper - lower
    rng = np.random.RandomState(1)
    for value in rng.uniform(lower - 5 * span, upper + 5 * span, 2000):
        assert lower <= levy._reflect(float(value), lower, upper) <= upper


def test_reflect_matches_reflection_at_the_edges():
    """One reflection past an edge is the mirror image of the excess."""
    assert levy._reflect(-0.25, 0.0, 1.0) == pytest.approx(0.25)
    assert levy._reflect(1.25, 0.0, 1.0) == pytest.approx(0.75)


# --------------------------------------------------------------------------
# np.Inf (was: removed in NumPy 2.0, killing table generation)
# --------------------------------------------------------------------------


def test_table_generation_works_on_current_numpy():
    """`python -m levy build` is the only way to regenerate the tables."""
    assert isinstance(levy._calculate_levy(1.0, 1.5, 0.0), float)


def test_quadrature_density_is_close_to_the_interpolated_value():
    """Ground truth still agrees with the shipped table after the np.inf swap."""
    for x, alpha, beta in [(1.0, 1.5, 0.0), (0.5, 1.2, 0.5), (2.0, 1.8, -0.3)]:
        exact = levy._calculate_levy(np.tan(np.arctan(x)), alpha, beta, False)
        interpolated = levy.levy(np.array([x]), alpha, beta)[0]
        assert interpolated == pytest.approx(exact, rel=1e-3)


# --------------------------------------------------------------------------
# Source hygiene (was: dead references and print() in library code)
# --------------------------------------------------------------------------


def test_no_references_to_functions_that_do_not_exist():
    """`_limits()` and `change_par()` were referenced in commented-out code
    but have never existed anywhere in the package.
    """
    with open(levy.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "_limits()" not in source
    assert "change_par(" not in source


def test_library_code_does_not_print_to_stdout():
    """A library logs; it does not write to stdout. The __main__ block may."""
    with open(levy.__file__, encoding="utf-8") as handle:
        source = handle.read()
    body = source.split('if __name__ == "__main__":')[0]
    offending = [
        line.strip() for line in body.splitlines() if line.strip().startswith("print(")
    ]
    assert not offending, "print() in library code: {}".format(offending)


def test_module_logger_has_a_null_handler():
    """Keeps the library silent unless the application configures logging."""
    import logging

    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logging.getLogger("levy").handlers
    )


def test_evaluating_levy_emits_no_output(capsys):
    levy.levy(np.array([1.0, 2.0, 3.0]), 1.5, 0.0)
    levy.fit_levy(levy.random(1.5, 0.0, 0.0, 1.0, shape=(100,)), par="0")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------------------------
# alpha/beta domain validation (was: out-of-range values silently indexed the
# lookup tables from the wrong end)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.4, 0.0, -1.0, 2.5, 3.0, np.nan])
def test_alpha_outside_the_supported_range_raises(alpha):
    """alpha=0.4 used to give grid index -4 -- a valid *negative* index -- so
    it silently returned the limits for alpha ~ 1.94 and produced
    plausible-looking numbers identical to alpha=0.5. Only positive overflow
    raised, and then as an IndexError from deep inside the lookup.
    """
    with pytest.raises(ValueError, match="alpha"):
        levy.levy(np.array([1.0, 2.0]), alpha, 0.0)


@pytest.mark.parametrize("beta", [-1.5, 1.5, 2.0, np.nan])
def test_beta_outside_the_supported_range_raises(beta):
    with pytest.raises(ValueError, match="beta"):
        levy.levy(np.array([1.0, 2.0]), 1.5, beta)


@pytest.mark.parametrize("alpha", [0.5, 0.7, 1.0, 1.5, 1.9, 2.0])
@pytest.mark.parametrize("beta", [-1.0, 0.0, 1.0])
def test_endpoints_of_the_supported_range_are_accepted(alpha, beta):
    """The bounds are inclusive; validation must not reject the corners."""
    result = levy.levy(np.array([0.5, 1.0]), alpha, beta)
    assert np.all(np.isfinite(result))


def test_validation_message_names_the_offending_parameter():
    with pytest.raises(ValueError) as info:
        levy.levy(np.array([1.0]), 0.3, 0.0)
    message = str(info.value)
    assert "alpha" in message and "0.3" in message


def test_neglog_levy_also_validates():
    """The fit path goes through neglog_levy, so it must reject too."""
    with pytest.raises(ValueError):
        levy.neglog_levy(np.array([1.0]), 0.4, 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# random() at alpha == 2 (was: returned before mu and sigma were applied)
# --------------------------------------------------------------------------


def test_random_at_alpha_2_respects_location_and_scale():
    """The alpha == 2 branch returned early, skipping `return mu + sigma * k`.

    Observed before the fix: random(2.0, 0.0, mu=100.0, sigma=5.0) had
    mean 0.002 and std 1.41 instead of 100 and 7.07.
    """
    np.random.seed(1)
    sample = levy.random(2.0, 0.0, mu=100.0, sigma=5.0, shape=(200000,))
    assert sample.mean() == pytest.approx(100.0, abs=0.1)
    assert sample.std() == pytest.approx(5.0 * np.sqrt(2.0), rel=0.02)


def test_random_at_alpha_2_is_a_location_scale_transform():
    """Scaling must compose the same way it does for every other alpha."""
    np.random.seed(0)
    base = levy.random(2.0, 0.0, mu=0.0, sigma=1.0, shape=(1000,))
    np.random.seed(0)
    shifted = levy.random(2.0, 0.0, mu=5.0, sigma=3.0, shape=(1000,))
    np.testing.assert_allclose(shifted, 5.0 + 3.0 * base, rtol=1e-14, atol=1e-14)


def test_random_at_alpha_2_agrees_with_the_package_cdf():
    """The sampler and the density must describe the same distribution."""
    stats = pytest.importorskip("scipy.stats")
    np.random.seed(2)
    sample = levy.random(2.0, 0.0, mu=1.0, sigma=2.0, shape=(40000,))
    result = stats.kstest(
        sample,
        lambda v: levy.levy(np.asarray(v, dtype=float), 2.0, 0.0, mu=1.0, sigma=2.0, cdf=True),
    )
    assert result.pvalue > 0.01, f"KS p={result.pvalue:.4g}"


# --------------------------------------------------------------------------
# random() near alpha == 1 (was: nudged onto the pole of tan, producing NaN
# and biased samples at beta = +-1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [-1.0, -0.8, -0.3, 0.0, 0.3, 0.8, 1.0])
def test_random_at_alpha_1_produces_no_nan(beta):
    """The old 1e-15 nudge made the base of a fractional power go negative
    for ~0.9% of draws at beta = +-1, yielding NaN.
    """
    np.random.seed(11)
    sample = levy.random(1.0, beta, 0.0, 1.0, shape=(50000,))
    assert np.all(np.isfinite(sample)), (
        f"{100 * np.mean(~np.isfinite(sample)):.3f}% non-finite samples"
    )


@pytest.mark.parametrize("beta", [-1.0, 0.0, 0.8, 1.0])
def test_random_at_alpha_1_matches_the_package_cdf(beta):
    """At beta=1 the surviving samples were also from the wrong distribution:
    KS against this package's own CDF gave p = 3.17e-07 over 200k draws.
    """
    stats = pytest.importorskip("scipy.stats")
    np.random.seed(11)
    sample = levy.random(1.0, beta, 0.0, 1.0, shape=(60000,))
    result = stats.kstest(
        sample,
        lambda v: levy.levy(np.asarray(v, dtype=float), 1.0, beta, cdf=True),
    )
    assert result.pvalue > 1e-3, f"KS p={result.pvalue:.4g} for beta={beta}"


@pytest.mark.parametrize("offset", [-1e-9, -1e-12, 0.0, 1e-12, 1e-9])
def test_alpha_1_nudge_keeps_the_side_the_caller_asked_for(offset):
    """The nudge must not move alpha across 1.

    Forcing both sides up to 1 + radius flips the sign of (1 - alpha) and makes
    random() discontinuous at alpha == 1. Exactly 1.0 still goes up: 1.0 - 1.0
    is +0.0, and copysign follows the sign of the zero.
    """
    alpha = 1.0 + offset
    np.random.seed(0)
    sample = levy.random(alpha, 0.5, 0.0, 1.0, shape=(2000,))
    assert np.isfinite(sample).all()

    expected_side = np.copysign(1.0, offset) if offset else 1.0
    np.random.seed(0)
    mirror = levy.random(
        1.0 + expected_side * levy._ALPHA_1_RADIUS, 0.5, 0.0, 1.0, shape=(2000,)
    )
    np.testing.assert_allclose(sample, mirror, rtol=0, atol=0)


def test_alpha_1_nudge_is_well_conditioned():
    """One ULP of alpha must not move phi appreciably.

    At the old radius of 1e-15, tan(pi*alpha/2) sat 1e-15 from its pole at
    -5.83e+14 and a single ULP moved phi by 11.4%.
    """
    alpha = 1.0 + levy._ALPHA_1_RADIUS
    phi = levy._phi(alpha, 0.3)
    phi_one_ulp_away = levy._phi(np.nextafter(alpha, 2.0), 0.3)
    swing = abs(phi_one_ulp_away - phi) / abs(phi)
    assert swing < 1e-6, f"one ULP in alpha moves _phi by {swing:.2%}"


def test_alpha_1_is_continuous_with_its_neighbourhood():
    """alpha exactly 1 must not be an outlier against alpha just either side."""
    stats = pytest.importorskip("scipy.stats")
    samples = {}
    for alpha in (0.9999, 1.0, 1.0001):
        np.random.seed(5)
        samples[alpha] = levy.random(alpha, 0.7, 0.0, 1.0, shape=(60000,))
    assert stats.ks_2samp(samples[1.0], samples[0.9999]).pvalue > 1e-3
    assert stats.ks_2samp(samples[1.0], samples[1.0001]).pvalue > 1e-3


# --------------------------------------------------------------------------
# Corrupt CDF table cells (was: levy(..., cdf=True) returning 6.4e+307)
# --------------------------------------------------------------------------


def test_loaded_cdf_table_is_a_probability():
    """The table as *used* must be a CDF, even though the shipped file is not.

    Four cells of cdf.npz hold 5.72e+307; they are repaired when the table is
    loaded. The file itself is still bad -- see test_known_bugs.py.
    """
    table = levy._read_from_cache("cdf")
    assert np.isfinite(table).all()
    assert table.min() >= -1e-6
    assert table.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("beta", [-0.74, 0.74])
def test_cdf_is_usable_at_the_formerly_corrupt_cells(beta):
    """Reproducer: this returned 6.4e+307 before the repair."""
    result = levy.levy(np.array([-1.0, 0.0, 1.0]), 0.58, beta, cdf=True)
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.all(np.diff(result) > 0), "CDF must increase in x"


def test_repaired_cells_respect_skew_symmetry():
    """An independent check that the interpolated values are right.

    For a stable law, F(x; alpha, beta) = 1 - F(-x; alpha, -beta). The repair
    fills the two beta = +-0.74 columns separately, so this holding to 1e-9 is
    real evidence and not a tautology.
    """
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    positive = levy.levy(x, 0.58, 0.74, cdf=True)
    negative = levy.levy(-x[::-1], 0.58, -0.74, cdf=True)
    np.testing.assert_allclose(positive, 1.0 - negative[::-1], rtol=0, atol=1e-9)


def test_repaired_column_is_monotone():
    table = levy._read_from_cache("cdf")
    for beta_index in (13, 87):
        column = table[95:105, 4, beta_index]
        assert np.all(np.diff(column) > 0), f"beta index {beta_index} not monotone"


def test_repair_survives_a_column_with_no_usable_cell():
    """Whole column bad -> left is -1 and right is x_size, and the neighbour
    copy used to index one past the end of the table.
    """
    table = np.zeros((4, 2, 2), dtype="float64")
    table[:, 0, 0] = 5.72e307          # every x in this column is unusable
    repaired = levy._repair_table("cdf", table)
    assert np.all((repaired[:, 0, 0] >= 0.0) & (repaired[:, 0, 0] <= 1.0))


def test_table_load_does_not_leak_the_npz_handle():
    """np.load returns a lazy NpzFile; it is now closed after extraction."""
    levy._data_cache.pop("pdf", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        table = levy._read_from_cache("pdf")
        # A leaked NpzFile only warns when it is collected, so force a
        # cycle while the filter is still active. Without this the test
        # passes whether or not the handle was closed.
        gc.collect()
    assert isinstance(table, np.ndarray)
