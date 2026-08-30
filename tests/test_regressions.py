"""Regression tests for defects that have been fixed.

These started life in ``test_known_bugs.py`` as ``xfail(strict=True)`` bug
reports. When a fix lands, its report moves here and loses the marker, so the
file is a running record of what has been repaired and a guard against any of
it coming back. Anything still marked xfail over there is still broken.

Fixed in "Fix NumPy 2 incompatibility, remove dead code, replace prints with
logging" -- see that commit for the reasoning behind each.
"""

from __future__ import annotations

import math
import subprocess
import sys

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
