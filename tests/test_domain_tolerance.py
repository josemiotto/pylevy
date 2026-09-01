"""A conversion that overshoots an endpoint by 44 ULP is not a domain error.

Validating alpha and beta against the tabulated range was the right fix for a
real bug -- alpha below 0.5 used to produce a negative grid index and silently
return values for alpha ~ 1.94. But checking the range *exactly* was too strict
to be correct, and it broke a whole parametrization.

`beta_0 = tan(beta_B * psi(alpha)) / tan(alpha * pi / 2)` is exactly +-1 at
beta_B = +-1 in exact arithmetic. The two tangents are evaluated separately and
their rounding does not cancel, so the quotient can land a few tens of ULP
outside [-1, 1]. An exact check turned that into a ValueError that aborted the
entire fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import levy
from levy.distribution import _DOMAIN_TOLERANCE, _check_alpha_beta
from levy.distribution import levy as levy_pdf
from levy.fitting import fit_levy
from levy.parametrization import convert_to_par0
from levy.sampling import random

PARS = ["0", "1", "M", "A", "B"]


# --------------------------------------------------------------------------
# the regression: a fit in B used to abort
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("alpha", "beta"), [
    (1.7, 0.95),
    (1.9, 1.0),
    (1.95, -1.0),
    (1.99, 0.99),
])
@pytest.mark.parametrize("par", PARS)
def test_fitting_near_the_skewness_boundary_completes(par, alpha, beta):
    # Before the tolerance, par='B' raised
    #   ValueError: beta must be in [-1.0, 1.0], got 1.0000000000000004
    # partway through the optimisation, losing the whole fit.
    np.random.seed(7)
    sample = random(alpha, beta, 0.0, 1.0, shape=(600,))
    parameters, nll = fit_levy(sample, par=par)
    assert np.isfinite(nll)

    # The contract that matters is not that the reported parameters sit inside
    # [-1, 1] to the last bit -- a fit in B can report beta_0 a few ULP outside
    # it, for the same rounding reason -- but that whatever comes out can be
    # fed straight back in.
    fitted = parameters.get("0")
    density = levy_pdf(np.array([0.0, 1.0]), fitted[0], fitted[1])
    assert np.all(np.isfinite(density))


def test_all_five_parametrizations_reach_the_same_optimum_there():
    np.random.seed(7)
    sample = random(1.7, 0.95, 0.0, 1.0, shape=(1500,))
    likelihoods = {par: fit_levy(sample, par=par)[1] for par in PARS}
    best = min(likelihoods.values())
    for par, nll in likelihoods.items():
        assert nll - best < 1e-6, f"par={par} reached {nll}, best was {best}"


# --------------------------------------------------------------------------
# the overshoot itself, measured
# --------------------------------------------------------------------------

def test_only_parametrization_b_overshoots_and_only_by_a_few_ulp():
    # The numbers in the comment on _DOMAIN_TOLERANCE, kept honest. A coarser
    # sweep than the one quoted there, so the test stays quick.
    alphas = np.linspace(0.5, 2.0, 151)
    betas = np.linspace(-1.0, 1.0, 151)

    worst = {par: 0.0 for par in PARS}
    for par in PARS:
        for alpha in alphas:
            for beta in betas:
                converted = convert_to_par0[par](np.array([alpha, beta, 0.0, 1.0]))
                value = converted[1]
                if np.isfinite(value):
                    worst[par] = max(worst[par], abs(value) - 1.0)

    for par in ("0", "1", "M", "A"):
        assert worst[par] <= 0.0, f"par={par} overshoots by {worst[par]:.3e}"

    assert worst["B"] > 0.0, "B no longer overshoots; the tolerance may be dead code"
    assert worst["B"] < _DOMAIN_TOLERANCE, (
        f"B overshoots by {worst['B']:.3e}, which the {_DOMAIN_TOLERANCE:.0e} "
        "tolerance no longer covers"
    )


def test_the_tolerance_has_orders_of_magnitude_of_headroom():
    # Measured worst overshoot is 9.77e-15 on a 601x601 sweep. The tolerance
    # should sit well above that and far below any real mistake.
    assert _DOMAIN_TOLERANCE > 1e-13
    assert _DOMAIN_TOLERANCE < 1e-9


# --------------------------------------------------------------------------
# what the tolerance must not let through
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("alpha", "beta"), [
    (0.4, 0.0),        # the original bug: gave a negative grid index
    (0.49999, 0.0),
    (2.1, 0.0),
    (2.000001, 0.0),
    (1.5, -1.5),
    (1.5, 1.000001),
    (0.0, 0.0),
    (-1.0, 0.0),
])
def test_genuinely_out_of_range_values_are_still_rejected(alpha, beta):
    with pytest.raises(ValueError):
        levy_pdf(np.array([1.0]), alpha, beta)


def test_the_smallest_plausible_mistake_is_still_far_outside_the_tolerance():
    # alpha=0.4 misses by 0.1, which is eleven orders of magnitude larger than
    # the tolerance. There is no overlap between "rounding" and "wrong".
    assert 0.1 / _DOMAIN_TOLERANCE > 1e10


# --------------------------------------------------------------------------
# snapping
# --------------------------------------------------------------------------

def test_values_inside_the_tolerance_are_snapped_to_the_endpoint():
    alpha, beta = _check_alpha_beta(2.0 + 1e-15, 1.0 + 1e-15)
    assert alpha == 2.0
    assert beta == 1.0

    alpha, beta = _check_alpha_beta(0.5 - 1e-15, -1.0 - 1e-15)
    assert alpha == 0.5
    assert beta == -1.0


def test_values_inside_the_range_are_returned_untouched():
    alpha, beta = _check_alpha_beta(1.3456789, -0.4321)
    assert alpha == 1.3456789
    assert beta == -0.4321


def test_snapping_changes_the_result_by_nothing_measurable():
    x = np.linspace(-6.0, 6.0, 121)
    at_edge = levy_pdf(x, 2.0, 1.0)
    just_over = levy_pdf(x, 2.0 + 1e-15, 1.0 + 1e-15)
    assert np.array_equal(at_edge, just_over)


# --------------------------------------------------------------------------
# the deprecated surface behaves the same way
# --------------------------------------------------------------------------

def test_the_1x_function_is_fixed_too():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.random.seed(7)
        sample = levy.random(1.7, 0.95, 0.0, 1.0, shape=(600,))
        parameters, nll = levy.fit_levy(sample, par="B")
    assert np.isfinite(nll)


def test_the_typed_api_still_rejects_out_of_range_before_the_tables():
    from pydantic import ValidationError

    from levy import api

    with pytest.raises(ValidationError):
        api.pdf(np.array([1.0]), alpha=0.4, beta=0.0)


# --------------------------------------------------------------------------
# the typed API used to reject its own results
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("alpha", "beta"), [
    (1.95, -1.0),
    (1.9, 1.0),
    (1.7, 0.95),
])
@pytest.mark.parametrize("par", PARS)
def test_api_fit_accepts_the_result_it_just_computed(par, alpha, beta):
    # StableParams bounds beta at -1, and a fit in B can produce
    # -1.000000000000002, so api.fit raised a ValidationError on a fit that had
    # completed perfectly well. Snapping happens before validation now.
    from levy import api

    np.random.seed(7)
    sample = random(alpha, beta, 0.0, 1.0, shape=(600,))
    result = api.fit(sample, par=par)
    assert -1.0 <= result.params.beta <= 1.0
    assert 0.5 <= result.params.alpha <= 2.0
    assert np.isfinite(result.negative_log_likelihood)


@pytest.mark.parametrize("par", PARS)
def test_from_par_accepts_the_endpoints_of_its_own_parametrization(par):
    from levy import api

    for beta in (-1.0, 1.0):
        for alpha in (0.5, 1.0, 1.5, 1.95, 2.0):
            params = api.StableParams.from_par(alpha, beta, 0.0, 1.0, par=par)
            assert -1.0 <= params.beta <= 1.0
            assert 0.5 <= params.alpha <= 2.0


def test_a_fit_result_round_trips_back_into_the_evaluator():
    from levy import api

    np.random.seed(7)
    sample = random(1.95, -1.0, 0.0, 1.0, shape=(600,))
    params = api.fit(sample, par="B").params
    values = api.pdf(np.array([-1.0, 0.0, 1.0]), **params.as_kwargs())
    assert np.all(np.isfinite(values))
