"""The typed API: parity with 1.x, validation, and the shapes of the results.

The point of every parity test here is that `api` adds a boundary and changes
no numbers. Where a 1.x call and an `api` call describe the same thing, the
floats must be bit-identical, not merely close.
"""

from __future__ import annotations

import numpy as np
import pytest

import levy

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from levy import api  # noqa: E402

PARAMETRIZATIONS = ["0", "1", "M", "A", "B"]

GRID = [
    (0.7, -0.5, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (1.3, 0.9, -2.0, 0.5),
    (1.5, 0.0, 0.0, 1.0),
    (1.9, -1.0, 3.0, 2.0),
    (2.0, 0.0, 0.0, 1.0),
]

XS = np.concatenate([
    np.linspace(-40.0, 40.0, 161),
    np.linspace(-2.0, 2.0, 41),
])


# --------------------------------------------------------------------------
# parity with the 1.x functions
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), GRID)
def test_pdf_is_bit_identical_to_levy(alpha, beta, mu, sigma):
    expected = levy.levy(XS, alpha, beta, mu, sigma, cdf=False)
    got = api.pdf(XS, alpha=alpha, beta=beta, mu=mu, sigma=sigma)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), GRID)
def test_cdf_is_bit_identical_to_levy(alpha, beta, mu, sigma):
    expected = levy.levy(XS, alpha, beta, mu, sigma, cdf=True)
    got = api.cdf(XS, alpha=alpha, beta=beta, mu=mu, sigma=sigma)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), GRID)
def test_logpdf_is_the_negation_of_neglog_levy(alpha, beta, mu, sigma):
    # Note the sign: neglog_levy returns -log(pdf), logpdf returns +log(pdf).
    expected = -levy.neglog_levy(XS, alpha, beta, mu, sigma)
    got = api.logpdf(XS, alpha=alpha, beta=beta, mu=mu, sigma=sigma)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), GRID)
def test_rvs_is_bit_identical_to_random(alpha, beta, mu, sigma):
    np.random.seed(4242)
    expected = levy.random(alpha, beta, mu, sigma, shape=(500,))
    got = api.rvs(alpha=alpha, beta=beta, mu=mu, sigma=sigma,
                  size=500, random_state=4242)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("par", PARAMETRIZATIONS)
def test_fit_agrees_with_fit_levy(par):
    np.random.seed(11)
    sample = levy.random(1.5, 0.0, 0.0, 1.0, shape=(300,))

    expected, expected_nll = levy.fit_levy(sample, par=par)
    result = api.fit(sample, par=par)

    assert np.array_equal(np.asarray(result.params.as_tuple()), expected.get("0"))
    assert result.negative_log_likelihood == expected_nll
    assert result.parametrization == par


# --------------------------------------------------------------------------
# validation at the boundary
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("field", "value"), [
    ("alpha", 0.4),      # below the tabulated range
    ("alpha", 2.1),      # above it
    ("beta", -1.5),
    ("beta", 1.5),
    ("sigma", 0.0),      # a zero scale divides by zero downstream
    ("sigma", -1.0),
])
def test_out_of_range_parameters_are_rejected(field, value):
    kwargs = {"alpha": 1.5, "beta": 0.0, "mu": 0.0, "sigma": 1.0}
    kwargs[field] = value
    with pytest.raises(ValidationError):
        api.StableParams(**kwargs)


def test_validation_error_names_the_offending_field():
    with pytest.raises(ValidationError, match="alpha"):
        api.StableParams(alpha=0.4, beta=0.0)


@pytest.mark.parametrize("function", [api.pdf, api.cdf, api.logpdf])
def test_functions_validate_before_touching_the_tables(function):
    with pytest.raises(ValidationError):
        function(np.array([1.0]), alpha=0.4, beta=0.0)


def test_rvs_validates_too():
    with pytest.raises(ValidationError):
        api.rvs(alpha=0.4, beta=0.0, size=3)


def test_params_are_frozen():
    params = api.StableParams(alpha=1.5, beta=0.0)
    with pytest.raises(ValidationError):
        params.alpha = 1.6


def test_params_are_hashable():
    a = api.StableParams(alpha=1.5, beta=0.0)
    b = api.StableParams(alpha=1.5, beta=0.0)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        api.StableParams(alpha=1.5, beta=0.0, skew=0.3)


def test_fit_rejects_a_misspelt_parameter_name():
    # fit_levy takes **kwargs and quietly ignores anything it does not know,
    # so `fit_levy(x, beta_=0)` silently fits beta instead of pinning it.
    np.random.seed(3)
    sample = levy.random(1.5, 0.0, shape=(50,))
    with pytest.raises(TypeError, match="beta_"):
        api.fit(sample, beta_=0.0)


def test_fit_levy_still_ignores_it__documenting_the_difference():
    np.random.seed(3)
    sample = levy.random(1.5, 0.0, shape=(50,))
    free, _ = levy.fit_levy(sample)
    typoed, _ = levy.fit_levy(sample, beta_=0.0)
    assert np.array_equal(typoed.get("0"), free.get("0"))


# --------------------------------------------------------------------------
# parametrization handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("par", PARAMETRIZATIONS)
def test_from_par_and_to_par_round_trip(par):
    original = api.StableParams(alpha=1.6, beta=0.4, mu=0.3, sigma=1.2)
    written = original.to_par(par)
    back = api.StableParams.from_par(*written, par=par)
    np.testing.assert_allclose(back.as_tuple(), original.as_tuple(), rtol=1e-12)


@pytest.mark.parametrize("par", PARAMETRIZATIONS)
def test_from_par_matches_parameters_convert(par):
    values = np.array([1.6, 0.4, 0.3, 1.2])
    expected = levy.Parameters.convert(values, par, "0")
    got = api.StableParams.from_par(*values, par=par)
    assert np.array_equal(np.asarray(got.as_tuple()), np.asarray(expected, dtype="d"))


@pytest.mark.parametrize("par", PARAMETRIZATIONS)
def test_pdf_accepts_parameters_in_any_parametrization(par):
    reference = api.StableParams(alpha=1.6, beta=0.4, mu=0.3, sigma=1.2)
    written = reference.to_par(par)
    got = api.pdf(XS, alpha=written[0], beta=written[1],
                  mu=written[2], sigma=written[3], par=par)
    expected = api.pdf(XS, **reference.as_kwargs())
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_conversion_out_of_range_is_caught_at_the_boundary():
    # A beta_B near the edge converts to a beta_0 outside [-1, 1]; 1.x would
    # hand that straight to _interpolate, which clamps the grid index and
    # returns a number that looks fine.
    with pytest.raises(ValidationError):
        api.StableParams.from_par(1.9, 3.0, 0.0, 1.0, par="B")


# --------------------------------------------------------------------------
# shapes, dtypes and the seeding contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("size", "expected"), [
    (None, ()),
    (5, (5,)),
    ((3,), (3,)),
    ((2, 3), (2, 3)),
])
def test_rvs_size_maps_onto_shape(size, expected):
    drawn = api.rvs(alpha=1.5, beta=0.0, size=size, random_state=1)
    assert np.shape(drawn) == expected


def test_rvs_with_a_seed_leaves_the_global_stream_untouched():
    np.random.seed(99)
    before = np.random.random(3)

    np.random.seed(99)
    api.rvs(alpha=1.5, beta=0.0, size=100, random_state=0)
    after = np.random.random(3)

    assert np.array_equal(before, after)


def test_rvs_without_a_seed_advances_the_global_stream():
    np.random.seed(5)
    first = api.rvs(alpha=1.5, beta=0.0, size=3)
    second = api.rvs(alpha=1.5, beta=0.0, size=3)
    assert not np.array_equal(first, second)


def test_scalar_input_gives_a_python_float():
    value = api.pdf(1.0, alpha=1.5, beta=0.0)
    assert isinstance(value, float)
    assert not isinstance(value, np.ndarray)


def test_array_input_gives_an_array():
    values = api.pdf(np.array([1.0, 2.0]), alpha=1.5, beta=0.0)
    assert isinstance(values, np.ndarray)
    assert values.shape == (2,)


def test_functions_are_keyword_only():
    with pytest.raises(TypeError):
        api.pdf(np.array([1.0]), 1.5, 0.0)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

def test_fit_result_carries_the_parametrization_and_converts_back():
    np.random.seed(7)
    sample = levy.random(1.5, 0.2, 0.0, 1.0, shape=(400,))
    result = api.fit(sample, par="1")

    assert result.parametrization == "1"
    np.testing.assert_allclose(
        result.as_par("1"), result.params.to_par("1"), rtol=0, atol=0)


def test_fit_recovers_parameters_it_generated_from():
    np.random.seed(2024)
    sample = levy.random(1.5, 0.0, 0.0, 1.0, shape=(4000,))
    fitted = api.fit(sample).params
    assert abs(fitted.alpha - 1.5) < 0.15
    assert abs(fitted.beta) < 0.25
    assert abs(fitted.sigma - 1.0) < 0.15


def test_fit_result_is_frozen():
    np.random.seed(1)
    sample = levy.random(1.5, 0.0, shape=(50,))
    result = api.fit(sample)
    with pytest.raises(ValidationError):
        result.negative_log_likelihood = 0.0


def test_the_new_api_emits_no_deprecation_warnings():
    # The 2.0 shims land in a later PR and will make the 1.x names warn. This
    # is what keeps the new spelling silent when they do.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        api.pdf(np.array([1.0]), alpha=1.5, beta=0.0)
        api.cdf(np.array([1.0]), alpha=1.5, beta=0.0)
        api.logpdf(np.array([1.0]), alpha=1.5, beta=0.0)
        api.rvs(alpha=1.5, beta=0.0, size=2, random_state=0)
        api.fit(np.array([0.1, -0.4, 1.2, 0.3, -1.1]))


def test_package_ships_py_typed():
    import pathlib

    marker = pathlib.Path(levy.__file__).with_name("py.typed")
    assert marker.exists(), "py.typed is what makes the annotations visible downstream"


def test_api_is_reachable_from_the_package_root():
    import importlib

    module = importlib.import_module("levy")
    assert module.api is api
