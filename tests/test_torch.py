"""The torch backend: same numbers, plus gradients.

Two claims, and the second is the reason the backend exists at all.

Parity: the torch implementation is a rewrite, not a wrapper, so it has to be
shown to agree with NumPy rather than assumed to. It is checked here at
``rtol=1e-6`` over a grid spanning both the interpolated region and both tails.

Gradients: ``torch.autograd.gradcheck`` compares the analytic gradient against
finite differences in float64, for all four parameters, for pdf, cdf and the
negative log density. Passing that is what makes "differentiable" a fact rather
than a claim -- and the optimisation test below shows it is *useful*: gradient
descent lands on the same optimum this package's own L-BFGS-B finds.

`tests/test_no_torch.py` covers the other half: an install without torch never
imports it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import levy  # noqa: E402
from levy import api, backends  # noqa: E402
from levy.backends import _numpy as numpy_backend  # noqa: E402
from levy.backends import _torch as torch_backend  # noqa: E402

DOUBLE = torch.float64

# Spans the interpolated region and both power-law tails.
XS = np.concatenate([
    np.linspace(-2000.0, 2000.0, 401),
    np.linspace(-25.0, 25.0, 401),
    np.linspace(-1.0, 1.0, 51),
])

CASES = [
    (1.5, 0.0, 0.0, 1.0),
    (0.7, -0.5, 1.0, 2.0),
    (1.0, 0.4, -1.0, 0.5),
    (1.9, 1.0, -2.0, 0.5),
    (1.3, -1.0, 0.0, 3.0),
    (2.0, 0.0, 0.5, 1.5),
]


@pytest.fixture(autouse=True)
def automatic_backend():
    """Leave the global selection as it was, whatever a test does to it."""
    previous = backends.set_backend(None)
    yield
    backends.set_backend(previous)


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), CASES)
@pytest.mark.parametrize("function", ["pdf", "cdf", "neglog"])
def test_matches_the_numpy_backend(function, alpha, beta, mu, sigma):
    expected = getattr(numpy_backend, function)(XS, alpha, beta, mu, sigma)
    got = getattr(torch_backend, function)(
        torch.tensor(XS, dtype=DOUBLE), alpha, beta, mu, sigma).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), CASES)
def test_agreement_is_far_tighter_than_the_contract(alpha, beta, mu, sigma):
    # The stated contract is 1e-6. What is actually observed is accumulated
    # float64 rounding, because both paths do the same arithmetic in the same
    # order over the same float32 grid -- nothing systematic. This second,
    # tighter assertion exists so that if the two ever genuinely diverge, the
    # loose 1e-6 above does not hide it.
    #
    # The bound is 1e-10 rather than the ~1e-16 a single operation would give:
    # the interpolation is a 64-term weighted sum, and torch and NumPy do not
    # order or vectorise it identically. Measured worst case is 2.6e-16 on
    # macOS/x86 and 1.4e-12 on the Linux CI runner -- a libm difference, not a
    # divergence. 1e-10 keeps two orders of magnitude of headroom over the
    # worse of those while staying four below the contract.
    expected = numpy_backend.pdf(XS, alpha, beta, mu, sigma)
    got = torch_backend.pdf(torch.tensor(XS, dtype=DOUBLE),
                            alpha, beta, mu, sigma).numpy()
    finite = np.abs(expected) > 1e-300
    relative = np.abs(got[finite] - expected[finite]) / np.abs(expected[finite])
    assert relative.max() < 1e-10, f"max relative deviation {relative.max():.3e}"


def test_float32_tensors_are_accepted_and_stay_float32():
    result = torch_backend.pdf(torch.tensor(XS, dtype=torch.float32), 1.5, 0.0)
    assert result.dtype == torch.float32
    expected = numpy_backend.pdf(XS, 1.5, 0.0)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

GRADCHECK_CASES = [
    (1.53, 0.17, 0.21, 1.07),
    (0.83, -0.41, -0.50, 2.10),
    (1.87, 0.63, 1.30, 0.40),
]


@pytest.mark.parametrize(("alpha", "beta", "mu", "sigma"), GRADCHECK_CASES)
@pytest.mark.parametrize("function", ["pdf", "cdf", "neglog"])
def test_gradcheck(function, alpha, beta, mu, sigma):
    x = torch.tensor([-4.3, -1.1, 0.3, 1.9, 6.7], dtype=DOUBLE)
    evaluate = getattr(torch_backend, function)

    args = tuple(torch.tensor(v, dtype=DOUBLE, requires_grad=True)
                 for v in (alpha, beta, mu, sigma))
    assert torch.autograd.gradcheck(
        lambda a, b, m, s: evaluate(x, a, b, m, s), args,
        eps=1e-6, atol=1e-5, rtol=1e-3)


def test_a_gradient_reaches_every_parameter():
    x = torch.tensor([-2.0, 0.5, 3.0], dtype=DOUBLE)
    params = {name: torch.tensor(value, dtype=DOUBLE, requires_grad=True)
              for name, value in
              (("alpha", 1.5), ("beta", 0.2), ("mu", 0.1), ("sigma", 1.1))}

    torch_backend.neglog(x, **params).sum().backward()

    for name, tensor in params.items():
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad), f"non-finite gradient for {name}"
        assert tensor.grad != 0.0, f"zero gradient for {name}"


def test_gradient_descent_finds_the_same_optimum_as_l_bfgs_b():
    from levy.sampling import random

    np.random.seed(0)
    sample = random(1.6, 0.3, 0.5, 1.2, shape=(4000,))
    tensor_sample = torch.tensor(sample, dtype=DOUBLE)

    guess = torch.tensor([1.4, 0.0, 0.0, 1.0], dtype=DOUBLE, requires_grad=True)
    optimizer = torch.optim.Adam([guess], lr=0.03)
    for _ in range(400):
        optimizer.zero_grad()
        loss = torch_backend.neglog(
            tensor_sample,
            guess[0].clamp(0.55, 1.95), guess[1].clamp(-0.95, 0.95),
            guess[2], guess[3].clamp(0.05, 20.0)).sum()
        loss.backward()
        optimizer.step()

    by_gradient = guess.detach().numpy()
    by_lbfgsb = np.asarray(api.fit(sample).params.as_tuple())

    # Two different optimizers on the same objective; they should agree far
    # more closely than either agrees with the true parameters.
    np.testing.assert_allclose(by_gradient, by_lbfgsb, atol=5e-3)


def test_the_numpy_backend_has_no_gradients_to_offer():
    # Stated as a test so the difference between the two is on the record.
    values = numpy_backend.pdf(XS, 1.5, 0.0)
    assert isinstance(values, np.ndarray)
    assert not hasattr(values, "requires_grad")


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_a_tensor_argument_selects_torch():
    assert backends.get(None, torch.tensor([1.0])).name == "torch"
    assert backends.get(None, np.array([1.0])).name == "numpy"


def test_a_tensor_parameter_also_selects_torch():
    result = api.pdf(np.array([1.0, 2.0]), alpha=torch.tensor(1.5), beta=0.0)
    assert isinstance(result, torch.Tensor)


def test_an_explicit_choice_wins():
    assert backends.get("numpy", torch.tensor([1.0])).name == "numpy"
    assert backends.get("torch", np.array([1.0])).name == "torch"


def test_set_backend_applies_globally_until_reset():
    levy.set_backend("torch")
    try:
        assert isinstance(api.pdf(np.array([1.0]), alpha=1.5, beta=0.0), torch.Tensor)
    finally:
        levy.set_backend(None)
    assert isinstance(api.pdf(np.array([1.0]), alpha=1.5, beta=0.0), np.ndarray)


def test_using_restores_the_previous_selection_even_on_failure():
    levy.set_backend("numpy")
    with pytest.raises(RuntimeError), levy.using("torch"):
        raise RuntimeError("boom")
    assert backends.get(None, torch.tensor([1.0])).name == "numpy"
    levy.set_backend(None)


def test_an_unknown_backend_is_rejected_immediately():
    with pytest.raises(ValueError, match="unknown backend"):
        backends.set_backend("jax")
    # and the selection is unchanged
    assert backends.get(None, np.array([1.0])).name == "numpy"


def test_torch_results_are_bit_identical_through_api_and_backend():
    direct = torch_backend.pdf(torch.tensor(XS, dtype=DOUBLE), 1.5, 0.2, 0.1, 1.1)
    through_api = api.pdf(torch.tensor(XS, dtype=DOUBLE),
                          alpha=1.5, beta=0.2, mu=0.1, sigma=1.1)
    assert torch.equal(direct, through_api)


def test_logpdf_through_the_api_is_the_negated_neglog():
    x = torch.tensor(XS, dtype=DOUBLE)
    assert torch.equal(api.logpdf(x, alpha=1.5, beta=0.0),
                       -torch_backend.neglog(x, 1.5, 0.0))


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_validation_still_runs_on_tensor_input():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.pdf(torch.tensor([1.0]), alpha=0.3, beta=0.0)


def test_tensor_parameters_in_another_parametrization_are_refused():
    with pytest.raises(ValueError, match="parametrization 0"):
        api.pdf(torch.tensor([1.0]),
                alpha=torch.tensor(1.5), beta=torch.tensor(0.5), par="B")


def test_the_refusal_says_why_and_what_to_do():
    with pytest.raises(ValueError, match="cut the gradient"):
        api.pdf(torch.tensor([1.0]), alpha=torch.tensor(1.5), beta=0.0, par="1")


def test_float_parameters_in_another_parametrization_are_fine():
    # Nothing to cut, so nothing to refuse.
    result = api.pdf(torch.tensor([1.0]), alpha=1.5, beta=0.5, par="B")
    assert isinstance(result, torch.Tensor)


def test_the_deprecated_1x_functions_stay_on_numpy():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = levy.levy(np.array([1.0]), 1.5, 0.0)
    assert isinstance(result, np.ndarray)
