"""The fit starts from the data's scale, not from a constant.

Upstream issue #20: a user fitting alpha=1.6 with sigma=0.005 found that fits
occasionally came back pinned near a boundary -- ``alpha=2.00, beta=1.00`` or
``alpha=0.513`` -- while scipy's own fit handled the same data.

The cause is not a local optimum. The reported stopping points are not even
stationary: perturbing alpha inward from 2.0 moves the objective *downhill* by
thousands of log-likelihood units. The search simply started at ``sigma = 1``,
200 times wider than the data, and L-BFGS-B did not recover.

Measured over 400 samples of 10,000 points at that scale, 2.25% of fits failed
that way, leaving between 1,290 and 9,054 log-likelihood units unclaimed.

The fix adds a second starting point derived from the data's median and
interquartile range, and keeps whichever optimum is better. The historical
start remains a candidate, which is what makes the change safe: the returned
likelihood can only stay the same or improve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from levy.distribution import neglog_levy
from levy.fitting import _data_scaled_start, _default_start, _starting_points, fit_levy
from levy.sampling import random

PARS = ["0", "1", "M", "A", "B"]

# Seeds that failed before the fix, from the 400-sample sweep.
FAILING_SEEDS = [10007, 10028, 10033, 10052, 10178, 10208, 10241, 10251, 10317]

TRUTH = (1.6, 0.0, 0.0, 0.005)


def _sample(seed, n=10000):
    np.random.seed(seed)
    return random(*TRUTH, shape=(n,))


# --------------------------------------------------------------------------
# the regression from issue #20
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", FAILING_SEEDS)
def test_previously_failing_samples_now_recover_the_parameters(seed):
    data = _sample(seed)
    parameters, nll = fit_levy(data)
    alpha, beta, _, sigma = parameters.get("0")

    assert abs(alpha - 1.6) < 0.25, f"alpha={alpha}, was pinned near a boundary"
    assert abs(beta) < 0.5, f"beta={beta}"
    assert 0.002 < sigma < 0.012, f"sigma={sigma}, truth 0.005"


@pytest.mark.slow
@pytest.mark.parametrize("seed", FAILING_SEEDS[:3])
def test_the_fit_no_longer_stops_short_of_the_optimum(seed):
    # The real symptom: the old fit stopped at a point that was not stationary
    # and left thousands of log-likelihood units on the table.
    data = _sample(seed)
    _, nll = fit_levy(data)
    at_truth = float(neglog_levy(data, *TRUTH).sum())
    assert nll < at_truth + 10.0, (
        f"fit reached {nll}, truth gives {at_truth}; "
        f"{nll - at_truth:.0f} units left unclaimed"
    )


# --------------------------------------------------------------------------
# the guarantee: never worse than before
# --------------------------------------------------------------------------

@pytest.mark.parametrize("par", PARS)
def test_the_historical_start_is_always_a_candidate(par):
    # This is what makes the change safe. Whatever the data-derived start does,
    # the optimum reached from the old constant start is still available, so
    # the returned likelihood cannot get worse.
    data = _sample(0, n=500)
    starts = _starting_points(data, par, dict.fromkeys(("alpha", "beta"), None))
    assert np.allclose(starts[0], _default_start(par, {}))


@pytest.mark.slow
@pytest.mark.parametrize("par", PARS)
def test_multi_start_is_never_worse_than_the_single_start(par):
    from scipy import optimize

    from levy.constants import par_bounds, par_names
    from levy.parametrization import Parameters

    data = _sample(10007, n=4000)

    def single_start_fit():
        values = dict.fromkeys(par_names[par], None)
        parameters = Parameters(par=par, **values)
        temp = Parameters(par=par, **values)

        def objective(v):
            temp.x = v
            return np.sum(neglog_levy(data, *temp.get("0")))

        bounds = tuple(par_bounds[i] for i in parameters.variables)
        result = optimize.minimize(
            objective, parameters.x, method="L-BFGS-B", bounds=bounds)
        parameters.x = result.x
        return objective(parameters.x)

    _, multi = fit_levy(data, par=par)
    assert multi <= single_start_fit() + 1e-9


# --------------------------------------------------------------------------
# the starting point itself
# --------------------------------------------------------------------------

def test_the_scaled_start_tracks_the_data_scale():
    for sigma in (0.001, 0.005, 1.0, 100.0):
        np.random.seed(3)
        data = random(1.5, 0.0, 0.0, sigma, shape=(5000,))
        start = _data_scaled_start(data, "0", {})
        assert start is not None
        # Within a factor of two of the truth, which is all a start needs.
        assert 0.4 * sigma < start[3] < 2.5 * sigma, (
            f"start sigma {start[3]} for data at scale {sigma}")


def test_unit_scale_data_starts_essentially_where_it_used_to():
    # IQR/2 gives 0.973 for a sigma=1 sample, within 3% of the old constant 1.0,
    # so a fit that was already well conditioned barely moves.
    np.random.seed(0)
    data = random(1.5, 0.0, 0.0, 1.0, shape=(20000,))
    start = _data_scaled_start(data, "0", {})
    assert 0.9 < start[3] < 1.1


@pytest.mark.parametrize("par", PARS)
def test_a_pinned_parameter_keeps_its_value_in_every_start(par):
    from levy.constants import par_names

    names = par_names[par]
    for index, name in enumerate(names):
        fixed = dict.fromkeys(names, None)
        fixed[name] = 1.0
        data = _sample(0, n=500)
        for start in _starting_points(data, par, fixed):
            assert start[index] == 1.0, (
                f"par={par}: pinned {name} became {start[index]}")


@pytest.mark.parametrize("par", PARS)
def test_starts_lie_inside_the_optimizer_bounds(par):
    from levy.constants import par_bounds

    for sigma in (1e-4, 1.0, 1e4):
        np.random.seed(1)
        data = random(1.5, 0.0, 0.0, sigma, shape=(2000,))
        for start in _starting_points(data, par, dict.fromkeys(["alpha"], None)):
            for index, (low, high) in enumerate(par_bounds):
                if low is not None:
                    assert start[index] >= low, f"par={par} index {index}"
                if high is not None:
                    assert start[index] <= high, f"par={par} index {index}"


# --------------------------------------------------------------------------
# degenerate data falls back rather than failing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("data", [
    np.ones(50),                    # zero spread
    np.array([0.1, -0.2, 0.3]),     # fewer than four points
    np.array([]),                   # nothing at all
    np.full(20, 5.0),               # constant, non-zero
])
def test_no_scale_can_be_estimated_falls_back_to_the_constant_start(data):
    starts = _starting_points(data, "0", dict.fromkeys(
        ("alpha", "beta", "mu", "sigma"), None))
    assert len(starts) == 1
    assert np.allclose(starts[0], _default_start("0", {}))


def test_a_fit_on_constant_data_still_returns_something():
    parameters, nll = fit_levy(np.ones(50))
    assert np.isfinite(nll)


# --------------------------------------------------------------------------
# the 1.x surface gets the fix too
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_deprecated_fit_levy_is_fixed_as_well():
    import levy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = _sample(10007)
        parameters, _ = levy.fit_levy(data)
    assert abs(parameters.get("0")[0] - 1.6) < 0.25


@pytest.mark.slow
def test_the_typed_api_is_fixed_as_well():
    from levy import api

    result = api.fit(_sample(10033))
    assert abs(result.params.alpha - 1.6) < 0.25
