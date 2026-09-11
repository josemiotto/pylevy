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


@pytest.mark.parametrize("pinned", [False, True], ids=["free", "alpha+beta pinned"])
@pytest.mark.parametrize("par", PARS)
def test_the_scaled_start_round_trips_in_every_parametrization(par, pinned):
    """The start is built as (alpha, beta, median, IQR/2) in parametrization 0
    and converted, so converting it back must recover the data's location and
    scale whatever the parametrization and whatever is pinned. With alpha and
    beta pinned, the conversion used to run with the *defaults* and the pinned
    values were written over the result afterwards -- in 1, M, A and B the
    location and scale it then carried belonged to a different distribution.
    """
    from levy.constants import par_names
    from levy.parametrization import Parameters

    np.random.seed(11)
    data = random(1.7, 0.6, 0.0, 0.01, shape=(20000,))
    names = par_names[par]
    fixed = dict.fromkeys(names, None)
    if pinned:
        fixed[names[0]] = 1.7
        fixed[names[1]] = 0.6

    start = _data_scaled_start(data, par, fixed)
    assert start is not None
    if pinned:
        assert start[0] == 1.7 and start[1] == 0.6

    back = Parameters.convert(start, par, "0")
    centre = np.median(data)
    spread = (np.percentile(data, 75) - np.percentile(data, 25)) / 2.0
    np.testing.assert_allclose(back[2:], [centre, spread], rtol=1e-9)


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


def test_non_finite_values_are_left_out_of_the_scale_estimate():
    """The start is built from the finite values only: a few NaN or inf in
    the sample must neither poison the median and IQR nor turn the start
    into NaN, which would have made the candidate silently unusable.
    """
    np.random.seed(5)
    clean = random(1.5, 0.0, 0.0, 0.01, shape=(5000,))
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf, np.nan]])
    fixed = dict.fromkeys(("alpha", "beta", "mu", "sigma"), None)
    np.testing.assert_array_equal(
        _data_scaled_start(dirty, "0", fixed), _data_scaled_start(clean, "0", fixed))


def test_too_few_finite_values_fall_back_to_the_constant_start():
    data = np.array([0.1, np.nan, -0.2, np.inf, 0.3, np.nan])   # three finite
    starts = _starting_points(data, "0", dict.fromkeys(("alpha", "beta", "mu", "sigma"), None))
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


# --------------------------------------------------------------------------
# a non-finite objective
# --------------------------------------------------------------------------

def test_a_non_finite_objective_still_returns_a_usable_result(monkeypatch):
    """With every candidate start scoring NaN, `value < best` was never true,
    so best_x stayed None and the returned Parameters held no values at all --
    a regression from the single-start code, which returned its one result
    with the NaN objective attached. It does again.
    """
    import levy.fitting as fitting

    monkeypatch.setattr(
        fitting, "neglog_levy", lambda x, *args: np.full(np.shape(x), np.nan))
    parameters, nll = fit_levy(np.array([0.1, -0.4, 2.0, 0.7, -1.1]))
    assert np.isnan(nll)
    assert parameters._x is not None
    assert parameters.x.shape == (4,)
    repr(parameters)  # used to raise from the None


def test_a_pinned_alpha_of_one_in_b_fits_without_a_warning():
    """B is singular at alpha = 1, so building the data-scaled start there
    divides 0/0. That used to escape as a RuntimeWarning -- an error under
    -W error -- even though the constant start was still available and the
    fit went through. The candidate is now dropped quietly.
    """
    np.random.seed(1)
    data = random(1.2, 0.3, 0.0, 1.0, shape=(300,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parameters, nll = fit_levy(data, par="B", alpha=1.0)
    assert np.isfinite(nll)
    assert parameters.get("B")[0] == 1.0


@pytest.mark.parametrize("par", PARS)
def test_a_pinned_scale_beyond_the_fit_bounds_is_kept(par):
    """The bound clamp rewrote pinned components too: fit_levy(x, sigma=1e11)
    could pick the data-scaled candidate and report sigma=1e10, the upper
    fit bound, instead of the value the caller pinned. A pinned parameter is
    outside the optimizer's box and keeps exactly what it was given.
    """
    from levy.constants import par_names

    np.random.seed(2)
    data = random(1.5, 0.0, 0.0, 1.0, shape=(400,))
    scale = par_names[par][3]
    fixed = dict.fromkeys(par_names[par], None)
    fixed[scale] = 1e11
    start = _data_scaled_start(data, par, fixed)
    assert start is not None and start[3] == 1e11

    parameters, _ = fit_levy(data, par=par, **{scale: 1e11})
    assert parameters.get(par)[3] == 1e11
