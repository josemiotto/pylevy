#    Copyright (C) 2017 José M. Miotto
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""Maximum-likelihood fitting."""

import numpy as np
from scipy import optimize

from levy.constants import default, par_bounds, par_names
from levy.distribution import neglog_levy
from levy.parametrization import Parameters

__all__ = ['fit_levy']

# The interquartile range of a standard (sigma = 1) stable variate, at the
# alpha the search starts from. Dividing the sample IQR by this turns it into a
# starting sigma.
#
# Measured over 20,000-point samples: IQR is 1.945 at alpha = 1.5 and moves
# only between 1.90 and 2.35 across the whole supported alpha range, so a
# single constant is enough for a starting point. 2.0 is used rather than
# 1.945 because it is the round number in the middle of that range and gives
# 0.973 for unit-scale data -- within 3% of the 1.0 this replaced, so a fit
# that was already well conditioned barely moves.
_IQR_OF_STANDARD_STABLE = 2.0


def _starting_points(x, par, fixed):
    """Return the starting points to run the search from, best first.

    Parameters
    ----------
    x : ndarray
        The sample.
    par : {'0', '1', 'M', 'A', 'B'}
        Parametrization the search runs in.
    fixed : dict
        The caller's keyword arguments, so that pinned parameters are left
        exactly as they were given.

    Returns
    -------
    list of ndarray
        One or two starting vectors in `par`. The historical constant start is
        always the first, so the search can only find a better optimum than it
        used to, never a worse one.
    """
    points = [_default_start(par, fixed)]
    scaled = _data_scaled_start(x, par, fixed)
    if scaled is not None and not np.allclose(scaled, points[0], rtol=1e-9, atol=0.0):
        points.append(scaled)
    return points


def _default_start(par, fixed):
    """Return the historical constant starting point.

    Parameters
    ----------
    par : {'0', '1', 'M', 'A', 'B'}
        Parametrization.
    fixed : dict
        Pinned parameters, which override the default.

    Returns
    -------
    ndarray
        ``[1.5, 0, 0, 1]`` in the parametrization's own units, with pinned
        values substituted.
    """
    names = par_names[par]
    return np.array([
        fixed[name] if fixed.get(name) is not None else default[par][name]
        for name in names
    ], dtype='d')


def _data_scaled_start(x, par, fixed):
    """Choose a second starting point from the scale of the data.

    Parameters
    ----------
    x : ndarray
        The sample.
    par : {'0', '1', 'M', 'A', 'B'}
        Parametrization the search runs in.
    fixed : dict
        The caller's keyword arguments, so that pinned parameters are left
        exactly as they were given.

    Returns
    -------
    ndarray or None
        Starting values in `par`, or None when no scale can be estimated.

    Notes
    -----
    The only starting point used to be the constant ``[1.5, 0, 0, 1]``, in
    every parametrization, whatever the data looked like. For a sample with
    ``sigma = 0.005`` that is 200 times too wide, and L-BFGS-B does not recover
    from it: measured over 400 samples of 10,000 points at that scale, 6% of
    fits stopped at a point that was not even stationary, leaving up to 9,000
    log-likelihood units on the table. See
    https://github.com/josemiotto/pylevy/issues/20.

    This start is *added* rather than substituted. Converting a location-scale
    start into M, A or B -- where the third and fourth parameters are not a
    location and a scale -- can leave the optimizer worse conditioned than the
    constant did, so both are tried and the better optimum wins.

    Location and scale are estimated with the median and the interquartile
    range rather than the mean and standard deviation, neither of which exists
    for ``alpha < 2``.
    """
    x = np.asarray(x, dtype='d')
    finite = x[np.isfinite(x)]
    if finite.size < 4:
        return None

    lower, upper = np.percentile(finite, [25, 75])
    spread = float(upper - lower) / _IQR_OF_STANDARD_STABLE
    centre = float(np.median(finite))
    if not np.isfinite(spread) or spread <= 0 or not np.isfinite(centre):
        return None

    # Built in parametrization 0 -- the only one in which the third and fourth
    # parameters are plainly a location and a scale -- and then converted, so
    # that this is correct for M, A and B without special-casing them.
    start = Parameters.convert(
        np.array([default['0']['alpha'], default['0']['beta'], centre, spread]),
        '0', par)
    if not np.all(np.isfinite(start)):
        return None

    # A pinned parameter keeps the value the caller gave it.
    names = par_names[par]
    for index, name in enumerate(names):
        if fixed.get(name) is not None:
            start[index] = fixed[name]

    # The start has to be inside the box L-BFGS-B is given.
    for index in range(4):
        low, high = par_bounds[index]
        if low is not None:
            start[index] = max(low, start[index])
        if high is not None:
            start[index] = min(high, start[index])
    return start


def fit_levy(x, par='0', **kwargs):
    """Estimate the parameters of a Levy stable distribution by maximum likelihood.

    By default, searches all possible Levy stable distributions. The search can
    be restricted by pinning one or more parameters, in any of the five
    available parametrizations.

    Parameters
    ----------
    x : array_like
        Values to be fitted.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the fit is carried out and reported in.
    **kwargs
        Any of the names in ``par_names[par]``. A parameter given a value is
        held fixed; one left out is estimated.

    Returns
    -------
    parameters : levy.parametrization.Parameters
        The fitted parameters.
    nll : float
        Negative log likelihood of the data under them.

    See Also
    --------
    levy.distribution.neglog_levy : The objective being minimised.

    Notes
    -----
    The objective is minimised with L-BFGS-B over the free parameters only, box
    constrained by ``par_bounds``. Since the likelihood is evaluated by
    interpolating a lookup table, the optimum's last digits are not portable
    across platforms; compare fitted parameters with a tolerance, never for
    equality.

    Examples
    --------
    >>> from levy.sampling import random
    >>> np.random.seed(0)
    >>> x = random(1.5, 0.0, 0.0, 1.0, shape=(200,))

    Fit a stable distribution to x:

    >>> parameters, nll = fit_levy(x)
    >>> parameters
    par=0, alpha=1.52, beta=-0.08, mu=0.05, sigma=0.99
    >>> bool(abs(nll - 402.3715) < 1e-3)
    True

    Fit a symmetric stable distribution to x:

    >>> fit_levy(x, beta=0.0)[0]
    par=0, alpha=1.53, beta=0.00, mu=0.03, sigma=0.99

    Fit a symmetric distribution centred on zero:

    >>> fit_levy(x, beta=0.0, mu=0.0)[0]
    par=0, alpha=1.53, beta=0.00, mu=0.00, sigma=0.99

    Fit a Cauchy distribution:

    >>> fit_levy(x, alpha=1.0, beta=0.0)[0]
    par=0, alpha=1.00, beta=0.00, mu=0.10, sigma=0.90
    """
    values = {par_name: kwargs.get(par_name) for par_name in par_names[par]}

    parameters = Parameters(par=par, **values)
    temp = Parameters(par=par, **values)

    def neglog_density(param):
        """Total negative log likelihood at one point of the free subspace.

        Parameters
        ----------
        param : ndarray
            Values for the free parameters only, in ``parameters.variables``
            order.

        Returns
        -------
        float
            The objective L-BFGS-B minimises.
        """
        temp.x = param
        alpha, beta, mu, sigma = temp.get('0')
        return np.sum(neglog_levy(x, alpha, beta, mu, sigma))

    bounds = tuple(par_bounds[i] for i in parameters.variables)

    # Run from each candidate start and keep whichever optimum is best. The
    # historical start is always among the candidates, so this cannot return a
    # worse likelihood than it used to -- only the same one or a better one.
    best_x, best_value = None, np.inf
    for start in _starting_points(x, par, values):
        parameters._x = np.array(start, dtype='d')
        temp._x = np.array(start, dtype='d')
        result = optimize.minimize(
            neglog_density, parameters.x, method='L-BFGS-B', bounds=bounds)
        parameters.x = result.x
        value = neglog_density(parameters.x)
        if value < best_value:
            best_x, best_value = np.array(parameters._x), value

    parameters._x = best_x
    # The stored value, not a re-evaluation. Reading `parameters.x` back out
    # and re-running the objective folds the free parameters through
    # `f_bounds` a second time, which perturbs the last bits -- enough to make
    # the returned likelihood differ from the one actually achieved by ~3e-11,
    # and enough to make "never worse than before" false by that much.
    return parameters, best_value
