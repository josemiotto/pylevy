# -*- encoding: utf-8 -*-
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

"""
This is a package for calculation of Levy stable distributions
(probability density function and cumulative density function) and for
fitting these distributions to data.

It operates by interpolating values from a table, as direct computation
of these distributions requires a lengthy numerical integration. This
interpolation scheme allows fast fitting of data by Maximum Likelihood.

Notes on the parameters
-----------------------
- the parameters of the Levy stable distribution can be given in multiple ways: parametrizations.
  Here, you can use both parametrizations 0 and 1, in the notation of Nolan
  (http://fs2.american.edu/jpnolan/www/stable/stable.html) and
  parametrizations A, B and M from Zolotarev (Chance and Stability).

- Nolan parametrizations are a bit easier to understand.
  Parametrization 0 is typically preferred for numerical calculations, and
  has :math:`E(X)=\\delta_0-\\beta\\gamma\\tan(\\pi\\alpha/2)` while
  parametrization 1 is preferred for better intuition, since :math:`E(X)=\\delta_1`.

- parametrizations are dealt automatically by the module, you just need
  to specify which one you want to use. Also, you can use the function
  Parameters.convert to transform the parameters from one parametrization
  to another. The module uses internally parametrization 0.

- pylevy does not support alpha values lower than 0.5.
"""

import logging
import sys
import os
import numpy as np
from scipy.special import gamma
from scipy import optimize

__version__ = "1.1"

# A library should not write to stdout. The NullHandler keeps this silent
# unless the application configures logging, per the stdlib logging guidance.
logger = logging.getLogger(__name__)
# Guarded: this module can be imported more than once under a different
# name, or reloaded, and an unguarded add would stack a NullHandler per
# import on the same logger object.
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

# Some constants of the program.
# Dimensions: 0 - x, 1 - alpha, 2 - beta
size = (200, 76, 101)  # size of the grid (xs, alpha, beta)
_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])  # lower limit of parameters
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])  # upper limit of parameters

#: How far random() moves alpha away from 1 before sampling a skewed draw, to
#: keep tan(pi*alpha/2) off its pole. Module level so that the tests assert
#: against the value the sampler actually uses instead of a copy of it; see
#: random() for why 1e-8, and why nothing here is tuned.
_ALPHA_1_RADIUS = 1e-8

par_bounds = ((_lower[1], _upper[1]), (_lower[2], _upper[2]), (None, None), (1e-6, 1e10))  # parameter bounds for fit.
par_names = {  # names of the parameters
    '0': ['alpha', 'beta', 'mu', 'sigma'],
    '1': ['alpha', 'beta', 'mu', 'sigma'],
    'M': ['alpha', 'beta', 'gamma', 'lambda'],
    'A': ['alpha', 'beta', 'gamma', 'lambda'],
    'B': ['alpha', 'beta', 'gamma', 'lambda']
}
default = [1.5, 0.0, 0.0, 1.0]  # default values of the parameters for fit.
default = {k: {par_names[k][i]: default[i] for i in range(4)} for k in par_names.keys()}
f_bounds = [
    lambda x: _reflect(x, *par_bounds[0]),
    lambda x: _reflect(x, *par_bounds[1]),
    lambda x: x,
    lambda x: _reflect(x, *par_bounds[3])
]
f_bounds = {k: {par_names[k][i]: f_bounds[i] for i in range(4)} for k in par_names.keys()}

ROOT = os.path.dirname(os.path.abspath(__file__))
_data_cache = {}

_TABLE_NAMES = ('pdf', 'cdf', 'lower_limit', 'upper_limit')


def user_cache_dir():
    """ Per-user directory where regenerated tables are looked for.

    Resolved without a third-party dependency: XDG_CACHE_HOME or ~/.cache on
    Unix, ~/Library/Caches on macOS, LOCALAPPDATA on Windows.
    """
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser(r'~\AppData\Local')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'pylevy')


def data_dir(writable=False):
    """ Directory the lookup tables are read from.

    Search order: ``$LEVY_DATA_DIR``, then the user cache directory if it holds
    a complete set, then the tables shipped inside the package.

    `writable=True` returns where a *new* build should go. Absent an override
    that is the user cache directory, never the installed package: writing
    there fails on a read-only or system install, and a partial run would
    corrupt the installation.

    ``$LEVY_DATA_DIR`` overrides both reads and writes. Pointing it at the
    installed package is therefore possible, but that is the caller saying so
    explicitly rather than the default doing it behind their back.
    """
    override = os.environ.get('LEVY_DATA_DIR')
    if override:
        return override
    cache = user_cache_dir()
    if writable:
        return cache
    if all(os.path.exists(os.path.join(cache, '{}.npz'.format(n))) for n in _TABLE_NAMES):
        return cache
    return ROOT


# Cells of cdf.npz that scipy.integrate.quad failed to evaluate when the table
# was generated. All four are at alpha index 4 (alpha = 0.58), beta indices 13
# and 87 (beta = -+0.74), x indices 99 and 100 -- the two grid points closest to
# x = 0, where the oscillatory weight used by _calculate_levy degenerates. They
# hold 5.72e+307 instead of a probability.
#
# This is not a storage error: _calculate_levy still returns 5.72e+307 for those
# arguments today, so regenerating the table with the same code reproduces them.
# Repairing at load time keeps the fix independent of the 12 MB binary; the
# generator needs its own fix before the tables are next rebuilt.
_CDF_TOLERANCE = 1e-6


def _repair_table(key, table):
    """Replace values the table generator failed to compute.

    A CDF cell counts as bad when it is non-finite, or when it lies outside
    [0, 1] by more than `_CDF_TOLERANCE` -- the slack is there so that
    ordinary rounding at the two ends is not mistaken for a failure. Bad
    cells are replaced by linear interpolation along x, which is well
    justified here: the neighbours of the known-bad cells are smooth and
    about 0.0128 apart.
    """
    if key != 'cdf':
        return table

    bad = ~np.isfinite(table) | (table < -_CDF_TOLERANCE) | (table > 1.0 + _CDF_TOLERANCE)
    if not bad.any():
        return table

    table = table.copy()
    x_size = table.shape[0]
    warned = set()
    clipped = 0
    for x_index, alpha_index, beta_index in np.argwhere(bad):
        low = x_index
        while low > 0 and bad[low - 1, alpha_index, beta_index]:
            low -= 1
        high = x_index
        while high < x_size - 1 and bad[high + 1, alpha_index, beta_index]:
            high += 1
        left, right = low - 1, high + 1
        if left < 0 and right > x_size - 1:
            # Every cell in this column is unusable, so there is no good
            # neighbour to interpolate or copy from -- and the copy below
            # would index one past the end. Nothing can be recovered here.
            # Once per column: an unusable column is unusable in every one
            # of its x cells, and warning per cell would emit x_size copies
            # of the same line.
            clipped += 1
            if (alpha_index, beta_index) not in warned:
                warned.add((alpha_index, beta_index))
                logger.warning(
                    'cdf column alpha=%d beta=%d has no usable cell; '
                    'leaving it clipped', alpha_index, beta_index)
            table[x_index, alpha_index, beta_index] = np.clip(
                table[x_index, alpha_index, beta_index], 0.0, 1.0)
            continue
        if left < 0 or right > x_size - 1:
            table[x_index, alpha_index, beta_index] = np.clip(
                table[left if left >= 0 else right, alpha_index, beta_index], 0.0, 1.0)
            continue
        weight = (x_index - left) / float(right - left)
        table[x_index, alpha_index, beta_index] = (
            (1.0 - weight) * table[left, alpha_index, beta_index]
            + weight * table[right, alpha_index, beta_index]
        )

    interpolated = int(bad.sum()) - clipped
    if interpolated:
        logger.warning(
            'Repaired %d unusable cell(s) in the shipped cdf table by '
            'interpolating along x; these are quadrature failures from when '
            'the table was generated. See '
            'https://github.com/josemiotto/pylevy/issues/22',
            interpolated,
        )
    if clipped:
        # Not interpolated: these had no usable neighbour to interpolate
        # from, so the line above must not count them as if they had.
        logger.warning(
            'A further %d cell(s) had no usable neighbour and were only '
            'clipped into [0, 1]; those columns are not trustworthy.',
            clipped,
        )
    return table


def _read_from_cache(key):
    """ Loads the file given by key """
    try:
        return _data_cache[key]
    except KeyError:
        # np.load returns a lazy NpzFile; materialise the array and let the
        # archive close instead of leaking the handle until garbage collection.
        with np.load(os.path.join(data_dir(), '{}.npz'.format(key))) as archive:
            table = archive['arr_0']
        _data_cache[key] = _repair_table(key, table)
        return _data_cache[key]


def _reflect(x, lower, upper):
    """ Folds a value back inside the bounds, reflecting at each edge.

    The in-bounds case returns x unchanged and bit-identical, which is what
    happens on essentially every call: L-BFGS-B already respects the bounds it
    is given. Out of bounds, the fold is computed in closed form rather than by
    repeated reflection, which used to be an unbounded `while 1:` loop -- with
    the sigma bounds (1e-6, 1e10), reflecting 1e30 needs ~1e20 iterations and
    never returns in practice.
    """
    if lower <= x <= upper:
        return x

    span = upper - lower
    if span < 0:
        raise ValueError(
            "reflection bounds are reversed: lower={}, upper={}".format(lower, upper)
        )
    if span == 0:
        return lower

    offset = (x - lower) % (2.0 * span)
    return lower + (2.0 * span - offset if offset > span else offset)


def _interpolate(points, grid, lower, upper):
    """ Perform multi-dimensional Catmull-Rom cubic interpolation. """
    point_shape = np.shape(points)[:-1]
    points = np.reshape(points, (np.multiply.reduce(point_shape), np.shape(points)[-1]))

    grid_shape = np.array(np.shape(grid))
    dims = len(grid_shape)
    points = (points - lower) * ((grid_shape - 1) / (upper - lower))
    floors = np.floor(points).astype('int')

    offsets = points - floors
    offsets2 = offsets * offsets
    offsets3 = offsets2 * offsets
    weighters = [
        -0.5 * offsets3 + offsets2 - 0.5 * offsets,
        1.5 * offsets3 - 2.5 * offsets2 + 1.0,
        -1.5 * offsets3 + 2 * offsets2 + 0.5 * offsets,
        0.5 * offsets3 - 0.5 * offsets2,
    ]

    ravel_grid = np.ravel(grid)

    result = np.zeros(np.shape(points)[:-1], 'float64')
    for i in range(1 << (dims * 2)):
        weights = np.ones(np.shape(points)[:-1], 'float64')
        ravel_offset = 0
        for j in range(dims):
            n = (i >> (j * 2)) % 4
            ravel_offset = ravel_offset * grid_shape[j] + np.maximum(0, np.minimum(grid_shape[j] - 1, floors[:, j] +
                                                                                   (n - 1)))
            weights *= weighters[n][:, j]

        result += weights * np.take(ravel_grid, ravel_offset)

    return np.reshape(result, point_shape)


def _check_alpha_beta(alpha, beta):
    """ Rejects parameters the lookup tables do not cover.

    Without this, `alpha` below 0.5 produced a *negative* grid index, which is
    a perfectly valid Python index: alpha=0.4 gave -4 and silently returned the
    limits for alpha ~ 1.94. The old `except IndexError` guard only fired for
    positive overflow, so under-range values failed silently while over-range
    values raised.
    """
    if not (_lower[1] <= alpha <= _upper[1]):
        raise ValueError(
            'alpha must be in [{}, {}], got {!r}. pylevy interpolates from a '
            'lookup table that does not cover values outside that range.'.format(
                _lower[1], _upper[1], alpha)
        )
    if not (_lower[2] <= beta <= _upper[2]):
        raise ValueError(
            'beta must be in [{}, {}], got {!r}.'.format(_lower[2], _upper[2], beta)
        )


def _grid_shape():
    """ Shape of the tables actually loaded.

    `size` is the resolution of the tables shipped with the package, but
    `levy-tables build --size` can produce others and $LEVY_DATA_DIR can point
    at them. Anything that converts a parameter into a grid index has to use
    the real shape, not the constant: doing otherwise raised
    ``IndexError: index 50 is out of bounds for axis 0 with size 10`` as soon
    as a table of a different resolution was loaded.
    """
    return _read_from_cache('pdf').shape


def _grid_index(value, axis, length=None):
    """ Index of the grid cell used to look up the tail-crossover limits.

    This truncates rather than rounding to nearest, which is almost certainly
    unintentional -- but it is deliberately left alone here, because measuring
    it showed that switching to nearest-neighbour does *not* improve accuracy.

    Over 300 sampled points where the two strategies disagree, compared against
    `_calculate_levy` ground truth:

        strategy      median rel err    mean       p90
        truncate        1.1164e-02    3.5488e-01  1.4560
        round           1.4552e-02    4.3251e-01  1.4656
        bilinear        1.0106e-02    2.9459e-01  0.9723

    Rounding is closer to truth on only 43% of those points and is worse on
    the median. The error is dominated by the discontinuity between the
    interpolated branch and the power-law tail branch -- the two do not meet at
    the crossover, and the CDF steps down by up to 2.57e-03 there -- so the
    choice of limit cell mostly decides which side of a bad seam you land on.
    Fixing that properly means regenerating the limit tables or reconciling the
    branches, not changing this line. Tracked separately.

    Callers must validate first; the clamp only guards against a value landing
    exactly on an endpoint after floating-point rounding.
    """
    if length is None:
        length = _grid_shape()[axis]
    span = _upper[axis] - _lower[axis]
    index = int((value - _lower[axis]) / span * (length - 1))
    return min(length - 1, max(0, index))


def _psi(alpha):
    return np.pi / 2 * (alpha - 1 - np.sign(alpha - 1))


def _phi(alpha, beta):
    """ Common function. """
    return beta * np.tan(np.pi * alpha / 2.0)


convert_to_par0 = {
    '0': lambda x: x,
    '1': lambda x: np.array([
        x[0],
        x[1],
        x[2] + x[3] * _phi(x[0], x[1]),
        x[3]
    ]),
    'M': lambda x: np.array([
        x[0],
        x[1],
        x[2] * x[3],
        x[3] ** (1 / x[0])
    ]),
    'A': lambda x: np.array([
        x[0],
        x[1],
        x[3] * (x[2] + _phi(x[0], x[1])),
        x[3] ** (1 / x[0])
    ]),
    'B': lambda x: np.array([
        x[0],
        np.tan(x[1] * _psi(x[0])) / np.tan(x[0] * np.pi / 2),
        x[3] * (x[2] + np.sin(x[1] * _psi(x[0]))),
        (x[3] * np.cos(x[1] * _psi(x[0]))) ** (1 / x[0])
    ])
}

convert_from_par0 = {
    '0': lambda x: x,
    '1': lambda x: np.array([
        x[0],
        x[1],
        x[2] - x[3] * _phi(x[0], x[1]),
        x[3],
    ]),
    'M': lambda x: np.array([
        x[0],
        x[1],
        x[2] / (x[3] ** x[0]),
        x[3] ** x[0]
    ]),
    'A': lambda x: np.array([
        x[0],
        x[1],
        x[2] / (x[3] ** x[0]) - _phi(x[0], x[1]),
        x[3] ** x[0]
    ]),
    'B': lambda x: np.array([
        x[0],
        np.arctan(_phi(x[0], x[1])) / _psi(x[0]),
        (x[2] / (x[3] ** x[0]) - _phi(x[0], x[1])) * np.cos(np.arctan(_phi(x[0], x[1]))),
        x[3] ** x[0] / np.cos(np.arctan(_phi(x[0], x[1])))
    ])
}


class Parameters(object):
    """
    This class is a wrap for the parameters; it works such that if we fit
    fixing one or more parameters, the optimization only acts on the other
    (the key thing here is the setter).
    The only useful function to be used directly is `convert`, which allows
    to transform parameters from one parametrization to another.
    Available parametrizations are {0, 1, A, B, M}.
    """

    @classmethod
    def convert(cls, pars, par_in, par_out):
        """
        Use to convert a parameter array from one parametrization to another.

        Examples:
            >>> a = np.array([1.6, 0.5, 0.3, 1.2])
            >>> b = Parameters.convert(a, '1', 'B')
            >>> b
            array([1.6       , 0.55457302, 0.2460079 , 1.4243171 ])
            >>> c = Parameters.convert(b, 'B', '1')
            >>> c
            array([1.6, 0.5, 0.3, 1.2])
            >>> np.testing.assert_allclose(a, c)

        :param pars: array of parameters to be converted
        :type pars: :class:`~numpy.ndarray`
        :param par_in: parametrization of the input array
        :type par_in: str
        :param par_out: parametrization of the output array
        :type par_out: str
        :return: array of parameters in the desired parametrization
        :rtype: :class:`~numpy.ndarray`
        """
        res = pars
        if par_out != par_in:
            res = convert_to_par0[par_in](pars)
            if par_out != '0':
                res = convert_from_par0[par_out](res)
        return res

    def __init__(self, par='0', **kwargs):
        self.par = par
        self.pnames = par_names[self.par]
        self._x = np.array([default[par][k] if kwargs[k] is None else kwargs[k] for k in self.pnames])
        self.variables = [i for i, k in enumerate(self.pnames) if kwargs[k] is None]
        self.fixed = [i for i, k in enumerate(self.pnames) if kwargs[k] is not None]
        self.fixed_values = [kwargs[k] for i, k in enumerate(self.pnames) if kwargs[k] is not None]

    def get(self, par_out=None):
        """
        Same as `convert` but using from within the Parameter object.

        Examples:
            >>> p = Parameters(par='1', alpha=1.5, beta=0.5, mu=0, sigma=1.2)  # to convert
            >>> p.get('B')  # returns the parameters in the parametrization B
            array([1.5       , 0.59033447, 0.03896531, 1.46969385])

        """
        if par_out is None:
            par_out = self.par
        return Parameters.convert(self._x, self.par, par_out)

    def __str__(self):
        txt = ', '.join(['{{0[{0}]}}: {{1[{1}]:.2f}}'.format(i, i) for i in range(4)])
        txt += '. Parametrization: {2}.'
        return txt.format(self.pnames, self.get(), self.par)

    def __repr__(self):
        txt = 'par={2}, ' + ', '.join(['{{0[{0}]}}={{1[{1}]:.2f}}'.format(i, i) for i in range(4)])
        return txt.format(self.pnames, self.get(), self.par)

    @property
    def x(self):
        return self._x[self.variables]

    @x.setter
    def x(self, values):
        # Dispatch on the type rather than on its name, and reject anything
        # else explicitly: without the else branch, `vals` stayed unbound and
        # assigning e.g. a list raised UnboundLocalError instead of TypeError.
        if isinstance(values, optimize.OptimizeResult):
            vals = values.x
        elif isinstance(values, np.ndarray):
            vals = values
        elif isinstance(values, (list, tuple)):
            vals = np.asarray(values, dtype='d')
        else:
            raise TypeError(
                'expected an OptimizeResult, ndarray, list or tuple, '
                'got {}'.format(type(values).__name__)
            )
        # One value per free parameter. Without this the loop below indexes off
        # the end and reports a bare IndexError naming neither the setter nor
        # the length it wanted.
        if len(vals) != len(self.variables):
            raise ValueError(
                'expected {} value(s) for the free parameters {}, got {}'.format(
                    len(self.variables),
                    [self.pnames[i] for i in self.variables],
                    len(vals)
                )
            )
        for j, i in enumerate(self.variables):
            self._x[i] = f_bounds[self.par][self.pnames[i]](vals[j])


def _approximate(x, alpha, beta, cdf=False):
    mask = (x > 0)
    values = np.sin(np.pi * alpha / 2.0) * gamma(alpha) / np.pi * np.power(np.abs(x), -alpha - 1.0)
    values[mask] *= (1.0 + beta)
    values[~mask] *= (1.0 - beta)
    if cdf:
        values[mask] = 1.0 - values[mask] * x[mask]
        values[~mask] = values[~mask] * (-x[~mask])
        return values
    else:
        return values * alpha


def levy(x, alpha, beta, mu=0.0, sigma=1.0, cdf=False):
    """
    Levy distribution with the tail replaced by the analytical (power law) approximation.

    `alpha` in [0.5, 2] is the index of stability, or characteristic exponent.
    Values outside that range raise ValueError: the lookup table this
    interpolates from does not cover them.
    `beta` in [-1, 1] is the skewness. `mu` in the reals and `sigma` > 0 are the
    location and scale of the distribution (corresponding to `delta` and `gamma`
    in Nolan's notation; note that sigma in levy corresponds to sqrt(2) sigma
    in the Normal distribution).
    *cdf* is a Boolean that specifies if it returns the cdf instead of the pdf.

    It uses parametrization 0 (to get it from another parametrization, convert).

    Example:
        >>> x = np.array([1, 2, 3])
        >>> levy(x, 1.5, 0, cdf=True)
        array([0.75634202, 0.89496045, 0.94840227])

    :param x: values where the function is evaluated
    :type x: :class:`~numpy.ndarray`
    :param alpha: alpha
    :type alpha: float
    :param beta: beta
    :type beta: float
    :param mu: mu
    :type mu: float
    :param sigma: sigma
    :type sigma: float
    :param cdf: it specifies if you want the cdf instead of the pdf
    :type cdf: bool
    :return: values of the pdf (or cdf if parameter 'cdf' is set to True) at 'x'
    :rtype: :class:`~numpy.ndarray`
    """

    _check_alpha_beta(alpha, beta)

    loc = mu

    what = _read_from_cache('cdf') if cdf else _read_from_cache('pdf')
    lower_limit = _read_from_cache('lower_limit')
    upper_limit = _read_from_cache('upper_limit')

    xr = (np.asarray(x, 'd') - loc) / sigma
    alpha_index = _grid_index(alpha, 1, lower_limit.shape[0])
    beta_index = _grid_index(beta, 2, lower_limit.shape[1])
    low_lims = lower_limit[alpha_index, beta_index]
    up_lims = upper_limit[alpha_index, beta_index]
    mask = (low_lims <= xr) & (xr <= up_lims)
    z = xr[mask]

    points = np.empty(np.shape(z) + (3,), 'float64')
    points[..., 0] = np.arctan(z)
    points[..., 1] = alpha
    points[..., 2] = beta

    interpolated = _interpolate(points, what, _lower, _upper)
    approximated = _approximate(xr[~mask], alpha, beta, cdf)

    res = np.empty(np.shape(xr), 'float64')
    res[mask] = interpolated
    res[~mask] = approximated
    if cdf is False:
        res /= sigma
    return float(res) if np.isscalar(x) else res


def neglog_levy(x, alpha, beta, mu, sigma):
    """
    Interpolate negative log densities of the Levy stable distribution
    specified by `alpha` and `beta`. Small/negative densities are capped
    at 1e-100 to preserve sanity.

    It uses parametrization 0 (to get it from another parametrization, convert).

    Example:
        >>> x = np.array([1,2,3])
        >>> neglog_levy(x, 1.5, 0.0, 0.0, 1.0)
        array([1.59929892, 2.47054131, 3.45747366])

    :param x: values where the function is evaluated
    :type x: :class:`~numpy.ndarray`
    :param alpha: alpha
    :type alpha: float
    :param beta: beta
    :type beta: float
    :param mu: mu
    :type mu: float
    :param sigma: sigma
    :type sigma: float
    :return: values of -log(pdf(x))
    :rtype: :class:`~numpy.ndarray`
    """

    return -np.log(np.maximum(1e-100, levy(x, alpha, beta, mu, sigma)))


def fit_levy(x, par='0', **kwargs):
    """
    Estimate parameters of Levy stable distribution given data x, using
    Maximum Likelihood estimation.

    By default, searches all possible Levy stable distributions. However
    you may restrict the search by specifying the values of one or more
    parameters. Notice that the parameters to be fixed can be chosen in
    all the available parametrizations {0, 1, A, B, M}.

    Examples:
        >>> np.random.seed(0)
        >>> x = random(1.5, 0, 0, 1, shape=(200,))
        >>> fit_levy(x) # -- Fit a stable distribution to x
        (par=0, alpha=1.52, beta=-0.08, mu=0.05, sigma=0.99, 402.37150603509247)

        >>> fit_levy(x, beta=0.0) # -- Fit a symmetric stable distribution to x
        (par=0, alpha=1.53, beta=0.00, mu=0.03, sigma=0.99, 402.43833088693725)

        >>> fit_levy(x, beta=0.0, mu=0.0) # -- Fit a symmetric distribution centered on zero to x
        (par=0, alpha=1.53, beta=0.00, mu=0.00, sigma=0.99, 402.4736618823546)

        >>> fit_levy(x, alpha=1.0, beta=0.0) # -- Fit a Cauchy distribution to x
        (par=0, alpha=1.00, beta=0.00, mu=0.10, sigma=0.90, 416.54249079255976)

    :param x: values to be fitted
    :type x: :class:`~numpy.ndarray`
    :param par: parametrization
    :type par: str
    :return: a tuple with a `Parameters` object and the negative log likelihood of the data.
    :rtype: tuple
    """

    values = {par_name: kwargs.get(par_name) for par_name in par_names[par]}

    parameters = Parameters(par=par, **values)
    temp = Parameters(par=par, **values)

    def neglog_density(param):
        temp.x = param
        alpha, beta, mu, sigma = temp.get('0')
        return np.sum(neglog_levy(x, alpha, beta, mu, sigma))

    bounds = tuple(par_bounds[i] for i in parameters.variables)
    res = optimize.minimize(neglog_density, parameters.x, method='L-BFGS-B', bounds=bounds)
    parameters.x = res.x

    return parameters, neglog_density(parameters.x)


def random(alpha, beta, mu=0.0, sigma=1.0, shape=()):
    """
    Generate random values sampled from an alpha-stable distribution.
    Notice that this method is "exact", in the sense that is derived
    directly from the definition of stable variable.
    It uses parametrization 0 (to get it from another parametrization, convert).

    Example:
        >>> rnd = random(1.5, 0, shape=(100,))  # parametrization 0 is implicit
        >>> par = np.array([1.5, 0.905, 0.707, 1.414])
        >>> rnd = random(*Parameters.convert(par ,'B' ,'0'), shape=(100,))  # example with convert

    :param alpha: alpha
    :type alpha: float
    :param beta: beta
    :type beta: float
    :param mu: mu
    :type mu: float
    :param sigma: sigma
    :type sigma: float
    :param shape: shape (numpy array type) of the resulting array
    :type shape: tuple
    :return: generated random values
    :rtype: :class:`~numpy.ndarray`
    """

    if alpha == 2:
        # mu and sigma have to be applied here too. This branch used to return
        # before reaching the `return mu + sigma * k` at the end of the
        # function, so random(2.0, 0.0, mu=100, sigma=5) came back centred on
        # zero with unit-ish scale.
        return mu + sigma * np.random.standard_normal(shape) * np.sqrt(2.0)

    # The sampler below divides by (1 - alpha) implicitly, through
    # phi = beta * tan(pi * alpha / 2), so alpha exactly 1 has to be nudged off
    # the pole of the tangent.
    #
    # The nudge used to be 1e-15, which is far too close: tan(pi*alpha/2) is
    # then evaluated 1e-15 from its pole, where the unavoidable ~1e-16 rounding
    # of the argument becomes a ~11% relative error in the result. That is not
    # cosmetic. At beta = +-1 it made the base of the fractional power below go
    # negative for about 0.9% of draws, producing NaN, and the samples that did
    # survive were measurably from the wrong distribution (Kolmogorov-Smirnov
    # against this package's own CDF: p = 3e-07 over 200k draws).
    #
    # 1e-8 keeps the same limiting value -- beta*tan(pi*alpha/2)*sin((1-alpha)*b*pi)
    # tends to 2*beta*b whatever the radius -- while evaluating it accurately.
    # It sits in the middle of a wide plateau: every radius from 1e-10 to 1e-6
    # gives NaN-free samples and KS p ~ 0.50, so this is not a tuned constant.
    # The distributional cost of shifting alpha by 1e-8 is far below sampling
    # noise.
    # copysign, not a bare +: nudging an alpha just *below* 1 up to
    # 1 + radius would hand the sampler the opposite side of the pole from
    # the one the caller asked for, and flip the sign of (1 - alpha). The
    # widened radius makes that band 1e-8 wide instead of 1e-15, so the
    # side is now worth preserving. alpha exactly 1.0 still goes up, since
    # 1.0 - 1.0 is +0.0 and copysign follows it.
    if np.absolute(alpha - 1.0) < _ALPHA_1_RADIUS:
        alpha = 1.0 + np.copysign(_ALPHA_1_RADIUS, alpha - 1.0)

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return mu + sigma * k


# Backwards compatibility: the table-generation helpers moved to levy._build.
# Exposed lazily (PEP 562) so importing levy does not pull in scipy.integrate
# for the sake of code only a maintainer regenerating tables ever runs.
_MOVED_TO_BUILD = {
    '_calculate_levy': 'calculate_levy',
    '_int_levy': 'interpolated_levy',
}


def __getattr__(name):
    if name in _MOVED_TO_BUILD:
        from levy import _build
        return getattr(_build, _MOVED_TO_BUILD[name])
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))


if __name__ == "__main__":
    from levy._build.cli import main
    logger.warning(
        "`python -m levy build` is superseded by the `levy-tables` command, "
        "which writes to a cache directory instead of into the installed package."
    )
    # Pass the subcommand through. Prepending 'build' unconditionally turned
    # `python levy/__init__.py where` into `build where`; no arguments still
    # means build, which is what this entry point has always done.
    sys.exit(main(sys.argv[1:] or ['build']))
