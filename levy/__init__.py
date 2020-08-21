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

import sys
import os
import numpy as np
from scipy.special import gamma
from scipy import optimize

__version__ = "1.1"

# Some constants of the program.
# Dimensions: 0 - x, 1 - alpha, 2 - beta
size = (200, 76, 101)  # size of the grid (xs, alpha, beta)
_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])  # lower limit of parameters
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])  # upper limit of parameters

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


def _read_from_cache(key):
    """ Loads the file given by key """
    try:
        return _data_cache[key]
    except KeyError:
        _data_cache[key] = np.load(os.path.join(ROOT, '{}.npz'.format(key)))['arr_0']
        return _data_cache[key]


def _reflect(x, lower, upper):
    """ Makes the parameters to be inside the bounds """
    while 1:
        if x < lower:
            x = lower - (x - lower)
        elif x > upper:
            x = upper - (x - upper)
        else:
            return x


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
        if values.__class__.__name__ == 'OptimizeResult':
            vals = values.x
        elif values.__class__.__name__ == 'ndarray':
            vals = values
        for j, i in enumerate(self.variables):
            self._x[i] = f_bounds[self.par][self.pnames[i]](vals[j])


def _calculate_levy(x, alpha, beta, cdf=False):
    """
    Calculation of Levy stable distribution via numerical integration.
    This is used in the creation of the lookup table.
    Notice that to compute it in a 'true' x, the tangent must be applied.
    Example: levy(2, 1.5, 0) = _calculate_levy(np.tan(2), 1.5, 0)
    "0" parametrization as per http://academic2.americanp.edu/~jpnolan/stable/stable.html
    Addition: the special case alpha=1.0 was added. Due to an error in the
    numerical integration, the limit was changed from 0 to 1e-10.
    """
    from scipy import integrate

    beta = -beta

    if alpha == 1:
        li = 1e-10

        # These functions need a correction, since the distribution is displaced, probably get rid of "-u" at the end
        def func_cos(u):
            # return np.exp(-u) * np.cos(-beta * 2 / np.pi * (u * np.log(u) - u))
            return np.exp(-u) * np.cos(-beta * 2 / np.pi * u * np.log(u))

        def func_sin(u):
            # return np.exp(-u) * np.sin(-beta * 2 / np.pi * (u * np.log(u) - u))
            return np.exp(-u) * np.sin(-beta * 2 / np.pi * u * np.log(u))

    else:
        li = 0

        def func_cos(u):
            ua = u ** alpha
            return np.exp(-ua) * np.cos(_phi(alpha, beta) * (ua - u))

        def func_sin(u):
            ua = u ** alpha
            return np.exp(-ua) * np.sin(_phi(alpha, beta) * (ua - u))

    if cdf:
        # Cumulative density function
        return (
            integrate.quad(lambda u: u and func_cos(u) / u or 0.0, li, np.Inf, weight="sin", wvar=x, limlst=1000)[0]
            + integrate.quad(lambda u: u and func_sin(u) / u or 0.0, li, np.Inf, weight="cos", wvar=x, limlst=1000)[0]
            ) / np.pi + 0.5
    else:
        # Probability density function
        return (
            integrate.quad(func_cos, li, np.Inf, weight="cos", wvar=x, limlst=1000)[0]
            - integrate.quad(func_sin, li, np.Inf, weight="sin", wvar=x, limlst=1000)[0]
            ) / np.pi


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


def _make_dist_data_file():
    """ Generates the lookup tables, writes it to .npz files. """

    xs, alphas, betas = [np.linspace(_lower[i], _upper[i], size[i], endpoint=True) for i in [0, 1, 2]]
    ts = np.tan(xs)

    print("Generating pdf.npz ...")
    pdf = np.zeros(size, 'float64')
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print("Calculating alpha={:.2f}, beta={:.2f}".format(alpha, beta))
            pdf[:, i, j] = [_calculate_levy(t, alpha, beta, False) for t in ts]
    np.savez(os.path.join(ROOT, 'pdf.npz'), pdf)

    print("Generating cdf.npz ...")
    cdf = np.zeros(size, 'float64')
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print("Calculating alpha={:.2f}, beta={:.2f}".format(alpha, beta))
            cdf[:, i, j] = [_calculate_levy(t, alpha, beta, True) for t in ts]
    np.savez(os.path.join(ROOT, 'cdf.npz'), cdf)


def _int_levy(x, alpha, beta, cdf=False):
    """
    Interpolate densities of the Levy stable distribution specified by alpha and beta.

    Specify cdf=True to obtain the *cumulative* density function.

    Note: may sometimes return slightly negative values, due to numerical inaccuracies.
    """
    points = np.empty(np.shape(x) + (3,), 'float64')
    points[..., 0] = np.arctan(x)
    points[..., 1] = alpha
    points[..., 2] = beta

    what = _read_from_cache('cdf') if cdf else _read_from_cache('pdf')
    return _interpolate(points, what, _lower, _upper)


def _get_closest_approx(alpha, beta, upper=True):
    n = 100000
    x1, x2 = -50.0, 1e4 - 50.0
    li1, li2 = 10, 500
    if upper is False:
        x1, x2 = -1e4 + 50, 50
        li1, li2 = -500, -10
    dx = (x2 - x1) / n
    x = np.linspace(x1, x2, num=n + 1, endpoint=True)
    y = 1.0 - _int_levy(x, alpha, beta, cdf=True)
    z = 1.0 - _approximate(x, alpha, beta, cdf=True)
    mask = (li1 < x) & (x < li2)
    return li1 + dx * np.argmin((np.log(z[mask]) - np.log(y[mask])) ** 2.0)


def _make_limit_data_files():
    for upper in [True, False]:
        string = 'lower' if upper is False else 'upper'

        limits = np.zeros(size[1:], 'float64')
        alphas, betas = [np.linspace(_lower[i], _upper[i], size[i], endpoint=True) for i in [1, 2]]

        print("Generating {}_limit.npz ...".format(string))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                limits[i, j] = _get_closest_approx(alpha, beta, upper=upper)
                print("Calculating alpha={:.2f}, beta={:.2f}, limit={:.2f}".format(alpha, beta, limits[i, j]))

        np.savez(os.path.join(ROOT, '{}_limit.npz'.format(string)), limits)


def levy(x, alpha, beta, mu=0.0, sigma=1.0, cdf=False):
    """
    Levy distribution with the tail replaced by the analytical (power law) approximation.

    `alpha` in (0, 2] is the index of stability, or characteristic exponent.
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

    loc = mu

    what = _read_from_cache('cdf') if cdf else _read_from_cache('pdf')
    # limits = _limits()
    lower_limit = _read_from_cache('lower_limit')
    upper_limit = _read_from_cache('upper_limit')

    xr = (np.asarray(x, 'd') - loc) / sigma
    alpha_index = int((alpha - _lower[1]) / (_upper[1] - _lower[1]) * (size[1] - 1))
    beta_index = int((beta - _lower[2]) / (_upper[2] - _lower[2]) * (size[2] - 1))
    try:
        # lims = limits[alpha_index, beta_index]
        low_lims = lower_limit[alpha_index, beta_index]
        up_lims = upper_limit[alpha_index, beta_index]
    except IndexError:
        print(alpha, alpha_index)
        print(beta, beta_index)
        print('This should not happen! If so, please open an issue in the pylevy github page please.')
        raise
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

    values = {par_name: None if par_name not in kwargs else kwargs[par_name] for i, par_name in
              enumerate(par_names[par])}

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

    # loc = change_par(alpha, beta, mu, sigma, par, 0)
    if alpha == 2:
        return np.random.standard_normal(shape) * np.sqrt(2.0)

    # Fails for alpha exactly equal to 1.0
    # but works fine for alpha infinitesimally greater or lower than 1.0
    radius = 1e-15  # <<< this number is *very* small
    if np.absolute(alpha - 1.0) < radius:
        # So doing this will make almost exactly no difference at all
        alpha = 1.0 + radius

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


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        _make_dist_data_file()
        _make_limit_data_files()

    print("Testing fit_levy using parametrization 0 and fixed alpha (1.5).")

    N = 1000
    print("{} points, result should be (1.5, 0.5, 0.0, 1.0).".format(N))
    x0 = random(1.5, 0.5, 0.0, 1.0, shape=(1000))

    result0 = fit_levy(x0, par='0', alpha=1.5)
    print(result0)
