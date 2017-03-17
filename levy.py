#    Copyright (C) 2005 Paul Harrison
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later versionp.
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
interpolation scheme allows fast fitting of Levy stable distributions 
to data using the Maximum Likelihood technique.

Does not support alpha values less than 0.5.
"""

import sys
import numpy as np
import scipy.special as sp
from builtins import range

__version__ = "0.6"

# Dimensions: 0 - x, 1 - alpha, 2 - beta
_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])
par_bounds = ((0.5, 2.0), (-1.0, 1.0), (None, None), (None, None))

par_names = ['alpha', 'beta', 'mu', 'sigma']
default = [1.5, 0.0, 0.0, 1.0]
default = {par_names[i]: default[i] for i in range(4)}
f_bounds = {
    'alpha': lambda x: _reflect(x, _lower[1], _upper[1]),
    'beta': lambda x: _reflect(x, _lower[2], _upper[2]),
    'mu': lambda x: x,
    'sigma': lambda x: x
}


class Parameters(object):
    def __init__(self, **kwargs):
        # self._f = [f_bounds[k] for k in par_names]
        self._x = np.array([default[k] if kwargs[k] is None else kwargs[k] for k in par_names])
        self.variables = [i for i, k in enumerate(par_names) if kwargs[k] is None]
        self.fixed = [i for i, k in enumerate(par_names) if kwargs[k] is not None]
        self.fixed_values = [kwargs[k] for i, k in enumerate(par_names) if kwargs[k] is not None]

    def get_all(self):
        return self._x

    def __str__(self):
        return self.x.__str__()

    @property
    def x(self):
        return self._x[self.variables]

    @x.setter
    def x(self, value):
        for j, i in enumerate(self.variables):
            f = f_bounds[par_names[i]]
            val = value[j]
            h = self._x[i]
            self._x[i] = f(val)


def phi(alpha, beta):
    return beta * np.tan(np.pi * alpha / 2.0)


def _calculate_levy(x, alpha, beta, cdf=False):
    """ Calculation of Levy stable distribution via numerical integration.
        This is used in the creation of the lookup table. """
    # "0" parameterization as per http://academic2.americanp.edu/~jpnolan/stable/stable.html
    # Note: fails for alpha=1.0
    #       (so make sure alpha=1.0 isn't exactly on the interpolation grid)
    from scipy import integrate

    C = beta * np.tan(np.pi * 0.5 * alpha)

    def func_cos(u):
        ua = u ** alpha
        if ua > 700.0: return 0.0
        return np.exp(-ua) * np.cos(C * ua - C * u)

    def func_sin(u):
        ua = u ** alpha
        if ua > 700.0: return 0.0
        return np.exp(-ua) * np.sin(C * ua - C * u)

    if cdf:
        # Cumulative density function
        return (integrate.quad(lambda u: u and func_cos(u) / u or 0.0, 0.0, np.Inf, weight="sin", wvar=x,
                               limlst=1000)[0]
                + integrate.quad(lambda u: u and func_sin(u) / u or 0.0, 0.0, np.Inf, weight="cos", wvar=x,
                                 limlst=1000)[0]
                ) / np.pi + 0.5
    else:
        # Probability density function
        return (integrate.quad(func_cos, 0.0, np.Inf, weight="cos", wvar=x, limlst=1000)[0]
                - integrate.quad(func_sin, 0.0, np.Inf, weight="sin", wvar=x, limlst=1000)[0]
                ) / np.pi


def _levy_tan(x, alpha, beta, cdf=False):
    """ Calculate the values stored in the lookup table. 
        The tan mapping allows the table to cover the range from -INF to INF. """
    x = np.tan(x)
    return _calculate_levy(x, alpha, beta, cdf)


def _interpolate(points, grid, lower, upper):
    """ Perform multi-dimensional Catmull-Rom cubic interpolationp. """
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
            ravel_offset = ravel_offset * grid_shape[j] + \
                           np.maximum(0, np.minimum(grid_shape[j] - 1, floors[:, j] + (n - 1)))
            weights *= weighters[n][:, j]

        result += weights * np.take(ravel_grid, ravel_offset)

    return np.reshape(result, point_shape)


def _approximate_pdf(x, alpha, beta):
    return (1.0 + np.abs(beta)) * np.sin(np.pi * alpha / 2.0) * \
           sp.gamma(alpha) / np.pi * np.power(x, -alpha - 1.0) * alpha


def _approximate_cdf(x, alpha, beta):
    return 1.0 - (1.0 + np.abs(beta)) * np.sin(np.pi * alpha / 2.0) * \
                 sp.gamma(alpha) / np.pi * np.power(x, -alpha)


def _make_data_file():
    """ Generates the lookup table, writes it to a .py file. """
    import base64

    size = (200, 50, 51)
    pdf = np.zeros(size, 'float64')
    cdf = np.zeros(size, 'float64')
    xs, alphas, betas = [np.linspace(_lower[i], _upper[i], size[i], endpoint=True) for i in range(len(size))]

    print("Generating levy_data.py ...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print("Calculating alpha={}, beta={}".format(alpha, beta))
            for k, x in enumerate(xs):
                pdf[k, i, j] = _levy_tan(x, alpha, beta)
                cdf[k, i, j] = _levy_tan(x, alpha, beta, True)


    np.savez('pdf.npz', pdf)
    np.savez('cdf.npz', cdf)


def _int_levy(x, alpha, beta, cdf=False):
    """ Interpolate densities of the Levy stable distribution specified by alpha and beta.

        Specify cdf=True to obtain the *cumulative* density functionp.

        Note: may sometimes return slightly negative values, due to numerical inaccuracies.
    """

    points = np.empty(np.shape(x) + (3,), 'float64')
    points[..., 0] = np.arctan(x)
    points[..., 1] = alpha
    points[..., 2] = beta

    if cdf:
        what = np.load('cdf.npz')
    else:
        what = np.load('pdf.npz')
    return _interpolate(points, what, _lower, _upper)


def _get_closest_approx(alpha, beta):
    x0, x1, n = -50.0, 10000.0 - 50.0, 100000
    dx = (x1 - x0) / n
    x = np.linspace(x0, x1, num=n, endpoint=True)
    y = 1.0 - _int_levy(x, alpha, -beta, cdf=True)
    z = 1.0 - _approximate_cdf(x, alpha, -beta)
    mask = (10.0 < x) & (x < 500.0)
    return 10.0 + dx * np.argmin((np.log(z[mask]) - np.log(y[mask])) ** 2.0)


def _make_approx_data_file():
    size = (50, 51)
    limits = np.zeros(size, 'float64')
    alphas, betas = [
        np.linspace(_lower[1], _upper[1], size[0], endpoint=True),
        np.linspace(0, _upper[2], size[1], endpoint=True)]

    print("Generating levy_approx_data.py ...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            limits[i, j] = _get_closest_approx(alpha, beta)
            print("Calculating alpha={}, beta={}, limit={}".format(alpha, beta, limits[i, j]))

    np.savez('limits.npz', limits)


def levy(x, alpha, beta, mu=0.0, sigma=1.0, cdf=False, par=0):
    """
    Levy with the tail replaced by the analytical approximation.
    Also, mu, sigma are parameters that shift and rescale the distribution.
    Parametrization can be chosen according to Nolan, par={0,1}.
    """

    if par == 0:
        loc = mu
    elif par == 1:
        loc = mu + beta * sigma * np.tan(np.pi * alpha / 2.0)  # Par 1 is changed

    if cdf:
        what = np.load('cdf.npz')
        app = _approximate_cdf
    else:
        what = np.load('pdf.npz')
        app = _approximate_pdf
    limits = np.load('limits.npz')

    xr = (x - loc) / sigma
    beta = -beta
    alpha_index = int((alpha - 0.5) * 49.0 / 1.5)
    beta_index = int(np.abs(beta) * 50.0)
    l = limits[alpha_index, beta_index]
    if beta <= 0.0:
        mask = (xr < l)
    elif beta > 0.0:
        mask = (xr > -l)
    z = xr[mask]

    points = np.empty(np.shape(z) + (3,), 'float64')
    points[..., 0] = np.arctan(z)
    points[..., 1] = alpha
    points[..., 2] = beta

    interpolated = _interpolate(points, what, _lower, _upper)
    approximated = app(xr[~mask], alpha, beta)

    res = np.empty(np.shape(xr), 'float64')
    res[mask] = interpolated
    res[~mask] = approximated
    if cdf is False:
        res /= sigma
    return res


def neglog_levy(x, alpha, beta, mu, sigma, par=0):
    """
    Interpolate negative log densities of the Levy stable distribution specified by alpha and beta.
    Small/negative densities are capped at 1e-100 to preserve sanity.
    """
    return -np.log(np.maximum(1e-100, levy(x, alpha, beta, mu, sigma, par=par)))


def _reflect(x, lower, upper):
    while 1:
        if x < lower:
            x = lower - (x - lower)
        elif x > upper:
            x = upper - (x - upper)
        else:
            return x


def fit_levy(x, alpha=None, beta=None, mu=None, sigma=None, par=0):
    """
    Estimate parameters of Levy stable distribution given data x, using the Maximum Likelihood method.

    By default, searches all possible Levy stable distributions.
    However you may restrict the search by specifying the values of one or more parameters.
        
    Examples:
        
        levy(x) -- Fit a stable distribution to x

        levy(x, beta=0.0) -- Fit a symmetric stable distribution to x

        levy(x, beta=0.0, mu=0.0) -- Fit a symmetric distribution centered on zero to x

        levy(x, alpha=1.0, beta=0.0) -- Fit a Cauchy distribution to x

    Returns a tuple of (alpha, beta, mu, sigma, negative log density)
    """

    # The parametrization is changed to par=0, if is par=1. At the end, the parametrization is reverted.
    if mu is not None:
        if par == 0:
            mu0 = mu
        elif par == 1:
            mu0 = mu + beta * sigma * np.tan(np.pi * alpha / 2.0)  # Par 1 is changed
    elif mu is None:
        mu0 = mu

    from scipy import optimize

    kwargs = {'alpha': alpha, 'beta': beta, 'mu': mu0, 'sigma': sigma}
    parameters = Parameters(**kwargs)

    def neglog_density(param):
        p = np.zeros(4)
        p[parameters.variables] = param
        p[parameters.fixed] = parameters.fixed_values
        alpha, beta, mu, sigma = p
        return np.sum(neglog_levy(x, alpha, beta, mu, sigma))

    parameters.x = optimize.minimize(neglog_density, parameters.x, method='L-BFGS-B', bounds=par_bounds, disp=0)
    alpha, beta, mu, sigma = parameters.get_all()
    return alpha, beta, mu, sigma, neglog_density(parameters.x)

    if par == 0:
        mu = mu0
    elif par == 1:
        mu = mu0 - beta * sigma * np.tan(np.pi * alpha / 2.0)

    return alpha, beta, mu, sigma, neglog_final


def random(alpha, beta, mu=0.0, sigma=1.0, shape=(), par=0):
    """
    Generate random values sampled from an alpha-stable distribution.
    """

    if par == 0:
        mu0 = mu
    elif par == 1:
        mu0 = mu + beta * sigma * np.tan(np.pi * alpha / 2.0)  # Par 1 is changed

    if alpha == 2:
        return np.random.standard_normal(shape) * np.sqrt(2.0)

    # Fails for alpha exactly equal to 1.0
    # but works fine for alpha infinitesimally greater or less than 1.0    
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
    e = phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return mu0 + sigma * k


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        _make_data_file()
        _make_approx_data_file()

    print("Testing fit_levy.")

    print("1000 points, result should be (1.5, 0.5, 0.0, 1.0).")
    print(fit_levy(random(1.5, 0.5, 0.0, 1.0, 1000)))
