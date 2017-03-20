# -*- encoding: utf-8 -*-
#    Copyright (C) 2017 José M. Miotto
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
interpolation scheme allows fast fitting of data by Maximum Likelihood .

Does not support alpha values less than 0.5.
"""

import sys
import numpy as np
import scipy.special as sp
from builtins import range

__version__ = "0.6"

# Some constants of the program.
# Dimensions: 0 - x, 1 - alpha, 2 - beta
size = (200, 76, 101)  # size of the grid (xs, alpha, beta)
_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])  # lower limit of parameters
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])  # upper limit of parameters

par_bounds = ((_lower[1], _upper[1]), (_lower[2], _upper[2]), (None, None), (0.0, None))  # parameter bounds for fit.
par_names = ['alpha', 'beta', 'mu', 'sigma']  # names of the parameters
default = [1.5, 0.0, 0.0, 1.0]  # default values of the parameters for fit.
default = {par_names[i]: default[i] for i in range(4)}
""" f_bounds function only useful if minimizing with fmin """
f_bounds = {
    'alpha': lambda x: _reflect(x, *par_bounds[0]),
    'beta': lambda x: _reflect(x, *par_bounds[1]),
    'mu': lambda x: x,
    'sigma': lambda x: x
}


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
            ravel_offset = ravel_offset * grid_shape[j] + \
                           np.maximum(0, np.minimum(grid_shape[j] - 1, floors[:, j] + (n - 1)))
            weights *= weighters[n][:, j]

        result += weights * np.take(ravel_grid, ravel_offset)

    return np.reshape(result, point_shape)


class Parameters(object):
    """ This class is a wrap for the parameters;
    it works such that if we fit fixing one or more parameters, the optimization only acts on the other.
    The key thing here is the setter.
    """

    def __init__(self, **kwargs):
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
            # If the fmin function is used to optimize, use this line:
            # self._x[i] = f_bounds[par_names[i]](value[j])
            # If the minimize function is used to optimize, use this line:
            self._x[i] = f_bounds[par_names[i]](value.x[j])


def _phi(alpha, beta):
    """ Common function. """
    return beta * np.tan(np.pi * alpha / 2.0)


def _calculate_levy(x, alpha, beta, cdf=False):
    """ Calculation of Levy stable distribution via numerical integration.
        This is used in the creation of the lookup table.
        Notice that to compute it in a 'true' x, the tangent must be applied.
        Example: levy(2, 1.5, 0) = _calculate_levy(np.tan(2), 1.5, 0)
        "0" parameterization as per http://academic2.americanp.edu/~jpnolan/stable/stable.html
        Note: fails for alpha=1.0 (so make sure alpha=1.0 isn't exactly on the interpolation grid)
    """
    from scipy import integrate

    beta = -beta
    C = _phi(alpha, beta)

    def func_cos(u):
        ua = u ** alpha
        # if ua > 700.0: return 0.0
        return np.exp(-ua) * np.cos(C * ua - C * u)

    def func_sin(u):
        ua = u ** alpha
        # if ua > 700.0: return 0.0
        return np.exp(-ua) * np.sin(C * ua - C * u)

    if cdf:
        # Cumulative density function
        return (integrate.quad(
            lambda u: u and func_cos(u) / u or 0.0, 0.0, np.Inf, weight="sin", wvar=x, limlst=1000)[0]
                + integrate.quad(
            lambda u: u and func_sin(u) / u or 0.0, 0.0, np.Inf, weight="cos", wvar=x, limlst=1000)[0]
                ) / np.pi + 0.5
    else:
        # Probability density function
        return (integrate.quad(
            func_cos, 0.0, np.Inf, weight="cos", wvar=x, limlst=1000)[0]
                - integrate.quad(
            func_sin, 0.0, np.Inf, weight="sin", wvar=x, limlst=1000)[0]
                ) / np.pi


def _approximate(x, alpha, beta, cdf=False):
    mask = (x > 0)
    values = np.sin(np.pi * alpha / 2.0) * sp.gamma(alpha) / np.pi * np.power(np.abs(x), -alpha - 1.0)
    values[mask] *= (1.0 + beta)
    values[~mask] *= (1.0 - beta)
    if cdf:
        return 1.0 - values
    else:
        return values * alpha


def _make_dist_data_file():
    """ Generates the lookup tables, writes it to .npz files. """

    xs, alphas, betas = [np.linspace(_lower[i], _upper[i], size[i], endpoint=True) for i in [0, 1, 2]]
    ts = np.tan(xs)
    from sys import getsizeof
    print("Generating levy_data.py ...")
    pdf = np.zeros(size, 'float64')
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print("Calculating alpha={:.2f}, beta={:.2f}".format(alpha, beta))
            # if beta > 0.0:
            #     f.write('s\n')
            # else:
            #     l = [_calculate_levy(t, alpha, beta, False) for t in ts]
            #     print(getsizeof(l))
            #     f = open('temp_pdf', 'a')
            #     f.write('\t'.join(map(str, l))+'\n')
            #     f.close()
            #     del l
            pdf[:, i, j] = [_calculate_levy(t, alpha, beta, False) for t in ts]
            # for k, t in enumerate(ts):
            #     pdf[k, i, j] = _calculate_levy(t, alpha, beta, False)

    # pdf = np.zeros(size, 'float64')
    # i = 0
    # j = 0
    # with open('temp_pdf', 'r') as f:
    #     for line in f:
    #         if line.rstrip('\n') == 's':
    #             l = np.flip(pdf[:, i, size[2] - 1 - j], 0)
    #         else:
    #             l = list(map(float, line.rstrip('\n').split('\t')))
    #         pdf[:, i, j] = l
    #         j += 1
    #         if i == len(betas):
    #             j = 0
    #             i += 1
    np.savez('pdf.npz', pdf)
    # exit()

    del pdf

    print("Generating levy_data.py ...")
    # f = open('temp_cdf', 'a')
    cdf = np.zeros(size, 'float64')
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print("Calculating alpha={:.2f}, beta={:.2f}".format(alpha, beta))
            # if beta > 0.0:
            #     f.write('s\n')
            # else:
            #     l = [_calculate_levy(np.tan(x), alpha, beta, True) for x in xs]
            #     f.write('\t'.join(l)+'\n')
            cdf[:, i, j] = [_calculate_levy(t, alpha, beta, False) for t in ts]
            # for k, x in enumerate(xs):
            #     pdf[k, i, j] = _calculate_levy(np.tan(x), alpha, beta, False)
    # f.close()

    # i = 0
    # j = 0
    # with open('temp_cdf', 'r') as f:
    #     for line in f:
    #         pdf[:, i, j] = list(map(float, line.rstrip('\n').split('\t')))
    #         j += 1
    #         if i == len(betas):
    #             j = 0
    #             i += 1
    np.savez('cdf.npz', cdf)

    del cdf

    # cdf = np.zeros(size, 'float64')
    # for i, alpha in enumerate(alphas):
    #     for j, beta in enumerate(betas):
    #         print("Calculating alpha={:.2f}, beta={:.2f}".format(alpha, beta))
    #         # cdf[:, i, j] = _calculate_levy(np.tan(xs), alpha, beta, True)
    #         for k, x in enumerate(xs):
    #             cdf[k, i, j] = _calculate_levy(np.tan(x), alpha, beta, True)
    # np.savez('cdf.npz', cdf)


def _int_levy(x, alpha, beta, cdf=False):
    """ Interpolate densities of the Levy stable distribution specified by alpha and beta.

        Specify cdf=True to obtain the *cumulative* density function.

        Note: may sometimes return slightly negative values, due to numerical inaccuracies.
    """
    points = np.empty(np.shape(x) + (3,), 'float64')
    points[..., 0] = np.arctan(x)
    points[..., 1] = alpha
    points[..., 2] = np.abs(beta)

    if cdf:
        what = np.load('cdf.npz')['arr_0']
    else:
        what = np.load('pdf.npz')['arr_0']
    return _interpolate(points, what, _lower, _upper)


def _get_closest_approx(alpha, beta):
    x0, x1, n = -50.0, 10000.0 - 50.0, 100000
    dx = (x1 - x0) / n
    x = np.linspace(x0, x1, num=n, endpoint=True)
    y = 1.0 - _int_levy(x, alpha, beta, cdf=True)
    z = 1.0 - _approximate(x, alpha, beta, cdf=True)
    mask = (10.0 < x) & (x < 500.0)
    return 10.0 + dx * np.argmin((np.log(z[mask]) - np.log(y[mask])) ** 2.0)


def _make_limit_data_file():
    # size = (50, 51)
    limits = np.zeros(size[1:], 'float64')
    alphas, betas = [np.linspace(_lower[i], _upper[i], size[i], endpoint=True) for i in [1, 2]]

    print("Generating levy_approx_data.py ...")
    # print(alphas)
    # print(betas)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            limits[i, j] = _get_closest_approx(alpha, beta)
            print("Calculating alpha={:.2f}, beta={:.2f}, limit={:.2f}".format(alpha, beta, limits[i, j]))

    np.savez('limits.npz', limits)


def change_par(alpha, beta, mu, sigma, par_input, par_output):
    if par_input == par_output:
        return mu
    elif (par_input == 0) and (par_output == 1):
        return mu - sigma * _phi(alpha, beta)
    elif (par_input == 1) and (par_output == 0):
        return mu + sigma * _phi(alpha, beta)


def levy(x, alpha, beta, mu=0.0, sigma=1.0, cdf=False, par=0):
    """
    Levy with the tail replaced by the analytical approximation.
    Also, mu, sigma are parameters that shift and rescale the distribution.
    Parametrization can be chosen according to Nolan, par={0,1}.
    """

    loc = change_par(alpha, beta, mu, sigma, par, 0)

    if cdf:
        what = np.load('cdf.npz')['arr_0']
    else:
        what = np.load('pdf.npz')['arr_0']
    limits = np.load('limits.npz')['arr_0']

    xr = (x - loc) / sigma
    alpha_index = int((alpha -_lower[1]) / (_upper[1] - _lower[1]) * (size[1] - 1))
    beta_index = int((beta - _lower[2]) / (_upper[2] - _lower[2]) * (size[2] - 1))
    try:
        l = limits[alpha_index, beta_index]
    except IndexError:
        print(alpha, alpha_index)
        print(beta, beta_index)
        raise
    mask = (np.abs(xr) < l)
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
    return res


def neglog_levy(x, alpha, beta, mu, sigma, par=0):
    """
    Interpolate negative log densities of the Levy stable distribution specified by alpha and beta.
    Small/negative densities are capped at 1e-100 to preserve sanity.
    """
    return -np.log(np.maximum(1e-100, levy(x, alpha, beta, mu, sigma, par=par)))


def fit_levy(x, alpha=None, beta=None, mu=None, sigma=None, par=0):
    """
    Estimate parameters of Levy stable distribution given data x, using the Maximum Likelihood method.

    By default, searches all possible Levy stable distributions.
    However you may restrict the search by specifying the values of one or more parameters.
    Parametrization can be chosen according to Nolan, par={0,1}.
        
    Examples:
        
        levy(x) -- Fit a stable distribution to x

        levy(x, beta=0.0) -- Fit a symmetric stable distribution to x

        levy(x, beta=0.0, mu=0.0) -- Fit a symmetric distribution centered on zero to x

        levy(x, alpha=1.0, beta=0.0) -- Fit a Cauchy distribution to x

    Returns a tuple of (alpha, beta, mu, sigma, negative log density)
    """

    # The parametrization is changed to par=0. At the end, the parametrization will change to par.
    if mu is not None:
        loc = change_par(alpha, beta, mu, sigma, par, 0)
    elif mu is None:
        loc = mu

    from scipy import optimize

    kwargs = {'alpha': alpha, 'beta': beta, 'mu': loc, 'sigma': sigma}
    parameters = Parameters(**kwargs)

    def neglog_density(param):
        p = np.zeros(4)
        p[parameters.variables] = param
        p[parameters.fixed] = parameters.fixed_values
        alpha, beta, mu, sigma = p
        return np.sum(neglog_levy(x, alpha, beta, mu, sigma))

    # parameters.x = optimize.fmin(neglog_density, parameters.x, disp=0)
    parameters.x = optimize.minimize(neglog_density, parameters.x, method='L-BFGS-B', bounds=par_bounds)
    alpha, beta, loc, sigma = parameters.get_all()
    mu = change_par(alpha, beta, loc, sigma, 0, par)

    return alpha, beta, mu, sigma, neglog_density(parameters.x)


def random(alpha, beta, mu=0.0, sigma=1.0, shape=(), par=0):
    """
    Generate random values sampled from an alpha-stable distribution.
    Parametrization can be chosen according to Nolan, par={0,1}.
    """

    loc = change_par(alpha, beta, mu, sigma, par, 0)
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
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return loc + sigma * k


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        _make_dist_data_file()
        _make_limit_data_file()

    print("Testing fit_levy.")

    print("1000 points, result should be (1.5, 0.5, 0.0, 1.0).")
    result = fit_levy(random(1.5, 0.5, 0.0, 1.0, 1000))
    print('alpha={:.2f}, beta={:.2f}, mu_0={:.2f}, sigma={:.2f}, neglog={:.2f}'.format(*result))
