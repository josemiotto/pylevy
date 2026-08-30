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

"""Maximum-likelihood fitting."""

import numpy as np
from scipy import optimize

from levy.constants import par_bounds, par_names
from levy.distribution import neglog_levy
from levy.parametrization import Parameters

__all__ = ['fit_levy']

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
