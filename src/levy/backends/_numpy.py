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

"""The default backend: the package's own NumPy implementation.

Deliberately nothing but a re-export. The numerical core is not reorganised to
accommodate a second backend, so adding torch cannot have moved a NumPy number
-- there is no shared code to have changed.
"""

import numpy as np

from levy.distribution import levy as _levy

__all__ = ['cdf', 'name', 'neglog', 'pdf']

#: Identifies this backend in error messages and tests.
name = 'numpy'


def pdf(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the density.

    Parameters
    ----------
    x : array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : float
        Parameters, in parametrization 0.

    Returns
    -------
    float or ndarray
        The density at `x`.
    """
    return _levy(x, alpha, beta, mu, sigma, cdf=False)


def cdf(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the distribution function.

    Parameters
    ----------
    x : array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : float
        Parameters, in parametrization 0.

    Returns
    -------
    float or ndarray
        The distribution function at `x`.
    """
    return _levy(x, alpha, beta, mu, sigma, cdf=True)


def neglog(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the negative log density.

    Parameters
    ----------
    x : array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : float
        Parameters, in parametrization 0.

    Returns
    -------
    float or ndarray
        ``-log(pdf(x))``, floored so the logarithm stays finite.
    """
    return -np.log(np.maximum(1e-100, _levy(x, alpha, beta, mu, sigma, cdf=False)))
