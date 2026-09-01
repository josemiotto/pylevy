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
    res = optimize.minimize(neglog_density, parameters.x, method='L-BFGS-B', bounds=bounds)
    parameters.x = res.x

    return parameters, neglog_density(parameters.x)
