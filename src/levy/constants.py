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

"""Grid geometry and parameter metadata shared across the package.

Kept in one place so the interpolation grid, the fit bounds and the
parametrization tables cannot drift apart.

Attributes
----------
size : tuple of int
    Shape ``(x, alpha, beta)`` of the lookup tables shipped with the package.
par_bounds : tuple of tuple
    Box constraints handed to L-BFGS-B, in parametrization-0 order
    ``(alpha, beta, mu, sigma)``. ``(None, None)`` means unbounded.
par_names : dict of {str: list of str}
    Parameter names for each of the five parametrizations. The order is
    positional and shared with ``par_bounds``.
default : dict of {str: dict}
    Starting value of each parameter, per parametrization, used for the
    components a caller did not fix.
f_bounds : dict of {str: dict}
    Per-parameter callables that fold a proposed value back into its feasible
    range; see :func:`levy.interpolation._reflect`.
"""

import numpy as np

from levy.interpolation import _reflect

__all__ = ['size', 'par_bounds', 'par_names', 'default', 'f_bounds']

# Some constants of the program.
# Dimensions: 0 - x, 1 - alpha, 2 - beta
#: Size of the grid (xs, alpha, beta).
size = (200, 76, 101)
#: Lower limit of the grid along each dimension. ``x`` is stored in tan-space,
#: hence the ``pi/2`` bounds, shrunk by 0.999 to stay off the pole.
_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])
#: Upper limit of the grid along each dimension.
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])

#: Parameter bounds for the fit, in parametrization-0 order.
par_bounds = (
    (_lower[1], _upper[1]),   # alpha
    (_lower[2], _upper[2]),   # beta
    (None, None),             # mu
    (1e-6, 1e10),             # sigma
)

#: Names of the parameters, per parametrization.
par_names = {
    '0': ['alpha', 'beta', 'mu', 'sigma'],
    '1': ['alpha', 'beta', 'mu', 'sigma'],
    'M': ['alpha', 'beta', 'gamma', 'lambda'],
    'A': ['alpha', 'beta', 'gamma', 'lambda'],
    'B': ['alpha', 'beta', 'gamma', 'lambda'],
}

# Positional defaults and bound-folders, expanded below into the per-name dicts
# that are the public surface. Named separately rather than rebound, so that
# `default` and `f_bounds` mean one thing only.
_default_values = [1.5, 0.0, 0.0, 1.0]
_bound_folders = [
    lambda x: _reflect(x, *par_bounds[0]),
    lambda x: _reflect(x, *par_bounds[1]),
    lambda x: x,
    lambda x: _reflect(x, *par_bounds[3]),
]

#: Default value of each parameter for the fit, keyed by parametrization.
default = {
    par: {names[i]: _default_values[i] for i in range(4)}
    for par, names in par_names.items()
}
#: Bound-folding callable for each parameter, keyed by parametrization.
f_bounds = {
    par: {names[i]: _bound_folders[i] for i in range(4)}
    for par, names in par_names.items()
}
