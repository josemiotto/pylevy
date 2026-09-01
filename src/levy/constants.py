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

"""Grid geometry and parameter metadata shared across the package.

Kept in one place so the interpolation grid, the fit bounds and the
parametrization tables cannot drift apart.
"""

import numpy as np

from levy.interpolation import _reflect

__all__ = ['size', 'par_bounds', 'par_names', 'default', 'f_bounds']

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
