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

"""Conversion between the five parametrizations, and the Parameters wrapper.

Nolan 0 and 1, Zolotarev M, A and B. Everything is routed through
parametrization 0, which is what the lookup tables are built in.
"""

import numpy as np
from scipy import optimize

from levy.constants import default, f_bounds, par_names

__all__ = ['Parameters', 'convert_to_par0', 'convert_from_par0']

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
        # One value per free parameter. Without this the loop below indexes
        # off the end and reports a bare IndexError naming neither the setter
        # nor the length it wanted.
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
