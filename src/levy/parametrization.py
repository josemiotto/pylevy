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

Nolan 0 and 1 [Nolan2020]_, Zolotarev M, A and B [Zolotarev1986]_. Everything is
routed through parametrization 0, which is what the lookup tables are built in.

Attributes
----------
convert_to_par0 : dict of {str: callable}
    Maps a parameter array from the named parametrization into 0.
convert_from_par0 : dict of {str: callable}
    Maps a parameter array from parametrization 0 into the named one.

References
----------
.. [Nolan2020] J. P. Nolan, "Univariate Stable Distributions", Springer, 2020.
.. [Zolotarev1986] V. M. Zolotarev, "One-dimensional Stable Distributions",
   AMS, 1986.
"""

import numpy as np
from scipy import optimize

from levy.constants import default, f_bounds, par_names

__all__ = ['Parameters', 'convert_to_par0', 'convert_from_par0']


def _psi(alpha):
    """Return Zolotarev's psi, the half-turn used by parametrization B.

    Parameters
    ----------
    alpha : float or ndarray
        Index of stability.

    Returns
    -------
    float or ndarray
        ``pi / 2 * (alpha - 1 - sign(alpha - 1))``.
    """
    return np.pi / 2 * (alpha - 1 - np.sign(alpha - 1))


def _phi(alpha, beta):
    """Return the skewness term shared by every parametrization change.

    Parameters
    ----------
    alpha : float or ndarray
        Index of stability.
    beta : float or ndarray
        Skewness.

    Returns
    -------
    float or ndarray
        ``beta * tan(pi * alpha / 2)``, which diverges as `alpha` approaches 1.
    """
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


class Parameters:
    """A parameter vector that knows which of its components are free.

    Fitting with one or more parameters held fixed works by letting the
    optimizer see only the free components: :attr:`x` reads and writes that
    subvector, and the setter folds each proposed value back into its feasible
    range. The only method generally useful on its own is :meth:`convert`.

    Parameters
    ----------
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the values are given in.
    **kwargs
        One entry per name in ``par_names[par]``. ``None`` marks a free
        parameter, which starts from its default; anything else fixes it.

    Attributes
    ----------
    par : str
        The parametrization in use.
    pnames : list of str
        Parameter names, in positional order.
    variables : list of int
        Positions of the free parameters.
    fixed : list of int
        Positions of the fixed parameters.
    fixed_values : list
        The values those fixed positions were pinned to.

    Examples
    --------
    >>> p = Parameters(par='0', alpha=1.5, beta=None, mu=0.0, sigma=None)
    >>> p.variables
    [1, 3]
    >>> p.x
    array([0., 1.])
    """

    @classmethod
    def convert(cls, pars, par_in, par_out):
        """Convert a parameter array from one parametrization to another.

        Parameters
        ----------
        pars : ndarray
            Array of parameters to be converted.
        par_in : {'0', '1', 'M', 'A', 'B'}
            Parametrization of the input array.
        par_out : {'0', '1', 'M', 'A', 'B'}
            Parametrization of the output array.

        Returns
        -------
        ndarray
            Array of parameters in the desired parametrization. `pars` itself
            is returned when the two parametrizations coincide.

        Examples
        --------
        >>> a = np.array([1.6, 0.5, 0.3, 1.2])
        >>> b = Parameters.convert(a, '1', 'B')
        >>> np.round(b, 6)
        array([1.6     , 0.554573, 0.246008, 1.424317])
        >>> c = Parameters.convert(b, 'B', '1')
        >>> np.round(c, 6)
        array([1.6, 0.5, 0.3, 1.2])
        >>> np.testing.assert_allclose(a, c)
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
        self._x = np.array(
            [default[par][k] if kwargs[k] is None else kwargs[k] for k in self.pnames])
        self.variables = [i for i, k in enumerate(self.pnames) if kwargs[k] is None]
        self.fixed = [i for i, k in enumerate(self.pnames) if kwargs[k] is not None]
        self.fixed_values = [kwargs[k] for k in self.pnames if kwargs[k] is not None]

    def get(self, par_out=None):
        """Return the full parameter vector, optionally in another parametrization.

        Parameters
        ----------
        par_out : {'0', '1', 'M', 'A', 'B'}, optional
            Parametrization to return. Defaults to the object's own.

        Returns
        -------
        ndarray
            All four parameters, fixed and free alike.

        Examples
        --------
        >>> p = Parameters(par='1', alpha=1.5, beta=0.5, mu=0, sigma=1.2)
        >>> np.round(p.get('B'), 6)   # the same distribution, written in B
        array([1.5     , 0.590334, 0.038965, 1.469694])
        """
        if par_out is None:
            par_out = self.par
        return Parameters.convert(self._x, self.par, par_out)

    def __str__(self):
        """Render the parameters and their parametrization, for humans.

        Returns
        -------
        str
            Formatted as ``"alpha: 1.50, ... . Parametrization: 1."``.
        """
        body = ', '.join(
            f'{name}: {value:.2f}' for name, value in zip(self.pnames, self.get()))
        return f'{body}. Parametrization: {self.par}.'

    def __repr__(self):
        """Render the parametrization and the parameters, keyword-style.

        Returns
        -------
        str
            Formatted as ``"par=1, alpha=1.50, beta=0.50, ..."``.
        """
        body = ', '.join(
            f'{name}={value:.2f}' for name, value in zip(self.pnames, self.get()))
        return f'par={self.par}, {body}'

    @property
    def x(self):
        """Get the free parameters only, in positional order.

        Assigning to this writes those positions back, folding each value into
        its feasible range first. It accepts a
        ``scipy.optimize.OptimizeResult``, an ndarray, a list or a tuple, which
        is what lets ``optimize.minimize``'s result be handed straight back.

        Returns
        -------
        ndarray
            The components listed in :attr:`variables`.
        """
        return self._x[self.variables]

    @x.setter
    def x(self, values):
        """Write the free parameters back, folded into their feasible ranges.

        Parameters
        ----------
        values : OptimizeResult or ndarray or list or tuple
            One value per free parameter, in :attr:`variables` order.

        Raises
        ------
        TypeError
            If `values` is none of the accepted types.
        ValueError
            If `values` does not hold exactly one entry per free parameter.
        """
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
                f'got {type(values).__name__}'
            )
        # One value per free parameter. Without this the loop below indexes
        # off the end and reports a bare IndexError naming neither the setter
        # nor the length it wanted.
        if len(vals) != len(self.variables):
            raise ValueError(
                f'expected {len(self.variables)} value(s) for the free '
                f'parameters {[self.pnames[i] for i in self.variables]}, '
                f'got {len(vals)}'
            )
        for j, i in enumerate(self.variables):
            self._x[i] = f_bounds[self.par][self.pnames[i]](vals[j])
