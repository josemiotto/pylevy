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

Module layout
-------------
The implementation used to be a single 800-line ``__init__.py``. It is now
split by concern, and this module re-exports the whole public surface, so
``import levy`` behaves exactly as before:

===================== =========================================================
``constants``         grid geometry, fit bounds, parametrization metadata
``interpolation``     Catmull-Rom interpolation and bound folding
``tables``            locating, loading, caching and repairing the tables
``parametrization``   the five parametrizations and the ``Parameters`` wrapper
``distribution``      ``levy`` and ``neglog_levy``
``fitting``           ``fit_levy``
``sampling``          ``random``
``_build``            offline table generation (only the CLI imports it)
===================== =========================================================
"""

from levy._logging import logger  # noqa: F401  (re-exported)
from levy.constants import (  # noqa: F401  (re-exported)
    _lower,
    _upper,
    default,
    f_bounds,
    par_bounds,
    par_names,
    size,
)
from levy.distribution import (  # noqa: F401  (re-exported)
    _approximate,
    _check_alpha_beta,
    _grid_index,
    _grid_shape,
    levy,
    neglog_levy,
)
from levy.fitting import fit_levy  # noqa: F401  (re-exported)
from levy.interpolation import _interpolate, _reflect
from levy.parametrization import (  # noqa: F401  (re-exported)
    Parameters,
    _phi,
    _psi,
    convert_from_par0,
    convert_to_par0,
)
from levy.sampling import (  # noqa: F401  (re-exported)
    _ALPHA_1_RADIUS,
    random,
)
from levy.tables import (  # noqa: F401  (re-exported)
    PACKAGED_DATA,
    ROOT,
    _CDF_TOLERANCE,
    _data_cache,
    _has_complete_tables,
    _load_table,
    _read_from_cache,
    _repair_table,
    _TABLE_NAMES,
    data_dir,
    user_cache_dir,
)

__version__ = "1.1"

__all__ = [
    # distribution
    'levy',
    'neglog_levy',
    # fitting
    'fit_levy',
    # sampling
    'random',
    # parametrizations
    'Parameters',
    'convert_to_par0',
    'convert_from_par0',
    'par_names',
    'par_bounds',
    'default',
    'f_bounds',
    'size',
    # tables
    'data_dir',
    'user_cache_dir',
    'PACKAGED_DATA',
    'ROOT',
    '__version__',
]

# Backwards compatibility: the table-generation helpers live in levy._build.
# Exposed lazily (PEP 562) so importing levy does not pull in scipy.integrate
# for the sake of code only a maintainer regenerating tables ever runs.
_MOVED_TO_BUILD = {
    '_calculate_levy': 'calculate_levy',
    '_int_levy': 'interpolated_levy',
}


def __getattr__(name):
    if name in _MOVED_TO_BUILD:
        from levy import _build
        return getattr(_build, _MOVED_TO_BUILD[name])
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))
