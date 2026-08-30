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

r"""Calculation and maximum-likelihood fitting of Levy alpha-stable distributions.

Direct computation of these distributions requires a lengthy numerical
integration, so the package interpolates values from a precomputed table
instead. That is what makes fitting by maximum likelihood fast enough to be
practical.

Notes
-----
**Parametrizations.** The parameters of a Levy stable distribution can be
written down in several ways. Available here are 0 and 1 in the notation of
Nolan [1]_, and M, A and B from Zolotarev [2]_.

Nolan's are the easier two to reason about. Parametrization 0 is typically
preferred for numerical calculations and has
:math:`E(X)=\delta_0-\beta\gamma\tan(\pi\alpha/2)`, while 1 is preferred for
intuition, since :math:`E(X)=\delta_1`.

Parametrizations are handled by the module; you only say which one you are
using. :meth:`~levy.parametrization.Parameters.convert` moves a parameter array
between any two of them. Internally everything runs in parametrization 0, which
is what the lookup tables are built in.

``alpha`` below 0.5 is not supported: the tables do not cover it, and passing a
smaller value raises :exc:`ValueError`.

**Module layout.** The implementation used to be a single 800-line
``__init__.py``. It is now split by concern, and this module re-exports the
whole public surface, so ``import levy`` behaves exactly as before:

===================== =========================================================
``constants``         grid geometry, fit bounds, parametrization metadata
``interpolation``     Catmull-Rom interpolation and bound folding
``tables``            locating, loading, caching and repairing the tables
``parametrization``   the five parametrizations and the ``Parameters`` wrapper
``distribution``      ``levy`` and ``neglog_levy``
``fitting``           ``fit_levy``
``sampling``          ``random``
``api``               the typed, validated API: ``pdf``/``cdf``/``rvs``/``fit``
``_build``            offline table generation (only the CLI imports it)
===================== =========================================================

References
----------
.. [1] J. P. Nolan, "Univariate Stable Distributions", Springer, 2020.
       https://edspace.american.edu/jpnolan/stable/
.. [2] V. M. Zolotarev, "One-dimensional Stable Distributions", AMS, 1986.

Examples
--------
>>> import numpy as np
>>> import levy
>>> np.round(levy.levy(np.array([1.0, 2.0]), 1.5, 0.0, cdf=True), 6)
array([0.756342, 0.89496 ])
>>> np.random.seed(0)
>>> x = levy.random(1.5, 0.0, 0.0, 1.0, shape=(200,))
>>> levy.fit_levy(x)[0]
par=0, alpha=1.52, beta=-0.08, mu=0.05, sigma=0.99
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
from levy.interpolation import _interpolate, _reflect  # noqa: F401  (re-exported)
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
    _CDF_TOLERANCE,
    _TABLE_NAMES,
    PACKAGED_DATA,
    ROOT,
    _data_cache,
    _has_complete_tables,
    _load_table,
    _read_from_cache,
    _repair_table,
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
    """Resolve `levy.api`, and the helpers that moved into ``levy._build``.

    PEP 562 module-level lookup, so ``levy._calculate_levy`` still works
    without ``import levy`` pulling in ``scipy.integrate``, and ``levy.api``
    resolves without it pulling in pydantic.

    Parameters
    ----------
    name : str
        Attribute being looked up.

    Returns
    -------
    object
        The relocated helper.

    Raises
    ------
    AttributeError
        For any other name, as a module lookup normally would.
    """
    if name == 'api':
        import levy.api as api_module
        return api_module
    if name in _MOVED_TO_BUILD:
        from levy import _build
        return getattr(_build, _MOVED_TO_BUILD[name])
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
