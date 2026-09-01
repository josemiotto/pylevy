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
``backends``          which array library evaluates: NumPy, or optional torch
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

import importlib
import warnings

from levy._logging import logger  # noqa: F401  (re-exported)
from levy.constants import _lower, _upper  # noqa: F401  (re-exported)
from levy.distribution import (  # noqa: F401  (re-exported)
    _approximate,
    _check_alpha_beta,
    _grid_index,
    _grid_shape,
)
from levy.interpolation import _interpolate, _reflect  # noqa: F401  (re-exported)
from levy.parametrization import _phi, _psi  # noqa: F401  (re-exported)
from levy.sampling import _ALPHA_1_RADIUS  # noqa: F401  (re-exported)
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

__version__ = "2.0.0"

#: The 2.0 surface.
_CURRENT = [
    'api',
    'backends',
    'set_backend',
    'using',
    'data_dir',
    'user_cache_dir',
    'PACKAGED_DATA',
    'ROOT',
    'logger',
    '__version__',
]

# The 1.x names. Every one of them still works and still returns exactly what
# it returned in 1.1; each maps to (module, attribute, what to use instead).
#
# They are resolved through the module __getattr__ below rather than imported
# up here, which is the whole point: the warning fires when a name is *used*,
# not when levy is imported. An import-time warning storm -- one line for every
# deprecated name, on every `import levy`, whether or not the caller touches
# any of them -- is the fastest way to get a deprecation reverted.
_DEPRECATED = {
    'levy': (
        'levy.distribution', 'levy',
        'levy.api.pdf() for a density and levy.api.cdf() for a distribution '
        'function; the cdf= flag is gone',
    ),
    'neglog_levy': (
        'levy.distribution', 'neglog_levy',
        'levy.api.logpdf(), which returns log(pdf) -- note the opposite sign',
    ),
    'fit_levy': (
        'levy.fitting', 'fit_levy',
        'levy.api.fit(), which returns a FitResult and rejects a misspelt '
        'parameter name instead of ignoring it',
    ),
    'random': (
        'levy.sampling', 'random',
        'levy.api.rvs(), which takes size= rather than shape=',
    ),
    'Parameters': (
        'levy.parametrization', 'Parameters',
        'levy.api.StableParams to carry parameters, or '
        'levy.parametrization.Parameters if you need the fitting wrapper that '
        'tracks which components are held fixed',
    ),
    'convert_to_par0': (
        'levy.parametrization', 'convert_to_par0',
        'levy.api.StableParams.from_par(), which validates the result',
    ),
    'convert_from_par0': (
        'levy.parametrization', 'convert_from_par0',
        'levy.api.StableParams.to_par()',
    ),
    'size': ('levy.constants', 'size', 'levy.constants.size'),
    'par_bounds': ('levy.constants', 'par_bounds', 'levy.constants.par_bounds'),
    'par_names': ('levy.constants', 'par_names', 'levy.constants.par_names'),
    'default': ('levy.constants', 'default', 'levy.constants.default'),
    'f_bounds': ('levy.constants', 'f_bounds', 'levy.constants.f_bounds'),
}

# Submodules, resolved on attribute access. `levy.distribution` and friends
# used to become attributes as a side effect of __init__ importing names out of
# them; now that the 1.x names are resolved lazily, that no longer happens, and
# `import levy; levy.sampling.random(...)` would break. `api` and `backends`
# are here for a second reason: resolving them lazily keeps pydantic and torch
# off the critical path of `import levy`.
_SUBMODULES = (
    'api',
    'constants',
    'distribution',
    'fitting',
    'interpolation',
    'parametrization',
    'sampling',
    'tables',
)

# The table-generation helpers moved to levy._build. Resolved lazily too, so
# importing levy does not pull in scipy.integrate for the sake of code only a
# maintainer regenerating tables ever runs.
_MOVED_TO_BUILD = {
    '_calculate_levy': 'calculate_levy',
    '_int_levy': 'interpolated_levy',
}

__all__ = _CURRENT + sorted(_DEPRECATED)


def __getattr__(name):
    """Resolve the 1.x names, `levy.api`, and the ``levy._build`` helpers.

    Parameters
    ----------
    name : str
        Attribute being looked up.

    Returns
    -------
    object
        The requested object, unchanged. A deprecated name additionally emits a
        :exc:`DeprecationWarning` naming its replacement.

    Raises
    ------
    AttributeError
        For any other name, as a module lookup normally would.

    Notes
    -----
    PEP 562 module-level lookup. Nothing here is a wrapper: a deprecated name
    resolves to the very same object 1.1 exported, so numbers cannot drift
    between the old spelling and the new one. Only the lookup is intercepted.
    """
    if name in ('api', 'backends'):
        return importlib.import_module(f'levy.{name}')

    if name in ('set_backend', 'using'):
        # levy.backends imports nothing heavier than levy._compat, so this
        # costs nothing for a caller who never selects a backend.
        return getattr(importlib.import_module('levy.backends'), name)

    if name in _DEPRECATED:
        module_name, attribute, replacement = _DEPRECATED[name]
        warnings.warn(
            f'levy.{name} is deprecated since 2.0 and will be removed in 3.0; '
            f'use {replacement}. It still lives at {module_name}.{attribute} '
            f'if you want the 1.x behaviour without the warning.',
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_name), attribute)

    if name in _MOVED_TO_BUILD:
        return getattr(importlib.import_module('levy._build'), _MOVED_TO_BUILD[name])

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    """List the package's attributes, deprecated names included.

    Returns
    -------
    list of str
        Sorted names, so tab completion and ``dir(levy)`` still show the 1.x
        spellings that lazy resolution keeps out of the module dictionary.
    """
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES)
                  | set(_MOVED_TO_BUILD))
