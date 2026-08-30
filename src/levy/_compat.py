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

"""Guards for the optional dependencies.

The rule these enforce: an optional dependency that is not installed costs
nothing and is never imported. Support for it is detected by looking in
``sys.modules``, not by trying an import -- if a caller has not imported pandas,
the object they just handed us cannot be a pandas object, so there is nothing
to check and nothing to load.

That distinction matters for import time and for the error a user sees. Probing
with ``try: import pandas`` in a hot path would pay the lookup on every call;
here the fast path is one dictionary get.
"""

import sys
from types import ModuleType
from typing import Optional

__all__ = ['have', 'loaded', 'require']

#: Optional dependencies, and the extra that installs each.
_EXTRAS: dict[str, str] = {
    'pandas': 'pandas',
    'torch': 'torch',
}


def loaded(name: str) -> Optional[ModuleType]:
    """Return the module only if the caller has already imported it.

    Parameters
    ----------
    name : str
        Top-level module name, e.g. ``'pandas'``.

    Returns
    -------
    module or None
        The module object if it is in ``sys.modules``, otherwise None. Never
        imports anything.

    Examples
    --------
    >>> loaded('sys') is sys
    True
    >>> loaded('a_module_nobody_has_imported') is None
    True
    """
    return sys.modules.get(name)


def have(name: str) -> bool:
    """Report whether an optional dependency can be imported.

    Parameters
    ----------
    name : str
        Top-level module name.

    Returns
    -------
    bool
        True if the module is importable. This does *not* import it: it asks
        the import system whether a spec exists, which touches the filesystem
        but leaves ``sys.modules`` alone. :func:`loaded` differs in reporting
        only whether the module has *already* been imported.

    Examples
    --------
    >>> have('sys')
    True
    >>> have('a_module_nobody_has_installed')
    False
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require(name: str) -> ModuleType:
    """Import an optional dependency, or explain how to install it.

    Parameters
    ----------
    name : str
        Top-level module name.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If it is not installed, naming the extra that provides it rather than
        leaving the user to guess. If it *is* installed but fails to import
        -- a broken install, or a dependency of its own missing -- the error
        says that instead, and chains the real cause; blaming the install
        would send the user off to reinstall something that is there.
    """
    import importlib
    import importlib.util

    try:
        return importlib.import_module(name)
    except ImportError as error:
        # find_spec itself can raise: a finder that refuses the name (as the
        # no-pandas tests do) or a half-initialised module both count as
        # "not there".
        try:
            installed = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            installed = False
        if not installed:
            extra = _EXTRAS.get(name, name)
            raise ImportError(
                f'{name} is needed for this, and is not installed. '
                f'Install it with: pip install "pylevy[{extra}]"'
            ) from error
        raise ImportError(
            f'{name} is installed but failed to import: {error}. The install '
            f'is broken, or something {name} itself depends on is missing; '
            f'see the chained exception.'
        ) from error
