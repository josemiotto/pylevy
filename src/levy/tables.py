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

"""Locating, loading, caching and repairing the lookup tables.

Attributes
----------
ROOT : str
    Directory the installed package lives in.
PACKAGED_DATA : str
    Directory of the tables shipped inside the package.
"""

import os
import sys

import numpy as np

from levy._logging import logger

__all__ = ['data_dir', 'user_cache_dir', 'PACKAGED_DATA', 'ROOT']

ROOT = os.path.dirname(os.path.abspath(__file__))
#: Tables shipped with the package. They used to sit directly in the package
#: directory as four float64 archives; they are now float32 and in data/, with
#: the two crossover-limit tables merged into limits.npz.
PACKAGED_DATA = os.path.join(ROOT, 'data')
_data_cache = {}

_TABLE_NAMES = ('pdf', 'cdf', 'lower_limit', 'upper_limit')


def user_cache_dir():
    """Return the per-user directory where regenerated tables are looked for.

    Returns
    -------
    str
        Path to the cache directory. It is not created here.

    Notes
    -----
    Resolved without a third-party dependency: ``XDG_CACHE_HOME`` or
    ``~/.cache`` on Unix, ``~/Library/Caches`` on macOS, ``LOCALAPPDATA`` on
    Windows.
    """
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser(r'~\AppData\Local')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'pylevy')


def data_dir(writable=False):
    """Return the directory the lookup tables are read from.

    Parameters
    ----------
    writable : bool, default False
        Ask for the directory a *new* build should be written to, rather than
        the one the current tables are read from.

    Returns
    -------
    str
        Path to the directory.

    Notes
    -----
    Search order: ``$LEVY_DATA_DIR``, then the user cache directory if it holds
    a complete set, then the tables shipped inside the package.

    `writable=True` returns where a *new* build should go. Absent an override
    that is the user cache directory, never the installed package: writing
    there fails on a read-only or system install, and a partial run would
    corrupt the installation.

    ``$LEVY_DATA_DIR`` overrides both reads and writes. Pointing it at the
    installed package is therefore possible, but that is the caller saying so
    explicitly rather than the default doing it behind their back.
    """
    override = os.environ.get('LEVY_DATA_DIR')
    if override:
        return override
    cache = user_cache_dir()
    if writable:
        return cache
    if _has_complete_tables(cache):
        return cache
    return PACKAGED_DATA


def _has_complete_tables(directory):
    """Report whether `directory` holds a usable set of tables, in either layout.

    Parameters
    ----------
    directory : str
        Directory to inspect. It need not exist.

    Returns
    -------
    bool
        True when the densities and the crossover limits are all present.

    Notes
    -----
    The crossover limits ship as a single ``limits.npz`` with ``lower`` and
    ``upper`` arrays; tables built by an older version have them as two
    separate files.
    """
    if not all(os.path.exists(os.path.join(directory, f'{n}.npz'))
               for n in ('pdf', 'cdf')):
        return False
    if os.path.exists(os.path.join(directory, 'limits.npz')):
        return True
    return all(os.path.exists(os.path.join(directory, f'{n}.npz'))
               for n in ('lower_limit', 'upper_limit'))


# Cells of cdf.npz that scipy.integrate.quad failed to evaluate when the table
# was generated. All four are at alpha index 4 (alpha = 0.58), beta indices 13
# and 87 (beta = -+0.74), x indices 99 and 100 -- the two grid points closest to
# x = 0, where the oscillatory weight used by _calculate_levy degenerates. They
# hold 5.72e+307 instead of a probability.
#
# This is not a storage error: _calculate_levy still returns 5.72e+307 for those
# arguments today, so regenerating the table with the same code reproduces them.
# Repairing at load time keeps the fix independent of the 12 MB binary; the
# generator needs its own fix before the tables are next rebuilt.
_CDF_TOLERANCE = 1e-6


def _repair_table(key, table):
    """Replace values the table generator failed to compute.

    Parameters
    ----------
    key : {'pdf', 'cdf', 'lower_limit', 'upper_limit'}
        Which table this is. Only ``'cdf'`` is inspected.
    table : ndarray
        The table as loaded from disk. Never modified in place.

    Returns
    -------
    ndarray
        `table` itself when there is nothing to repair, otherwise a repaired
        copy.

    Notes
    -----
    CDF cells outside ``[0, 1]`` (or non-finite) are replaced by linear
    interpolation along x, which is well justified here: the neighbours of the
    known-bad cells are smooth and about 0.0128 apart. A repair is logged once
    at ``WARNING``.
    """
    if key != 'cdf':
        return table

    bad = ~np.isfinite(table) | (table < -_CDF_TOLERANCE) | (table > 1.0 + _CDF_TOLERANCE)
    if not bad.any():
        return table

    table = table.copy()
    x_size = table.shape[0]
    warned = set()
    clipped = 0
    for x_index, alpha_index, beta_index in np.argwhere(bad):
        low = x_index
        while low > 0 and bad[low - 1, alpha_index, beta_index]:
            low -= 1
        high = x_index
        while high < x_size - 1 and bad[high + 1, alpha_index, beta_index]:
            high += 1
        left, right = low - 1, high + 1
        if left < 0 and right > x_size - 1:
            # Every cell in this column is unusable, so there is no good
            # neighbour to interpolate or copy from -- and the copy below
            # would index one past the end. Nothing can be recovered here.
            clipped += 1
            # Once per column: an unusable column is unusable in every one
            # of its x cells, and warning per cell would emit x_size copies
            # of the same line.
            if (alpha_index, beta_index) not in warned:
                warned.add((alpha_index, beta_index))
                logger.warning(
                    'cdf column alpha=%d beta=%d has no usable cell; '
                    'leaving it clipped', alpha_index, beta_index)
            table[x_index, alpha_index, beta_index] = np.clip(
                table[x_index, alpha_index, beta_index], 0.0, 1.0)
            continue
        if left < 0 or right > x_size - 1:
            table[x_index, alpha_index, beta_index] = np.clip(
                table[left if left >= 0 else right, alpha_index, beta_index], 0.0, 1.0)
            continue
        weight = (x_index - left) / float(right - left)
        table[x_index, alpha_index, beta_index] = (
            (1.0 - weight) * table[left, alpha_index, beta_index]
            + weight * table[right, alpha_index, beta_index]
        )

    interpolated = int(bad.sum()) - clipped
    if interpolated:
        logger.warning(
            'Repaired %d unusable cell(s) in the shipped cdf table by '
            'interpolating along x; these are quadrature failures from when '
            'the table was generated. See '
            'https://github.com/josemiotto/pylevy/issues/22',
            interpolated,
        )
    if clipped:
        # Not interpolated: these had no usable neighbour to interpolate
        # from, so the line above must not count them as if they had.
        logger.warning(
            'A further %d cell(s) had no usable neighbour and were only '
            'clipped into [0, 1]; those columns are not trustworthy.',
            clipped,
        )
    return table


def _read_from_cache(key):
    """Return the named table, loading and repairing it on first use.

    Parameters
    ----------
    key : {'pdf', 'cdf', 'lower_limit', 'upper_limit'}
        Which table to return.

    Returns
    -------
    ndarray
        The table. The same object is handed out on every call, so callers must
        treat it as read-only.
    """
    try:
        return _data_cache[key]
    except KeyError:
        _data_cache[key] = _repair_table(key, _load_table(data_dir(), key))
        return _data_cache[key]


def _load_table(directory, key):
    """Read one table from `directory`, in either storage layout.

    Parameters
    ----------
    directory : str
        Directory holding the ``.npz`` archives.
    key : {'pdf', 'cdf', 'lower_limit', 'upper_limit'}
        Which table to read.

    Returns
    -------
    ndarray
        The materialised array.

    Notes
    -----
    ``np.load`` returns a lazy ``NpzFile``; the array is materialised and the
    archive closed rather than leaking the handle until garbage collection.
    """
    if key in ('lower_limit', 'upper_limit'):
        merged = os.path.join(directory, 'limits.npz')
        if os.path.exists(merged):
            with np.load(merged) as archive:
                return archive[key.split('_')[0]]

    with np.load(os.path.join(directory, f'{key}.npz')) as archive:
        return archive[archive.files[0]]
