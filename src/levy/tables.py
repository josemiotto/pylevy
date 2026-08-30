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

"""Locating, loading, caching and repairing the lookup tables."""

import os
import sys
import zipfile

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
    """ Per-user directory where regenerated tables are looked for.

    Resolved without a third-party dependency: XDG_CACHE_HOME or ~/.cache on
    Unix, ~/Library/Caches on macOS, LOCALAPPDATA on Windows.
    """
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser(r'~\AppData\Local')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'pylevy')


def data_dir(writable=False):
    """ Directory the lookup tables are read from.

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
    """ True if `directory` holds a usable set of tables, in either layout.

    The crossover limits ship as a single limits.npz with `lower` and `upper`
    arrays; tables built by an older version have them as two separate files.
    """
    if not all(os.path.exists(os.path.join(directory, '{}.npz'.format(n)))
               for n in ('pdf', 'cdf')):
        return False
    if os.path.exists(os.path.join(directory, 'limits.npz')):
        return True
    return all(os.path.exists(os.path.join(directory, '{}.npz'.format(n)))
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
    """ Replaces values the table generator failed to compute.

    CDF cells outside [0, 1] (or non-finite) are replaced by linear
    interpolation along x, which is well justified here: the neighbours of the
    known-bad cells are smooth and about 0.0128 apart. A bad cell at either
    end of the x range has a usable neighbour on one side only and is copied
    from it instead. A repair is logged once at ``WARNING``.
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
    copied = 0
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
                # Grid indices, not parameter values: the message is for
                # locating the cells in the file, and the values would be
                # ambiguous across table resolutions anyway.
                logger.warning(
                    'cdf column at alpha index %d, beta index %d has no '
                    'usable cell; leaving it clipped', alpha_index, beta_index)
            # np.clip passes NaN through, so a non-finite cell has to be
            # given a value explicitly: -inf becomes 0, +inf and NaN become
            # 1. No number is *right* for a column like this -- the point is
            # that the table comes back finite, as promised, so the
            # interpolator cannot spread NaN into every neighbour it touches.
            value = table[x_index, alpha_index, beta_index]
            if not np.isfinite(value):
                value = 0.0 if value < 0.0 else 1.0
            table[x_index, alpha_index, beta_index] = np.clip(value, 0.0, 1.0)
            continue
        if left < 0 or right > x_size - 1:
            copied += 1
            table[x_index, alpha_index, beta_index] = np.clip(
                table[left if left >= 0 else right, alpha_index, beta_index], 0.0, 1.0)
            continue
        weight = (x_index - left) / float(right - left)
        table[x_index, alpha_index, beta_index] = (
            (1.0 - weight) * table[left, alpha_index, beta_index]
            + weight * table[right, alpha_index, beta_index]
        )

    interpolated = int(bad.sum()) - clipped - copied
    if interpolated:
        logger.warning(
            'Repaired %d unusable cell(s) in the shipped cdf table by '
            'interpolating along x; these are quadrature failures from when '
            'the table was generated. See '
            'https://github.com/josemiotto/pylevy/issues/22',
            interpolated,
        )
    if copied:
        # Not interpolated either: one usable neighbour is not enough to
        # interpolate between, so the cell took that neighbour's value.
        logger.warning(
            'A further %d cell(s) at an end of the x range had a usable '
            'neighbour on one side only and were copied from it.',
            copied,
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
    """ Loads the file given by key """
    try:
        return _data_cache[key]
    except KeyError:
        directory = data_dir()
        table = _load_table(directory, key)
        _check_table_shape(key, table, directory)
        _data_cache[key] = _repair_table(key, table)
        return _data_cache[key]


def _check_table_shape(key, table, directory):
    """ Refuse a table whose shape does not match the pdf table's.

    A directory is chosen by `data_dir()` on the strength of which files
    exist, not what is in them, so a partial rebuild -- `levy-tables build
    --what pdf --size ...` into a cache that already held a full set -- could
    leave pdf.npz at one resolution next to a cdf and limits at another. The
    grid index is derived from the pdf table's shape, so the mismatched
    tables would be read with the wrong indices and return wrong numbers
    rather than fail. Every table is therefore checked against the pdf.
    """
    if key == 'pdf':
        return
    reference = _read_from_cache('pdf').shape
    expected = reference if key == 'cdf' else reference[1:]
    if tuple(table.shape) != tuple(expected):
        raise RuntimeError(
            'the lookup tables in {} do not belong together: {} is {} but the pdf '
            'table is {}. A partial rebuild has probably left tables of two '
            'resolutions side by side; rebuild the whole set with `levy-tables '
            'build`, or delete the directory to go back to the packaged tables.'.format(
                directory, key, 'x'.join(map(str, table.shape)),
                'x'.join(map(str, reference))))


def _load_table(directory, key):
    """ Read one table from `directory`, in either storage layout.

    np.load returns a lazy NpzFile; the array is materialised and the archive
    closed rather than leaking the handle until garbage collection.
    """
    if key in ('lower_limit', 'upper_limit'):
        merged = os.path.join(directory, 'limits.npz')
        if os.path.exists(merged):
            return _load_array(merged, directory, key.split('_')[0])
    return _load_array(os.path.join(directory, '{}.npz'.format(key)), directory, None)


def _load_array(path, directory, name):
    """ Read one array out of an archive, or say exactly why that failed. """
    try:
        with np.load(path) as archive:
            return archive[archive.files[0] if name is None else name]
    except (OSError, EOFError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
        # Everything np.load raises for a missing, empty, truncated or foreign
        # file, plus KeyError/IndexError for an archive without the array.
        raise _unreadable_table(path, directory, error) from error


def _unreadable_table(path, directory, error):
    """Build the error for a table that could not be loaded.

    Parameters
    ----------
    path : str
        The archive that failed to load.
    directory : str
        The directory ``data_dir()`` chose, which decides what the remedy is.
    error : Exception
        What ``np.load`` raised.

    Returns
    -------
    RuntimeError
        Says which file, why it failed, where the file came from, and what to
        do about it -- a bare ``BadZipFile: File is not a zip file`` from
        three frames down says none of those.
    """
    what = 'cannot read the lookup table {} ({}: {}).'.format(
        path, type(error).__name__, error)
    if os.environ.get('LEVY_DATA_DIR'):
        return RuntimeError(
            what + ' $LEVY_DATA_DIR points at {}; fix or rebuild the tables '
            'there, or unset it to fall back to the packaged ones.'.format(directory))
    if directory == PACKAGED_DATA:
        return RuntimeError(
            what + ' This is the copy shipped inside the package, so the '
            'installation itself is damaged; reinstall it.')
    return RuntimeError(
        what + ' The user cache at {} holds a table that cannot be read. Delete '
        'that directory to go back to the packaged tables, or rerun '
        '`levy-tables build`; `levy-tables where` shows which is in use.'.format(directory))
