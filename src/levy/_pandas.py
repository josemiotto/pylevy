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

"""Unwrapping pandas objects on the way in, and restoring them on the way out.

Everything here is boundary work. The numerical core never sees a ``Series`` or
a ``DataFrame``: :mod:`levy.api` unwraps to a plain array, computes, and puts
the labels back. A density evaluated at a labelled index should come back with
that index attached -- losing it silently is how a misaligned join happens.

pandas is optional and is never imported on its behalf. Detection goes through
:func:`levy._compat.loaded`: if pandas is not already in ``sys.modules``, the
object the caller passed cannot be a pandas object.
"""

from typing import Any, Optional

import numpy as np

from levy._compat import loaded
from levy._typing import FloatArray

__all__ = ['labels_of', 'relabel']


def labels_of(x: Any) -> Optional[dict[str, Any]]:
    """Describe how to put labels back on a result computed from `x`.

    Parameters
    ----------
    x : object
        Whatever the caller passed as the evaluation points.

    Returns
    -------
    dict or None
        None if `x` carries no pandas labels. Otherwise a description of them,
        to be handed to :func:`relabel`.

    Notes
    -----
    Returning a plain description rather than the object itself keeps this
    module free of pandas types in its signatures, which is what lets
    :mod:`levy.api` stay importable and type-checkable without pandas.
    """
    pandas = loaded('pandas')
    if pandas is None:
        return None

    if isinstance(x, pandas.Series):
        return {'kind': 'series', 'index': x.index, 'name': x.name}
    if isinstance(x, pandas.DataFrame):
        return {'kind': 'frame', 'index': x.index, 'columns': x.columns}
    return None


def relabel(values: Any, labels: Optional[dict[str, Any]]) -> Any:
    """Put pandas labels back on a computed array.

    Parameters
    ----------
    values : ndarray
        The result, shaped like the input it was computed from.
    labels : dict or None
        The description returned by :func:`labels_of`. None returns `values`
        unchanged, which is the path every non-pandas caller takes.

    Returns
    -------
    ndarray or Series or DataFrame
        `values` itself when `labels` is None; otherwise a pandas object
        carrying the original index, and the original name or columns.
    """
    if labels is None:
        return values

    import pandas  # already imported, or labels_of would have returned None

    if labels['kind'] == 'series':
        return pandas.Series(np.asarray(values), index=labels['index'],
                             name=labels['name'])
    return pandas.DataFrame(np.asarray(values), index=labels['index'],
                            columns=labels['columns'])


def as_sample(x: Any) -> FloatArray:
    """Reduce a fitting input to a one-dimensional array of observations.

    Parameters
    ----------
    x : array_like or Series or DataFrame
        The sample. A one-column DataFrame is accepted; a wider one is not,
        because there is no single sensible reading of it.

    Returns
    -------
    ndarray
        The observations, as float64.

    Raises
    ------
    ValueError
        If a DataFrame has more than one column. Flattening it would fit one
        distribution to several variables pooled together, silently -- which is
        almost never what was meant.
    """
    pandas = loaded('pandas')
    if pandas is not None and isinstance(x, pandas.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f'expected a Series or a single-column DataFrame, got '
                f'{x.shape[1]} columns. Fit one column at a time, or pass '
                f'df[column]; pooling them would fit one distribution to '
                f'several variables at once.'
            )
        x = x.iloc[:, 0]
    return np.asarray(x, dtype='d')
