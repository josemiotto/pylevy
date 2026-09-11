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

"""Multi-dimensional Catmull-Rom interpolation, and bound folding.

Deliberately free of any Levy-specific knowledge: this module knows about
grids and points, nothing about alpha or beta. That is what lets the tests
exercise it against a tiny synthetic grid instead of the 10 MB tables.
"""

import numpy as np

__all__ = ['_interpolate', '_reflect']


def _reflect(x, lower, upper):
    """Fold a value back inside the bounds, reflecting at each edge.

    Parameters
    ----------
    x : float
        The value to fold.
    lower : float
        Lower bound, inclusive.
    upper : float
        Upper bound, inclusive.

    Returns
    -------
    float
        ``x`` unchanged when it is already inside ``[lower, upper]``, otherwise
        the point reached by reflecting it off the two edges. A zero span
        collapses to ``lower``.

    Raises
    ------
    ValueError
        If ``upper`` lies below ``lower``. Silently returning ``lower`` there
        would hide a caller bug; the fixed bound boxes never produce it.

    Notes
    -----
    The in-bounds case returns ``x`` unchanged and bit-identical, which is what
    happens on essentially every call: L-BFGS-B already respects the bounds it
    is given. Out of bounds, the fold is computed in closed form rather than by
    repeated reflection, which used to be an unbounded ``while 1:`` loop --
    with the sigma bounds ``(1e-6, 1e10)``, reflecting ``1e30`` needs ~1e20
    iterations and never returns in practice.

    Examples
    --------
    >>> _reflect(0.5, 0.0, 1.0)
    0.5
    >>> _reflect(1.25, 0.0, 1.0)   # 0.25 past the top edge, so 0.25 back down
    0.75
    >>> _reflect(-0.25, 0.0, 1.0)
    0.25
    """
    if lower <= x <= upper:
        return x

    span = upper - lower
    if span < 0:
        raise ValueError(
            f"reflection bounds are reversed: lower={lower}, upper={upper}"
        )
    if span == 0:
        return lower

    offset = (x - lower) % (2.0 * span)
    return lower + (2.0 * span - offset if offset > span else offset)


def _interpolate(points, grid, lower, upper):
    """Perform multi-dimensional Catmull-Rom cubic interpolation.

    Parameters
    ----------
    points : ndarray
        Coordinates to evaluate at, shaped ``(..., ndim)``. The last axis
        indexes the dimension and must match the rank of ``grid``.
    grid : ndarray
        Samples of the function on a regular ``ndim``-dimensional lattice.
    lower : ndarray
        Coordinate of the first grid node along each dimension, shape
        ``(ndim,)``.
    upper : ndarray
        Coordinate of the last grid node along each dimension, shape
        ``(ndim,)``.

    Returns
    -------
    ndarray
        Interpolated values, shaped like ``points`` without its last axis.

    Notes
    -----
    Catmull-Rom takes its tangents from centred differences, which are exact
    for a quadratic but not for a cubic; it therefore reproduces quadratics to
    machine precision. Indices are clamped at the edges, so points outside
    ``[lower, upper]`` are extrapolated from the boundary cell rather than
    raising.

    The accumulator is ``float64`` regardless of the dtype of ``grid``. That is
    what lets the shipped tables be stored as ``float32`` without changing a
    line here: the weights promote the gathered values on multiplication.

    Examples
    --------
    Sampling ``f(t) = t**2`` on five nodes over ``[0, 4]`` and asking for a
    point that is not a node reproduces the quadratic exactly:

    >>> grid = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> points = np.array([[2.5]])
    >>> np.round(_interpolate(points, grid, np.array([0.0]), np.array([4.0])), 12)
    array([6.25])
    """
    point_shape = np.shape(points)[:-1]
    points = np.reshape(points, (np.multiply.reduce(point_shape), np.shape(points)[-1]))

    grid_shape = np.array(np.shape(grid))
    dims = len(grid_shape)
    points = (points - lower) * ((grid_shape - 1) / (upper - lower))
    floors = np.floor(points).astype('int')

    offsets = points - floors
    offsets2 = offsets * offsets
    offsets3 = offsets2 * offsets
    weighters = [
        -0.5 * offsets3 + offsets2 - 0.5 * offsets,
        1.5 * offsets3 - 2.5 * offsets2 + 1.0,
        -1.5 * offsets3 + 2 * offsets2 + 0.5 * offsets,
        0.5 * offsets3 - 0.5 * offsets2,
    ]

    ravel_grid = np.ravel(grid)

    result = np.zeros(np.shape(points)[:-1], 'float64')
    for i in range(1 << (dims * 2)):
        weights = np.ones(np.shape(points)[:-1], 'float64')
        ravel_offset = 0
        for j in range(dims):
            n = (i >> (j * 2)) % 4
            ravel_offset = ravel_offset * grid_shape[j] + np.maximum(
                0, np.minimum(grid_shape[j] - 1, floors[:, j] + (n - 1)))
            weights *= weighters[n][:, j]

        result += weights * np.take(ravel_grid, ravel_offset)

    return np.reshape(result, point_shape)
