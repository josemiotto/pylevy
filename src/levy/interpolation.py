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

"""Multi-dimensional Catmull-Rom interpolation, and bound folding.

Deliberately free of any Levy-specific knowledge: this module knows about
grids and points, nothing about alpha or beta. That is what lets the tests
exercise it against a tiny synthetic grid instead of the 10 MB tables.
"""

import numpy as np

__all__ = ['_interpolate', '_reflect']

def _reflect(x, lower, upper):
    """ Folds a value back inside the bounds, reflecting at each edge.

    The in-bounds case returns x unchanged and bit-identical, which is what
    happens on essentially every call: L-BFGS-B already respects the bounds it
    is given. Out of bounds, the fold is computed in closed form rather than by
    repeated reflection, which used to be an unbounded `while 1:` loop -- with
    the sigma bounds (1e-6, 1e10), reflecting 1e30 needs ~1e20 iterations and
    never returns in practice.
    """
    if lower <= x <= upper:
        return x

    span = upper - lower
    if span < 0:
        raise ValueError(
            "reflection bounds are reversed: lower={}, upper={}".format(lower, upper)
        )
    if span == 0:
        return lower

    offset = (x - lower) % (2.0 * span)
    return lower + (2.0 * span - offset if offset > span else offset)


def _interpolate(points, grid, lower, upper):
    """ Perform multi-dimensional Catmull-Rom cubic interpolation. """
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
            ravel_offset = ravel_offset * grid_shape[j] + np.maximum(0, np.minimum(grid_shape[j] - 1, floors[:, j] +
                                                                                   (n - 1)))
            weights *= weighters[n][:, j]

        result += weights * np.take(ravel_grid, ravel_offset)

    return np.reshape(result, point_shape)
