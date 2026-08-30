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

"""Type aliases shared by the typed API.

Kept in a module of their own so that :mod:`levy.api` can be checked with
``mypy --strict`` without dragging the untyped numerical core along with it.

Attributes
----------
Parametrization : type alias
    The five parametrization tags, as a ``Literal``. Spelling these out is what
    turns a mistyped ``par='b'`` into an error your editor shows you, rather
    than a ``KeyError`` from inside a dict lookup at runtime.
ArrayLike : type alias
    Anything :func:`numpy.asarray` accepts.
FloatArray : type alias
    A NumPy array of ``float64``.
Size : type alias
    The shape argument of :func:`levy.api.rvs`.

Notes
-----
``FloatArray`` and ``ArrayLike`` resolve to the precise ``numpy.typing`` forms
under a type checker and to plain ``numpy.ndarray`` at run time. That is
deliberate: ``numpy.typing.NDArray`` only appeared in NumPy 1.21, and the
package supports 1.20. A type checker always has a modern NumPy; an installed
copy may not.
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

__all__ = ['ArrayLike', 'FloatArray', 'Parametrization', 'Size']

Parametrization = Literal['0', '1', 'M', 'A', 'B']

Size = Union[int, tuple[int, ...], None]

if TYPE_CHECKING:
    import numpy.typing as npt

    ArrayLike = npt.ArrayLike
    FloatArray = npt.NDArray[np.float64]
else:
    ArrayLike = object
    FloatArray = np.ndarray

#: Result of a scalar-or-array evaluation: a Python float for scalar input.
ScalarOrArray = Union[float, "FloatArray"]

#: A seed for the legacy global NumPy stream, or nothing.
Seed = Optional[int]
