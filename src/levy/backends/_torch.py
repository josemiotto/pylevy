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

"""The torch backend: the same interpolation, written to be differentiable.

This is a reimplementation, not a wrapper. The NumPy code mutates arrays under
boolean masks, which autograd cannot follow; the same arithmetic is expressed
here with ``torch.where``, ``torch.stack`` and out-of-place indexing so that a
gradient reaches every one of ``alpha``, ``beta``, ``mu`` and ``sigma``.

What that buys is a stable distribution you can put inside a larger torch model
and fit by gradient descent along with the rest of it, instead of only through
this package's own L-BFGS-B.

Notes
-----
Two things are deliberately not differentiable, and both are choices rather
than values:

* the grid cell used to look up the tail crossover limits, which is an integer
  index derived from ``alpha`` and ``beta``;
* the mask separating the interpolated region from the power-law tail, which
  follows from those limits.

A gradient with respect to ``alpha`` therefore ignores the fact that moving
``alpha`` far enough would move the crossover. Over any distance a gradient step
cares about, that term is zero anyway -- the crossover is piecewise constant in
``alpha`` -- so the gradient is right almost everywhere and wrong only exactly
at a cell boundary, where a step function has no derivative to begin with.

The lookup tables are stored as ``float32``. They are converted to whatever
dtype the caller's tensors use and cached per (table, dtype, device), so a
``float64`` computation carries ``float64`` weights over a ``float32`` grid --
exactly what the NumPy path does.
"""

import math

import numpy as np

from levy.constants import _lower, _upper
from levy.distribution import _check_alpha_beta, _grid_index
from levy.tables import _read_from_cache

__all__ = ['approximate', 'cdf', 'interpolate', 'name', 'neglog', 'pdf']

#: Identifies this backend in error messages and tests.
name = 'torch'

_table_cache = {}


def _torch():
    """Return the torch module.

    Returns
    -------
    module
        ``torch``. Imported here rather than at module scope so that merely
        importing :mod:`levy.backends` costs nothing.
    """
    import torch

    return torch


def _table(key, like):
    """Return a lookup table as a tensor matching `like`.

    Parameters
    ----------
    key : {'pdf', 'cdf', 'lower_limit', 'upper_limit'}
        Which table.
    like : Tensor
        Tensor whose dtype and device the result should match.

    Returns
    -------
    Tensor
        The table. Cached, and shared between calls, so treat it as read-only.
    """
    torch = _torch()
    dtype = like.dtype if like.dtype.is_floating_point else torch.get_default_dtype()
    signature = (key, dtype, str(like.device))
    cached = _table_cache.get(signature)
    if cached is None:
        cached = torch.as_tensor(
            np.asarray(_read_from_cache(key)), dtype=dtype, device=like.device)
        _table_cache[signature] = cached
    return cached


def _as_tensor(value, like):
    """Bring a parameter into the tensor world without breaking its gradient.

    Parameters
    ----------
    value : Tensor or float
        A parameter.
    like : Tensor
        Tensor whose dtype and device to match.

    Returns
    -------
    Tensor
        `value` itself when it is already a tensor -- untouched, so that any
        ``requires_grad`` on it survives -- otherwise a scalar tensor.
    """
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(dtype=like.dtype, device=like.device) \
            if (value.dtype != like.dtype or value.device != like.device) else value
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def interpolate(points, grid, lower, upper):
    """Multi-dimensional Catmull-Rom interpolation, differentiably.

    Parameters
    ----------
    points : Tensor
        Coordinates to evaluate at, shaped ``(..., ndim)``.
    grid : Tensor
        Samples of the function on a regular lattice.
    lower : array_like
        Coordinate of the first grid node along each dimension.
    upper : array_like
        Coordinate of the last grid node along each dimension.

    Returns
    -------
    Tensor
        Interpolated values, shaped like `points` without its last axis.

    Notes
    -----
    The same 4**ndim weighted gather as :func:`levy.interpolation._interpolate`,
    with indices clamped at the edges. Everything that carries a gradient --
    the offsets, the cubic weights -- is out of place; only the integer indices
    are not, and integers have no gradient to lose.
    """
    torch = _torch()

    point_shape = points.shape[:-1]
    flat = points.reshape(-1, points.shape[-1])

    dims = grid.dim()
    sizes = list(grid.shape)
    scale = torch.as_tensor(
        [(sizes[j] - 1) / (upper[j] - lower[j]) for j in range(dims)],
        dtype=flat.dtype, device=flat.device)
    origin = torch.as_tensor(
        [float(lower[j]) for j in range(dims)], dtype=flat.dtype, device=flat.device)

    scaled = (flat - origin) * scale
    floors = torch.floor(scaled).detach().to(torch.long)

    offsets = scaled - floors.to(flat.dtype)
    offsets2 = offsets * offsets
    offsets3 = offsets2 * offsets
    weighters = [
        -0.5 * offsets3 + offsets2 - 0.5 * offsets,
        1.5 * offsets3 - 2.5 * offsets2 + 1.0,
        -1.5 * offsets3 + 2.0 * offsets2 + 0.5 * offsets,
        0.5 * offsets3 - 0.5 * offsets2,
    ]

    ravel_grid = grid.reshape(-1)
    result = torch.zeros(flat.shape[:-1], dtype=flat.dtype, device=flat.device)

    for i in range(1 << (dims * 2)):
        weights = torch.ones(flat.shape[:-1], dtype=flat.dtype, device=flat.device)
        ravel_offset = torch.zeros(
            flat.shape[:-1], dtype=torch.long, device=flat.device)
        for j in range(dims):
            n = (i >> (j * 2)) % 4
            index = torch.clamp(floors[:, j] + (n - 1), 0, sizes[j] - 1)
            ravel_offset = ravel_offset * sizes[j] + index
            weights = weights * weighters[n][:, j]
        result = result + weights * ravel_grid[ravel_offset]

    return result.reshape(point_shape)


def approximate(x, alpha, beta, cdf=False):
    """Evaluate the power-law tail approximation, differentiably.

    Parameters
    ----------
    x : Tensor
        Standardised values outside the crossover limits.
    alpha : Tensor
        Index of stability.
    beta : Tensor
        Skewness.
    cdf : bool, default False
        Return the distribution function instead of the density.

    Returns
    -------
    Tensor
        The asymptotic pdf or cdf at `x`.

    Notes
    -----
    ``scipy.special.gamma(alpha)`` becomes ``exp(lgamma(alpha))``, which is
    differentiable in ``alpha`` and, for the range this package covers, equal to
    it to within rounding. The in-place masked writes of the NumPy version
    become :func:`torch.where`, which is what autograd can follow.
    """
    torch = _torch()

    positive = x > 0
    magnitude = (torch.sin(math.pi * alpha / 2.0) * torch.exp(torch.lgamma(alpha))
                 / math.pi * torch.abs(x) ** (-alpha - 1.0))
    values = torch.where(positive, magnitude * (1.0 + beta), magnitude * (1.0 - beta))
    if cdf:
        return torch.where(positive, 1.0 - values * x, values * (-x))
    return values * alpha


def _evaluate(x, alpha, beta, mu, sigma, cdf):
    """Evaluate the density or distribution function on tensors.

    Parameters
    ----------
    x : Tensor
        Points to evaluate at.
    alpha, beta, mu, sigma : Tensor or float
        Parameters, in parametrization 0.
    cdf : bool
        Return the distribution function instead of the density.

    Returns
    -------
    Tensor
        The result, shaped like `x`.
    """
    torch = _torch()

    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())

    alpha_t = _as_tensor(alpha, x)
    beta_t = _as_tensor(beta, x)
    mu_t = _as_tensor(mu, x)
    sigma_t = _as_tensor(sigma, x)

    alpha_value = float(alpha_t.detach().reshape(-1)[0]) if alpha_t.dim() \
        else float(alpha_t.detach())
    beta_value = float(beta_t.detach().reshape(-1)[0]) if beta_t.dim() \
        else float(beta_t.detach())
    alpha_value, beta_value = _check_alpha_beta(alpha_value, beta_value)

    table = _table('cdf' if cdf else 'pdf', x)
    lower_limit = _table('lower_limit', x)
    upper_limit = _table('upper_limit', x)

    xr = (x - mu_t) / sigma_t

    # An integer cell, chosen from detached values: see the module docstring on
    # why this is a choice rather than a quantity, and has no gradient.
    alpha_index = _grid_index(alpha_value, 1, lower_limit.shape[0])
    beta_index = _grid_index(beta_value, 2, lower_limit.shape[1])
    low = lower_limit[alpha_index, beta_index]
    up = upper_limit[alpha_index, beta_index]

    inside = (low <= xr) & (xr <= up)
    z = xr[inside]

    points = torch.stack(
        [torch.atan(z), alpha_t.expand_as(z), beta_t.expand_as(z)], dim=-1)
    interpolated = interpolate(points, table, _lower, _upper)
    approximated = approximate(xr[~inside], alpha_t, beta_t, cdf)

    result = torch.zeros_like(xr)
    result = result.masked_scatter(inside, interpolated)
    result = result.masked_scatter(~inside, approximated)
    if not cdf:
        result = result / sigma_t
    return result


def pdf(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the density on tensors.

    Parameters
    ----------
    x : Tensor or array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : Tensor or float
        Parameters, in parametrization 0.

    Returns
    -------
    Tensor
        The density at `x`, carrying gradients from any parameter that has them.
    """
    return _evaluate(x, alpha, beta, mu, sigma, cdf=False)


def cdf(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the distribution function on tensors.

    Parameters
    ----------
    x : Tensor or array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : Tensor or float
        Parameters, in parametrization 0.

    Returns
    -------
    Tensor
        The distribution function at `x`.
    """
    return _evaluate(x, alpha, beta, mu, sigma, cdf=True)


def neglog(x, alpha, beta, mu=0.0, sigma=1.0):
    """Evaluate the negative log density on tensors.

    Parameters
    ----------
    x : Tensor or array_like
        Points to evaluate at.
    alpha, beta, mu, sigma : Tensor or float
        Parameters, in parametrization 0.

    Returns
    -------
    Tensor
        ``-log(pdf(x))``. Summing this over a sample gives an objective that
        ``loss.backward()`` can differentiate.

    Notes
    -----
    The floor at 1e-100 matches the NumPy path. It is applied with
    :func:`torch.clamp`, whose gradient is zero for clamped entries -- the right
    answer, since a density the table reports as non-positive carries no
    information about which way to move.
    """
    torch = _torch()

    return -torch.log(torch.clamp(
        _evaluate(x, alpha, beta, mu, sigma, cdf=False), min=1e-100))
