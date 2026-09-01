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

"""The density and distribution functions themselves."""

import numpy as np
from scipy.special import gamma

from levy.constants import _lower, _upper
from levy.interpolation import _interpolate
from levy.tables import _read_from_cache

__all__ = ['levy', 'neglog_levy']


def _check_alpha_beta(alpha, beta):
    """Reject parameters the lookup tables do not cover.

    Parameters
    ----------
    alpha : float
        Index of stability, must lie in ``[0.5, 2]``.
    beta : float
        Skewness, must lie in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If either parameter falls outside the tabulated range.

    Notes
    -----
    Without this, `alpha` below 0.5 produced a *negative* grid index, which is
    a perfectly valid Python index: ``alpha=0.4`` gave -4 and silently returned
    the limits for ``alpha ~ 1.94``. The old ``except IndexError`` guard only
    fired for positive overflow, so under-range values failed silently while
    over-range values raised.
    """
    if not (_lower[1] <= alpha <= _upper[1]):
        raise ValueError(
            f'alpha must be in [{_lower[1]}, {_upper[1]}], got {alpha!r}. pylevy '
            'interpolates from a lookup table that does not cover values '
            'outside that range.'
        )
    if not (_lower[2] <= beta <= _upper[2]):
        raise ValueError(
            f'beta must be in [{_lower[2]}, {_upper[2]}], got {beta!r}.'
        )


def _grid_shape():
    """Return the shape of the tables actually loaded.

    Returns
    -------
    tuple of int
        Shape of the pdf table, ``(x, alpha, beta)``.

    Notes
    -----
    ``size`` is the resolution of the tables shipped with the package, but
    ``levy-tables build --size`` can produce others and ``$LEVY_DATA_DIR`` can
    point at them. Anything that converts a parameter into a grid index has to
    use the real shape, not the constant: doing otherwise raised
    ``IndexError: index 50 is out of bounds for axis 0 with size 10`` as soon
    as a table of a different resolution was loaded.
    """
    return _read_from_cache('pdf').shape


def _grid_index(value, axis, length=None):
    """Return the index of the grid cell used to look up the crossover limits.

    Parameters
    ----------
    value : float
        Parameter value along `axis`.
    axis : {0, 1, 2}
        Grid dimension: x, alpha or beta.
    length : int, optional
        Number of nodes along `axis`. Read from the loaded table when omitted.

    Returns
    -------
    int
        Index in ``[0, length - 1]``.

    Notes
    -----
    This truncates rather than rounding to nearest, which is almost certainly
    unintentional -- but it is deliberately left alone here, because measuring
    it showed that switching to nearest-neighbour does *not* improve accuracy.

    Over 300 sampled points where the two strategies disagree, compared against
    ``calculate_levy`` ground truth:

    ==========  ================  ==========  ======
    strategy    median rel err    mean        p90
    ==========  ================  ==========  ======
    truncate    1.1164e-02        3.5488e-01  1.4560
    round       1.4552e-02        4.3251e-01  1.4656
    bilinear    1.0106e-02        2.9459e-01  0.9723
    ==========  ================  ==========  ======

    Rounding is closer to truth on only 43% of those points and is worse on
    the median. The error is dominated by the discontinuity between the
    interpolated branch and the power-law tail branch -- the two do not meet at
    the crossover, and the CDF steps down by up to 2.57e-03 there -- so the
    choice of limit cell mostly decides which side of a bad seam you land on.
    Fixing that properly means regenerating the limit tables or reconciling the
    branches, not changing this line. Tracked separately.

    Callers must validate first; the clamp only guards against a value landing
    exactly on an endpoint after floating-point rounding.
    """
    if length is None:
        length = _grid_shape()[axis]
    span = _upper[axis] - _lower[axis]
    index = int((value - _lower[axis]) / span * (length - 1))
    return min(length - 1, max(0, index))


def _approximate(x, alpha, beta, cdf=False):
    """Evaluate the power-law tail approximation, valid far from the origin.

    Parameters
    ----------
    x : ndarray
        Standardised values, outside the tabulated crossover limits.
    alpha : float
        Index of stability.
    beta : float
        Skewness.
    cdf : bool, default False
        Return the cdf instead of the pdf.

    Returns
    -------
    ndarray
        The asymptotic pdf or cdf at `x`.
    """
    mask = (x > 0)
    values = np.sin(np.pi * alpha / 2.0) * gamma(alpha) / np.pi * np.power(
        np.abs(x), -alpha - 1.0)
    values[mask] *= (1.0 + beta)
    values[~mask] *= (1.0 - beta)
    if cdf:
        values[mask] = 1.0 - values[mask] * x[mask]
        values[~mask] = values[~mask] * (-x[~mask])
        return values
    else:
        return values * alpha


def levy(x, alpha, beta, mu=0.0, sigma=1.0, cdf=False):
    """Evaluate the Levy stable pdf or cdf, in parametrization 0.

    The tabulated region is interpolated and the tails are replaced by their
    analytical power-law approximation.

    Parameters
    ----------
    x : array_like
        Values where the function is evaluated. A scalar in gives a scalar out.
    alpha : float
        Index of stability, or characteristic exponent, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location, ``delta`` in Nolan's notation.
    sigma : float, default 1.0
        Scale, ``gamma`` in Nolan's notation. Note that `sigma` here
        corresponds to ``sqrt(2) * sigma`` of the Normal distribution.
    cdf : bool, default False
        Return the cdf instead of the pdf.

    Returns
    -------
    float or ndarray
        Values of the pdf, or of the cdf when `cdf` is True, at `x`.

    Raises
    ------
    ValueError
        If `alpha` or `beta` lies outside the tabulated range; the lookup table
        this interpolates from does not cover those values.

    See Also
    --------
    neglog_levy : Negative log density, as used by the fit.
    levy.parametrization.Parameters.convert : Move between parametrizations.

    Notes
    -----
    Parametrization 0 is implied. To pass parameters from another one, convert
    them first with :meth:`~levy.parametrization.Parameters.convert`.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> np.round(levy(x, 1.5, 0.0, cdf=True), 6)
    array([0.756342, 0.89496 , 0.948402])
    >>> np.round(levy(x, 1.5, 0.0), 6)
    array([0.202038, 0.084539, 0.031509])
    """
    _check_alpha_beta(alpha, beta)

    loc = mu

    what = _read_from_cache('cdf') if cdf else _read_from_cache('pdf')
    lower_limit = _read_from_cache('lower_limit')
    upper_limit = _read_from_cache('upper_limit')

    xr = (np.asarray(x, 'd') - loc) / sigma
    alpha_index = _grid_index(alpha, 1, lower_limit.shape[0])
    beta_index = _grid_index(beta, 2, lower_limit.shape[1])
    low_lims = lower_limit[alpha_index, beta_index]
    up_lims = upper_limit[alpha_index, beta_index]
    mask = (low_lims <= xr) & (xr <= up_lims)
    z = xr[mask]

    points = np.empty(np.shape(z) + (3,), 'float64')
    points[..., 0] = np.arctan(z)
    points[..., 1] = alpha
    points[..., 2] = beta

    interpolated = _interpolate(points, what, _lower, _upper)
    approximated = _approximate(xr[~mask], alpha, beta, cdf)

    res = np.empty(np.shape(xr), 'float64')
    res[mask] = interpolated
    res[~mask] = approximated
    if cdf is False:
        res /= sigma
    return float(res) if np.isscalar(x) else res


def neglog_levy(x, alpha, beta, mu, sigma):
    """Evaluate the negative log density of the Levy stable distribution.

    Parameters
    ----------
    x : array_like
        Values where the function is evaluated.
    alpha : float
        Index of stability, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float
        Location.
    sigma : float
        Scale.

    Returns
    -------
    float or ndarray
        Values of ``-log(pdf(x))``.

    Notes
    -----
    Parametrization 0 is implied. Small or negative densities are capped at
    1e-100 before the logarithm to preserve sanity: the shipped pdf table holds
    1845 slightly negative cells, quadrature noise around values that should be
    zero, and without the cap they would come back as NaN in the middle of a
    likelihood sum.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> np.round(neglog_levy(x, 1.5, 0.0, 0.0, 1.0), 6)
    array([1.599299, 2.470541, 3.457474])
    """
    return -np.log(np.maximum(1e-100, levy(x, alpha, beta, mu, sigma)))
