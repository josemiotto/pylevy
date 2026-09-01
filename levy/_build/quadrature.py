# -*- encoding: utf-8 -*-
"""Direct numerical evaluation of the Levy stable density, by quadrature.

This is the ground truth the interpolation tables are built from, and the
reference the test suite measures interpolation error against. It is far too
slow for general use -- that is the whole reason the tables exist.

Moved out of ``levy/__init__.py`` unchanged apart from the ``np.Inf`` -> ``np.inf``
fix and the rename; the numerical results are identical.
"""

import numpy as np

from levy import _interpolate, _lower, _phi, _upper

__all__ = ["calculate_levy", "interpolated_levy"]


def calculate_levy(x, alpha, beta, cdf=False):
    """ Levy stable distribution by numerical integration.

    Used to build the lookup tables, and as the reference in accuracy tests.
    Note that to evaluate at a 'true' x the tangent must be applied:
    ``levy(2, 1.5, 0) == calculate_levy(np.tan(2), 1.5, 0)``.

    "0" parametrization as per Nolan. The special case alpha == 1 is handled
    separately; because of an error in the numerical integration, its lower
    limit is 1e-10 rather than 0.

    Known limitation: for alpha = 0.58, beta = -+0.74 and |x| near 0.0079 --
    the two grid points closest to zero -- ``integrate.quad``'s oscillatory
    routine fails and returns ~5.72e+307. Those are exactly the four unusable
    cells in the shipped cdf.npz. The weight passed to quad is sin(x*u) or
    cos(x*u), whose period grows without bound as x approaches zero, which is
    where the routine breaks down. Callers building tables should validate the
    result rather than trusting it; see levy._build.tables.
    """
    from scipy import integrate

    beta = -beta

    if alpha == 1:
        li = 1e-10

        def func_cos(u):
            return np.exp(-u) * np.cos(-beta * 2 / np.pi * u * np.log(u))

        def func_sin(u):
            return np.exp(-u) * np.sin(-beta * 2 / np.pi * u * np.log(u))

    else:
        li = 0

        def func_cos(u):
            ua = u ** alpha
            return np.exp(-ua) * np.cos(_phi(alpha, beta) * (ua - u))

        def func_sin(u):
            ua = u ** alpha
            return np.exp(-ua) * np.sin(_phi(alpha, beta) * (ua - u))

    if cdf:
        # Cumulative density function
        return (
            integrate.quad(
                lambda u: u and func_cos(u) / u or 0.0,
                li, np.inf, weight="sin", wvar=x, limlst=1000)[0]
            + integrate.quad(
                lambda u: u and func_sin(u) / u or 0.0,
                li, np.inf, weight="cos", wvar=x, limlst=1000)[0]
            ) / np.pi + 0.5
    else:
        # Probability density function
        return (
            integrate.quad(
                func_cos, li, np.inf, weight="cos", wvar=x, limlst=1000)[0]
            - integrate.quad(
                func_sin, li, np.inf, weight="sin", wvar=x, limlst=1000)[0]
            ) / np.pi


def interpolated_levy(x, alpha, beta, cdf=False, table=None):
    """ Interpolate the table without replacing the tails.

    ``levy.levy`` splices the power-law asymptote onto the tails; this does not,
    which is what the crossover search needs in order to find where the two
    agree.

    May return slightly negative values: Catmull-Rom weights have negative
    lobes, so even a non-negative grid can interpolate below zero.
    """
    from levy import _read_from_cache

    points = np.empty(np.shape(x) + (3,), 'float64')
    points[..., 0] = np.arctan(x)
    points[..., 1] = alpha
    points[..., 2] = beta

    if table is None:
        table = _read_from_cache('cdf') if cdf else _read_from_cache('pdf')
    return _interpolate(points, table, _lower, _upper)
