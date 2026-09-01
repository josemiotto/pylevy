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

"""Random variate generation."""

import numpy as np

from levy.parametrization import _phi

__all__ = ['random']

# The Chambers-Mallows-Stuck sampler divides by (1 - alpha) implicitly, through
# phi = beta * tan(pi * alpha / 2), so alpha exactly 1 has to be nudged off the
# pole of the tangent.
#
# The nudge used to be 1e-15, which is far too close: tan(pi*alpha/2) is then
# evaluated 1e-15 from its pole, where the unavoidable ~1e-16 rounding of the
# argument becomes a ~11% relative error in the result. That is not cosmetic.
# At beta = +-1 it made the base of the fractional power below go negative for
# about 0.9% of draws, producing NaN, and the samples that did survive were
# measurably from the wrong distribution (Kolmogorov-Smirnov against this
# package's own CDF: p = 3e-07 over 200k draws).
#
# 1e-8 keeps the same limiting value -- beta*tan(pi*alpha/2)*sin((1-alpha)*b*pi)
# tends to 2*beta*b whatever the radius -- while evaluating it accurately. It
# sits in the middle of a wide plateau: every radius from 1e-10 to 1e-6 gives
# NaN-free samples and KS p ~ 0.50, so this is not a tuned constant. The
# distributional cost of shifting alpha by 1e-8 is far below sampling noise.
_ALPHA_1_RADIUS = 1e-8


def random(alpha, beta, mu=0.0, sigma=1.0, shape=()):
    """Draw random values from an alpha-stable distribution, in parametrization 0.

    Parameters
    ----------
    alpha : float
        Index of stability.
    beta : float
        Skewness.
    mu : float, default 0.0
        Location.
    sigma : float, default 1.0
        Scale.
    shape : tuple of int, default ()
        Shape of the resulting array. The default draws a single scalar.

    Returns
    -------
    float or ndarray
        Generated random values.

    See Also
    --------
    levy.parametrization.Parameters.convert : Move between parametrizations.

    Notes
    -----
    Exact, in the sense of being derived directly from the definition of a
    stable variable [CMS1976]_ rather than by inverting an interpolated cdf.
    Draws come from the legacy global ``numpy.random`` stream, so
    ``np.random.seed`` is what makes a run reproducible.

    ``alpha`` within 1e-8 of 1 is snapped to ``1 + 1e-8``, off the pole of the
    tangent; the module-level comment on ``_ALPHA_1_RADIUS`` explains why.

    References
    ----------
    .. [CMS1976] J. M. Chambers, C. L. Mallows and B. W. Stuck, "A Method for
       Simulating Stable Random Variables", Journal of the American Statistical
       Association 71(354), 340-344, 1976.

    Examples
    --------
    >>> np.random.seed(0)
    >>> np.round(random(1.5, 0.0, shape=(4,)), 6)   # parametrization 0 is implicit
    array([0.218654, 0.775259, 0.454604, 0.10272 ])

    Parameters given in another parametrization have to be converted first:

    >>> from levy.parametrization import Parameters
    >>> par = np.array([1.5, 0.905, 0.707, 1.414])
    >>> rnd = random(*Parameters.convert(par, 'B', '0'), shape=(100,))
    >>> rnd.shape
    (100,)
    """
    if alpha == 2:
        # mu and sigma have to be applied here too. This branch used to return
        # before reaching the `return mu + sigma * k` at the end of the
        # function, so random(2.0, 0.0, mu=100, sigma=5) came back centred on
        # zero with unit-ish scale.
        return mu + sigma * np.random.standard_normal(shape) * np.sqrt(2.0)

    # copysign, not a bare +: nudging an alpha just *below* 1 up to
    # 1 + radius would hand the sampler the opposite side of the pole from
    # the one the caller asked for, and flip the sign of (1 - alpha). alpha
    # exactly 1.0 still goes up, since 1.0 - 1.0 is +0.0.
    if np.absolute(alpha - 1.0) < _ALPHA_1_RADIUS:
        alpha = 1.0 + np.copysign(_ALPHA_1_RADIUS, alpha - 1.0)

    # Chambers, Mallows & Stuck (1976). Two uniforms are transformed into a
    # standard stable variate; mu and sigma are applied at the end.
    #
    # These were a..k before. The algebra is unchanged, only the names.
    uniform_angle = np.random.random(shape)          # was r1
    uniform_exponential = np.random.random(shape)    # was r2
    pi = np.pi

    one_minus_alpha = 1.0 - alpha                    # was a
    centred_uniform = uniform_angle - 0.5            # was b, in [-1/2, 1/2)
    angle = one_minus_alpha * centred_uniform * pi   # was c
    skew_term = _phi(alpha, beta)                    # was e; beta*tan(pi*alpha/2)

    # was f
    scale_factor = (
        -(np.cos(angle) + skew_term * np.sin(angle))
        / (np.log(uniform_exponential) * np.cos(centred_uniform * pi))
    ) ** (one_minus_alpha / alpha)

    tan_half_turn = np.tan(pi * centred_uniform / 2.0)   # was g
    tan_half_angle = np.tan(angle / 2.0)                 # was h
    one_minus_tan_squared = 1.0 - tan_half_turn ** 2.0   # was i

    # was j
    numerator = scale_factor * (
        2.0 * (tan_half_turn - tan_half_angle) * (tan_half_turn * tan_half_angle + 1.0)
        - (tan_half_angle * one_minus_tan_squared - 2.0 * tan_half_turn)
        * skew_term * 2.0 * tan_half_angle
    )

    # was k
    standard_sample = (
        numerator / (one_minus_tan_squared * (tan_half_angle ** 2.0 + 1.0))
        + skew_term * (scale_factor - 1.0)
    )

    return mu + sigma * standard_sample
