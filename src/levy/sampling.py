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

"""Random variate generation."""

import numpy as np

from levy.parametrization import _phi

__all__ = ['random']

#: How far random() moves alpha away from 1 before sampling a skewed draw,
#: to keep tan(pi*alpha/2) off its pole. Module level so the tests assert
#: against the value the sampler actually uses instead of a copy of it; the
#: reasoning for the size of it is at the use site in random().
_ALPHA_1_RADIUS = 1e-8

def random(alpha, beta, mu=0.0, sigma=1.0, shape=()):
    """
    Generate random values sampled from an alpha-stable distribution.
    Notice that this method is "exact", in the sense that is derived
    directly from the definition of stable variable.
    It uses parametrization 0 (to get it from another parametrization, convert).

    Example:
        >>> rnd = random(1.5, 0, shape=(100,))  # parametrization 0 is implicit
        >>> par = np.array([1.5, 0.905, 0.707, 1.414])
        >>> rnd = random(*Parameters.convert(par ,'B' ,'0'), shape=(100,))  # example with convert

    :param alpha: alpha
    :type alpha: float
    :param beta: beta
    :type beta: float
    :param mu: mu
    :type mu: float
    :param sigma: sigma
    :type sigma: float
    :param shape: shape (numpy array type) of the resulting array
    :type shape: tuple
    :return: generated random values
    :rtype: :class:`~numpy.ndarray`
    """

    if alpha == 2:
        # mu and sigma have to be applied here too. This branch used to return
        # before reaching the `return mu + sigma * k` at the end of the
        # function, so random(2.0, 0.0, mu=100, sigma=5) came back centred on
        # zero with unit-ish scale.
        return mu + sigma * np.random.standard_normal(shape) * np.sqrt(2.0)

    # The sampler below divides by (1 - alpha) implicitly, through
    # phi = beta * tan(pi * alpha / 2), so alpha exactly 1 has to be nudged off
    # the pole of the tangent.
    #
    # The nudge used to be 1e-15, which is far too close: tan(pi*alpha/2) is
    # then evaluated 1e-15 from its pole, where the unavoidable ~1e-16 rounding
    # of the argument becomes a ~11% relative error in the result. That is not
    # cosmetic. At beta = +-1 it made the base of the fractional power below go
    # negative for about 0.9% of draws, producing NaN, and the samples that did
    # survive were measurably from the wrong distribution (Kolmogorov-Smirnov
    # against this package's own CDF: p = 3e-07 over 200k draws).
    #
    # 1e-8 keeps the same limiting value -- beta*tan(pi*alpha/2)*sin((1-alpha)*b*pi)
    # tends to 2*beta*b whatever the radius -- while evaluating it accurately.
    # It sits in the middle of a wide plateau: every radius from 1e-10 to 1e-6
    # gives NaN-free samples and KS p ~ 0.50, so this is not a tuned constant.
    # The distributional cost of shifting alpha by 1e-8 is far below sampling
    # noise.
    # copysign, not a bare +: nudging an alpha just *below* 1 up to
    # 1 + radius would hand the sampler the opposite side of the pole from
    # the one the caller asked for, and flip the sign of (1 - alpha). alpha
    # exactly 1.0 still goes up, since 1.0 - 1.0 is +0.0.
    if np.absolute(alpha - 1.0) < _ALPHA_1_RADIUS:
        alpha = 1.0 + np.copysign(_ALPHA_1_RADIUS, alpha - 1.0)

    r1 = np.random.random(shape)
    r2 = np.random.random(shape)
    pi = np.pi

    a = 1.0 - alpha
    b = r1 - 0.5
    c = a * b * pi
    e = _phi(alpha, beta)
    f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
    g = np.tan(pi * b / 2.0)
    h = np.tan(c / 2.0)
    i = 1.0 - g ** 2.0
    j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
    k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)

    return mu + sigma * k
