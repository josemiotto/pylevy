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

"""The typed, validated public API.

Five keyword-only functions -- :func:`pdf`, :func:`cdf`, :func:`logpdf`,
:func:`rvs` and :func:`fit` -- with the names a reader coming from
``scipy.stats`` already knows, and a frozen :class:`StableParams` carrying
validated parameters between them.

The numerical core is untouched; every function here converts, validates once,
and delegates. That is the whole design rule:

**Pydantic sits at the boundary and never enters the hot loop.** A
:class:`StableParams` is built when a call arrives, and once more when
:func:`fit` returns -- one construction per fit, whatever the sample size.
:func:`fit` runs thousands of likelihood evaluations, and not one of them
constructs a model; ``tests/test_hot_loop.py`` counts the constructions during
a 1000-point fit and fails if the number grows with the data.

Examples
--------
>>> import numpy as np
>>> from levy import api
>>> np.round(api.cdf(np.array([1.0, 2.0]), alpha=1.5, beta=0.0), 6)
array([0.756342, 0.89496 ])

Parameters can be carried around as a validated object instead of four loose
floats:

>>> params = api.StableParams(alpha=1.5, beta=0.0)
>>> params
StableParams(alpha=1.5, beta=0.0, mu=0.0, sigma=1.0)
>>> np.round(api.pdf(np.array([1.0]), **params.as_kwargs()), 6)
array([0.202038])

Out-of-range values are rejected where they are given, not deep inside an
interpolation:

>>> try:
...     api.StableParams(alpha=0.2, beta=0.0)
... except Exception as error:
...     print(type(error).__name__)
ValidationError
"""

import operator
from typing import Any, Optional, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from levy._pandas import as_sample, labels_of, relabel
from levy._typing import (
    ArrayLike,
    FloatArray,
    Parametrization,
    ScalarOrArray,
    Seed,
    Size,
)
from levy.constants import par_names
from levy.distribution import levy as _levy
from levy.fitting import fit_levy as _fit_levy
from levy.parametrization import Parameters
from levy.sampling import random as _random

__all__ = ['FitResult', 'StableParams', 'cdf', 'fit', 'logpdf', 'pdf', 'rvs']


def _check_par(par: Any) -> None:
    """Reject a parametrization name the package does not know.

    Parameters
    ----------
    par : object
        The value given for ``par``.

    Raises
    ------
    ValueError
        If `par` is not one of the five names, listing them. Without this an
        unknown name surfaced as a ``KeyError`` from a conversion table
        several frames down, naming neither the argument nor the choices.
    """
    if par not in par_names:
        raise ValueError(
            f'par must be one of {", ".join(repr(name) for name in par_names)}, got {par!r}')


class StableParams(BaseModel):
    """Validated parameters of a stable distribution, in parametrization 0.

    Frozen and hashable, so an instance can be reused, cached or used as a
    dictionary key without any risk of a later mutation invalidating the
    validation it passed.

    Attributes
    ----------
    alpha : float
        Index of stability, in ``[0.5, 2]``. The lower bound is where the
        lookup tables stop, not where the distribution family does.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location, ``delta_0`` in Nolan's notation.
    sigma : float, default 1.0
        Scale, strictly positive. ``gamma`` in Nolan's notation; note that it
        corresponds to ``sqrt(2) * sigma`` of the Normal distribution.

    See Also
    --------
    levy.parametrization.Parameters : The fitting-time wrapper, which also
        tracks which components are held fixed.

    Examples
    --------
    >>> StableParams(alpha=1.5, beta=-0.5, mu=1.0, sigma=2.0)
    StableParams(alpha=1.5, beta=-0.5, mu=1.0, sigma=2.0)

    Values from another parametrization are converted on the way in:

    >>> p = StableParams.from_par(1.6, 0.5, 0.3, 1.2, par='1')
    >>> round(p.mu, 6)
    -0.135926
    """

    # allow_inf_nan=False: sigma=inf satisfies gt=0 and then divides the
    # density into zeros; an infinite or NaN parameter is not a distribution.
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)

    alpha: float = Field(ge=0.5, le=2.0)
    beta: float = Field(ge=-1.0, le=1.0)
    mu: float = 0.0
    sigma: float = Field(default=1.0, gt=0.0)

    @classmethod
    def from_par(
        cls,
        alpha: float,
        beta: float,
        loc: float,
        scale: float,
        par: Parametrization = '0',
    ) -> "StableParams":
        """Build from values written in any of the five parametrizations.

        Parameters
        ----------
        alpha : float
            First parameter, the index of stability in every parametrization.
        beta : float
            Second parameter: the skewness in 0, 1, M and A; ``beta_B`` in B.
        loc : float
            Third parameter: ``mu`` in 0 and 1, ``gamma`` in M, A and B.
        scale : float
            Fourth parameter: ``sigma`` in 0 and 1, ``lambda`` in M, A and B.
        par : {'0', '1', 'M', 'A', 'B'}, default '0'
            Which parametrization the four values are written in.

        Returns
        -------
        StableParams
            The same distribution, in parametrization 0.

        Notes
        -----
        Validation happens *after* the conversion, because parametrization 0 is
        what the lookup tables cover. A ``beta_B`` that is perfectly legal in B
        can convert to a ``beta_0`` outside ``[-1, 1]``; catching that here is
        the point.
        """
        _check_par(par)
        values = np.asarray([alpha, beta, loc, scale], dtype='d')
        if par != '0':
            values = Parameters.convert(values, par, '0')
        return cls(
            alpha=float(values[0]),
            beta=float(values[1]),
            mu=float(values[2]),
            sigma=float(values[3]),
        )

    def to_par(self, par: Parametrization) -> tuple[float, float, float, float]:
        """Write these parameters in another parametrization.

        Parameters
        ----------
        par : {'0', '1', 'M', 'A', 'B'}
            Parametrization to convert to.

        Returns
        -------
        tuple of float
            The four values, in that parametrization's own order and meaning.

        Raises
        ------
        ValueError
            If the conversion is singular for these values -- B at
            ``alpha = 1`` -- and would return a non-finite parameter.

        Examples
        --------
        >>> p = StableParams(alpha=1.5, beta=0.5, mu=0.0, sigma=1.2)
        >>> tuple(round(v, 6) for v in p.to_par('1'))
        (1.5, 0.5, 0.6, 1.2)
        """
        _check_par(par)
        values = np.asarray(self.as_tuple(), dtype='d')
        if par == '0':
            return self.as_tuple()
        # errstate: the singular case below divides 0/0 and x/0 on purpose,
        # and the ValueError is the report -- not a RuntimeWarning as well.
        with np.errstate(divide='ignore', invalid='ignore'):
            converted = Parameters.convert(values, '0', par)
        if not np.all(np.isfinite(converted)):
            # Zolotarev's B is singular at alpha = 1: beta_B is
            # arctan(beta_0 * tan(pi*alpha/2)) / psi(alpha) with both parts
            # degenerate there, so the result is nan or inf. Returning that
            # would hand back a tuple from_par cannot accept.
            raise ValueError(
                f'parametrization {par!r} is singular at alpha={self.alpha}: '
                f'the conversion gives {tuple(converted)}. Use another '
                f'parametrization there, or move alpha off the singular value.'
            )
        return (
            float(converted[0]), float(converted[1]),
            float(converted[2]), float(converted[3]),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the four parameters positionally.

        Returns
        -------
        tuple of float
            ``(alpha, beta, mu, sigma)``.
        """
        return (self.alpha, self.beta, self.mu, self.sigma)

    def as_kwargs(self) -> dict[str, float]:
        """Return the four parameters as keyword arguments for this module.

        Returns
        -------
        dict
            ``{'alpha': ..., 'beta': ..., 'mu': ..., 'sigma': ...}``, ready to
            splat into :func:`pdf`, :func:`cdf`, :func:`logpdf` or :func:`rvs`.
        """
        return {'alpha': self.alpha, 'beta': self.beta,
                'mu': self.mu, 'sigma': self.sigma}


class FitResult(BaseModel):
    """Outcome of a maximum-likelihood fit.

    Attributes
    ----------
    params : StableParams
        The fitted parameters, always in parametrization 0.
    negative_log_likelihood : float
        Value of the objective at the optimum. Lower is a better fit.
    parametrization : {'0', '1', 'M', 'A', 'B'}
        The parametrization the search was carried out in. It changes the
        feasible region and the starting point, so it is worth recording.

    Examples
    --------
    >>> x = rvs(alpha=1.5, beta=0.0, size=200, random_state=0)
    >>> result = fit(x)
    >>> tuple(round(v, 2) for v in result.params.as_tuple())
    (1.52, -0.08, 0.05, 0.99)
    >>> round(result.negative_log_likelihood, 3)
    402.372
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    params: StableParams
    negative_log_likelihood: float
    parametrization: Parametrization = '0'

    def to_series(self, par: Optional[Parametrization] = None) -> Any:
        """Return the fitted parameters as a named pandas Series.

        Parameters
        ----------
        par : {'0', '1', 'M', 'A', 'B'}, optional
            Parametrization to report in. Defaults to the one the fit ran in,
            which is what makes the index labels meaningful -- ``gamma`` and
            ``lambda`` in M, A and B, ``mu`` and ``sigma`` in 0 and 1.

        Returns
        -------
        pandas.Series
            Four values, indexed by parameter name.

        Raises
        ------
        ImportError
            If pandas is not installed, naming the extra that provides it.
        ValueError
            If `par` is B and the fitted ``alpha`` is 1, where that
            parametrization is singular (see :meth:`StableParams.to_par`).
            A fit run in B with ``alpha`` pinned at 1 is the way to get there;
            ask for another parametrization, ``to_series(par='0')`` say.

        Notes
        -----
        The return type is not annotated as ``pandas.Series`` because pandas is
        an optional extra and must not become a type-checking dependency.
        """
        from levy._compat import require

        pandas = require('pandas')
        par = self.parametrization if par is None else par
        return pandas.Series(
            list(self.params.to_par(par)), index=list(par_names[par]), name='levy')

    def as_par(self, par: Parametrization) -> tuple[float, float, float, float]:
        """Write the fitted parameters in another parametrization.

        Parameters
        ----------
        par : {'0', '1', 'M', 'A', 'B'}
            Parametrization to convert to.

        Returns
        -------
        tuple of float
            The four fitted values in that parametrization.

        Raises
        ------
        ValueError
            If `par` is B and the fitted ``alpha`` is 1, where that
            parametrization is singular; see :meth:`StableParams.to_par`.
        """
        return self.params.to_par(par)


def _narrow(value: Any) -> ScalarOrArray:
    """Give the untyped core's return value a declared type.

    Parameters
    ----------
    value : object
        Whatever :mod:`levy.distribution` or :mod:`levy.sampling` returned.

    Returns
    -------
    float or ndarray
        `value` itself if it is an array, otherwise a Python float.

    Notes
    -----
    The numerical core carries no annotations, so mypy sees ``Any`` coming back
    from it. Rather than asserting a type with a bare ``cast``, this checks
    one -- the cost is a single ``isinstance`` per call, not per element, and
    it means the declared return type of every public function is something
    that was actually verified.
    """
    if isinstance(value, np.ndarray):
        return cast(FloatArray, value)
    return float(value)


def _labelled(values: Any, labels: dict[str, Any]) -> ScalarOrArray:
    """Put pandas labels back on a result, and declare the type of the outcome.

    Parameters
    ----------
    values : ndarray
        The computed result.
    labels : dict
        The label description from :func:`levy._pandas.labels_of`.

    Returns
    -------
    Series or DataFrame
        Declared as ``ScalarOrArray``, which is a deliberate and localised
        inaccuracy: what actually comes back is a pandas object. Expressing
        that in the annotation would make pandas a type-checking dependency of
        an *optional* extra. Every non-pandas call -- which is every call a
        type checker will normally see -- really does return float or ndarray.
        The docstrings of :func:`pdf`, :func:`cdf` and :func:`logpdf` say what
        happens with pandas input.
    """
    return cast(ScalarOrArray, relabel(values, labels))


def _validated(
    alpha: float, beta: float, mu: float, sigma: float, par: Parametrization
) -> StableParams:
    """Validate one set of parameters at the API boundary.

    Parameters
    ----------
    alpha, beta, mu, sigma : float
        The four parameters, in `par`.
    par : {'0', '1', 'M', 'A', 'B'}
        Parametrization they are written in.

    Returns
    -------
    StableParams
        The validated parameters, in parametrization 0.
    """
    if par == '0':
        return StableParams(alpha=alpha, beta=beta, mu=mu, sigma=sigma)
    return StableParams.from_par(alpha, beta, mu, sigma, par=par)


def pdf(
    x: ArrayLike,
    *,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    par: Parametrization = '0',
) -> ScalarOrArray:
    """Evaluate the probability density function.

    Parameters
    ----------
    x : array_like or Series or DataFrame
        Points to evaluate at. A scalar input gives a float result; a pandas
        object gives one back, carrying the same index.
    alpha : float
        Index of stability, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location.
    sigma : float, default 1.0
        Scale, strictly positive.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the four parameters are written in.

    Returns
    -------
    float or ndarray, or Series or DataFrame
        The density at `x`; a pandas object when `x` is one, with its index.

    Raises
    ------
    pydantic.ValidationError
        If the parameters, once converted to parametrization 0, fall outside
        what the lookup tables cover.

    See Also
    --------
    cdf : The distribution function.
    logpdf : The log density, which is what fitting actually uses.

    Examples
    --------
    >>> np.round(pdf(np.array([1.0, 2.0]), alpha=1.5, beta=0.0), 6)
    array([0.202038, 0.084539])
    """
    p = _validated(alpha, beta, mu, sigma, par)
    labels = labels_of(x)
    values = _levy(np.asarray(x, dtype='d') if labels else x,
                   p.alpha, p.beta, p.mu, p.sigma, cdf=False)
    return _labelled(values, labels) if labels else _narrow(values)


def cdf(
    x: ArrayLike,
    *,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    par: Parametrization = '0',
) -> ScalarOrArray:
    """Evaluate the cumulative distribution function.

    Parameters
    ----------
    x : array_like or Series or DataFrame
        Points to evaluate at. A scalar input gives a float result; a pandas
        object gives one back, carrying the same index.
    alpha : float
        Index of stability, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location.
    sigma : float, default 1.0
        Scale, strictly positive.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the four parameters are written in.

    Returns
    -------
    float or ndarray, or Series or DataFrame
        The distribution function at `x`; a pandas object when `x` is one,
        with its index.

    Raises
    ------
    pydantic.ValidationError
        If the parameters, once converted to parametrization 0, fall outside
        what the lookup tables cover.

    See Also
    --------
    pdf : The density.

    Examples
    --------
    >>> np.round(cdf(np.array([1.0, 2.0]), alpha=1.5, beta=0.0), 6)
    array([0.756342, 0.89496 ])
    """
    p = _validated(alpha, beta, mu, sigma, par)
    labels = labels_of(x)
    values = _levy(np.asarray(x, dtype='d') if labels else x,
                   p.alpha, p.beta, p.mu, p.sigma, cdf=True)
    return _labelled(values, labels) if labels else _narrow(values)


def logpdf(
    x: ArrayLike,
    *,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    par: Parametrization = '0',
) -> ScalarOrArray:
    """Evaluate the log of the probability density function.

    Parameters
    ----------
    x : array_like or Series or DataFrame
        Points to evaluate at. A pandas object gives one back, carrying the
        same index.
    alpha : float
        Index of stability, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location.
    sigma : float, default 1.0
        Scale, strictly positive.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the four parameters are written in.

    Returns
    -------
    float or ndarray, or Series or DataFrame
        ``log(pdf(x))``; a pandas object when `x` is one, with its index.

    See Also
    --------
    levy.distribution.neglog_levy : The same quantity negated, as the fit uses
        it.

    Notes
    -----
    Note the sign. This returns the log density, following ``scipy.stats``;
    the 1.x function ``levy.neglog_levy`` returns its negative, and both remain
    available. Densities are floored at 1e-100 before the logarithm, so the
    result is bounded below by about -230 rather than being ``-inf``.

    Examples
    --------
    >>> np.round(logpdf(np.array([1.0, 2.0]), alpha=1.5, beta=0.0), 6)
    array([-1.599299, -2.470541])
    """
    p = _validated(alpha, beta, mu, sigma, par)
    labels = labels_of(x)
    values = np.log(np.maximum(1e-100, _levy(
        np.asarray(x, dtype='d') if labels else x,
        p.alpha, p.beta, p.mu, p.sigma, cdf=False)))
    return _labelled(values, labels) if labels else _narrow(values)


def rvs(
    *,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    par: Parametrization = '0',
    size: Size = None,
    random_state: Seed = None,
) -> ScalarOrArray:
    """Draw random variates.

    Parameters
    ----------
    alpha : float
        Index of stability, in ``[0.5, 2]``.
    beta : float
        Skewness, in ``[-1, 1]``.
    mu : float, default 0.0
        Location.
    sigma : float, default 1.0
        Scale, strictly positive.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization the four parameters are written in.
    size : int or tuple of int, optional
        Shape of the result. The default draws a single scalar.
    random_state : int, optional
        Seed for the draw. The global ``numpy.random`` stream is used either
        way; this seeds it, then restores whatever state it had, so calling
        with a seed does not disturb a surrounding sequence.

    Returns
    -------
    float or ndarray
        The variates.

    See Also
    --------
    levy.sampling.random : The 1.x spelling, taking ``shape`` instead of
        ``size``.

    Notes
    -----
    Sampling still runs on the legacy global ``numpy.random`` stream rather
    than a ``Generator``, so that seeded results match 1.x exactly.
    ``random_state`` therefore takes a seed, not a ``Generator``; accepting one
    would mean reimplementing the sampler and changing every seeded number.

    For the same reason a seeded call saves the global stream's state, seeds,
    draws and restores it. That sequence is not synchronised: two threads
    making seeded calls at once can restore each other's stale state, and
    the guarantee that the surrounding stream is left where it was then
    fails. Seeded calls are single-threaded by contract; unseeded ones are
    as thread-safe as ``numpy.random`` itself.

    Examples
    --------
    >>> np.round(rvs(alpha=1.5, beta=0.0, size=4, random_state=0), 6)
    array([0.218654, 0.775259, 0.454604, 0.10272 ])

    The surrounding stream is left where it was:

    >>> np.random.seed(7)
    >>> before = np.random.random()
    >>> np.random.seed(7)
    >>> _ = rvs(alpha=1.5, beta=0.0, size=3, random_state=0)
    >>> bool(np.random.random() == before)
    True
    """
    p = _validated(alpha, beta, mu, sigma, par)
    # np.integer, not just int: a NumPy integer scalar is neither an int
    # subclass nor iterable, so it used to fall through to tuple(size) and
    # raise. `rvs(size=np.int64(10))` is an ordinary thing to write.
    # operator.index, not int(): int(2.5) is 2, so a float shape would be
    # silently truncated, whereas __index__ is defined only by integers --
    # built-in and NumPy alike -- and raises TypeError for anything else.
    shape: tuple[int, ...] = () if size is None else (
        (operator.index(size),) if isinstance(size, (int, np.integer))
        else tuple(operator.index(item) for item in size))

    if random_state is None:
        return _narrow(_random(p.alpha, p.beta, p.mu, p.sigma, shape=shape))

    state = np.random.get_state()
    try:
        np.random.seed(random_state)
        return _narrow(_random(p.alpha, p.beta, p.mu, p.sigma, shape=shape))
    finally:
        np.random.set_state(state)


def fit(
    x: ArrayLike,
    *,
    par: Parametrization = '0',
    **fixed: float,
) -> FitResult:
    """Fit a stable distribution to data by maximum likelihood.

    Parameters
    ----------
    x : array_like or Series or DataFrame
        The sample. A single-column DataFrame is unwrapped; a wider one is
        rejected rather than silently pooled.
    par : {'0', '1', 'M', 'A', 'B'}, default '0'
        Parametrization to search in. It sets the feasible region and the
        starting point, so different choices can reach different optima.
    **fixed
        Any of the names in ``levy.par_names[par]``, pinned to a value instead
        of being estimated.

    Returns
    -------
    FitResult
        The fitted parameters, in parametrization 0, and the negative log
        likelihood at the optimum.

    Raises
    ------
    TypeError
        If a keyword is not a parameter name of `par`. The 1.x
        :func:`levy.fitting.fit_levy` silently ignored such a keyword, so a
        typo cost you an unconstrained fit and no warning.

    See Also
    --------
    levy.fitting.fit_levy : The 1.x spelling, returning a ``Parameters`` object.

    Notes
    -----
    A :class:`StableParams` is constructed twice per call -- never inside the
    optimizer's loop. See the module docstring.

    Examples
    --------
    >>> x = rvs(alpha=1.5, beta=0.0, size=200, random_state=0)
    >>> tuple(round(v, 2) for v in fit(x).params.as_tuple())
    (1.52, -0.08, 0.05, 0.99)

    Pinning a parameter restricts the search:

    >>> tuple(round(v, 2) for v in fit(x, beta=0.0).params.as_tuple())
    (1.53, 0.0, 0.03, 0.99)
    """
    _check_par(par)
    unknown = set(fixed) - set(par_names[par])
    if unknown:
        raise TypeError(
            f'fit() got unexpected keyword argument(s) '
            f'{", ".join(sorted(unknown))}; parametrization {par!r} takes '
            f'{", ".join(par_names[par])}'
        )

    # The names were checked; the values were not. A pinned scale of 0 reaches
    # the optimizer and divides by zero inside levy(), and an out-of-range
    # alpha indexes the lookup tables from the wrong end. These are the same
    # bounds StableParams enforces on the evaluation side; the third name is
    # a location and is unbounded, the fourth is a scale in every
    # parametrization.
    names = par_names[par]
    # Normalised once, and the normalised values are what the fitter gets:
    # checking float(value) and then passing `value` through would let a
    # string or a 0-d array that happens to convert reach the optimizer as
    # it came.
    pinned = {name: float(value) for name, value in fixed.items()}
    # Finite first: inf satisfies `> 0` and nan satisfies nothing, and either
    # would otherwise reach the optimizer, where StableParams would have
    # refused it on the evaluation side.
    for name, value in pinned.items():
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite, got {fixed[name]!r}')
    for name, low, high in ((names[0], 0.5, 2.0), (names[1], -1.0, 1.0)):
        if name in pinned and not low <= pinned[name] <= high:
            raise ValueError(
                f'{name} must lie in [{low}, {high}], got {fixed[name]!r}'
            )
    if names[3] in pinned and not pinned[names[3]] > 0.0:
        raise ValueError(
            f'{names[3]} must be strictly positive, got {fixed[names[3]]!r}'
        )

    parameters, nll = _fit_levy(as_sample(x), par=par, **pinned)
    alpha, beta, mu, sigma = (float(v) for v in parameters.get('0'))
    return FitResult(
        params=StableParams(alpha=alpha, beta=beta, mu=mu, sigma=sigma),
        negative_log_likelihood=float(nll),
        parametrization=par,
    )
