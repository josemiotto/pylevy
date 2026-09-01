"""Case definitions shared by the golden generator and the characterization tests.

Both :mod:`tests.golden.generate` and :mod:`tests.test_characterization` iterate
this single list, so the generator and the assertions can never drift apart.

Each case is a :class:`Case` with a stable ``id`` (the golden-file key), a
``group`` (used to pick the comparison tolerance), and a zero-argument ``run``
that must be deterministic: anything using the global RNG seeds it itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import levy

# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

#: Index of stability. Includes both endpoints of the supported range and the
#: alpha == 1 and alpha == 2 special cases, which take separate code paths.
ALPHAS = (0.5, 0.7, 1.0, 1.3, 1.5, 1.9, 2.0)

#: Skewness, including both endpoints.
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)

#: The tail-crossover limits read from ``lower_limit.npz``/``upper_limit.npz``
#: range over -500..-10 and 10..500, so this spans the interpolated central
#: region *and* both power-law tail regions for every alpha on the grid. That
#: crossover is exactly what PR "fix/grid-index-rounding-and-domain" moves, so
#: it has to be pinned here.
X = np.array(
    [
        -5000.0, -1000.0, -200.0, -60.0, -20.0, -3.0, -1.0, -0.1,
        0.0,
        0.1, 1.0, 3.0, 20.0, 60.0, 200.0, 1000.0, 5000.0,
    ]
)

#: Deliberately *off* the lookup grid.
#:
#: Every value in ALPHAS and BETAS lands on an exact grid node -- alpha maps to
#: (alpha - 0.5) * 50 and beta to (beta + 1) * 50, which are integers for all of
#: them. That is the natural grid to pick and it is also a blind spot: the
#: tail-crossover limits are chosen by snapping (alpha, beta) to a cell, so on
#: an exact node every snapping rule agrees and a change to that rule is
#: invisible. These pairs sit between nodes, where the choice is observable.
#:
#: The last pair is the worst case found by sweeping the grid: the two candidate
#: cells there carry crossovers of 71.30 and 499.80, and at x = 285.55 the
#: interpolated and power-law branches differ by 64%.
OFF_GRID = (
    (0.61, 0.137),      # alpha index 5.5 -- exactly halfway between two cells
    (1.234, -0.409),
    (1.777, 0.6146),
    (0.913, -0.911),
    (1.410182, -0.5),
)

#: x values that straddle the tail crossover for the OFF_GRID pairs.
X_CROSSOVER = np.array(
    [-499.8, -285.55, -71.3, -48.8, -10.2, 10.0, 48.8, 71.3, 285.55, 499.8]
)

#: All five parametrizations: Nolan 0 and 1, Zolotarev M, A and B.
PARS = ("0", "1", "M", "A", "B")

#: Parameter vectors used for conversion round-trips, in parametrization 1.
CONVERT_POINTS = (
    (1.6, 0.5, 0.3, 1.2),
    (0.7, -0.9, -2.0, 0.4),
    (1.99, 1.0, 0.0, 3.0),
    (1.2, 0.0, 1.0, 1.0),
    (0.5, -1.0, 0.0, 2.5),
)


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    run: Callable[[], object] = field(repr=False)
    slow: bool = False


# --------------------------------------------------------------------------
# levy() -- pdf and cdf
# --------------------------------------------------------------------------


def _levy_cases():
    for alpha in ALPHAS:
        for beta in BETAS:
            for cdf in (False, True):
                tag = "cdf" if cdf else "pdf"
                yield Case(
                    id=f"levy/{tag}/a{alpha}/b{beta}",
                    group="levy",
                    run=(lambda a=alpha, b=beta, c=cdf: levy.levy(X, a, b, cdf=c)),
                )

    # Non-default location and scale.
    for mu, sigma in ((0.0, 1.0), (2.5, 0.5), (-1.0, 3.0), (100.0, 0.01)):
        for cdf in (False, True):
            tag = "cdf" if cdf else "pdf"
            yield Case(
                id=f"levy/{tag}/loc_scale/mu{mu}/sigma{sigma}",
                group="levy",
                run=(
                    lambda m=mu, s=sigma, c=cdf: levy.levy(
                        X, 1.5, 0.3, mu=m, sigma=s, cdf=c
                    )
                ),
            )

    # Scalar input returns a Python float; 0-d array input returns an array.
    # Both are part of the de-facto public contract (added in 11bb583).
    yield Case(
        id="levy/pdf/scalar_input",
        group="levy",
        run=lambda: levy.levy(1.0, 1.5, 0.0),
    )
    yield Case(
        id="levy/pdf/zero_d_array_input",
        group="levy",
        run=lambda: levy.levy(np.array(1.0), 1.5, 0.0),
    )
    yield Case(
        id="levy/pdf/2d_input",
        group="levy",
        run=lambda: levy.levy(X.reshape(17, 1) / 2.0, 1.5, 0.25),
    )

    # Off-grid (alpha, beta), where the tail-crossover cell is actually chosen
    # rather than landing exactly on a node. Without these, any change to the
    # snapping rule is invisible to the whole suite.
    for i, (alpha, beta) in enumerate(OFF_GRID):
        for cdf in (False, True):
            tag = "cdf" if cdf else "pdf"
            yield Case(
                id=f"levy/{tag}/off_grid/{i}/a{alpha}/b{beta}",
                group="levy",
                run=(lambda a=alpha, b=beta, c=cdf: levy.levy(X, a, b, cdf=c)),
            )
            yield Case(
                id=f"levy/{tag}/off_grid_crossover/{i}/a{alpha}/b{beta}",
                group="levy",
                run=(
                    lambda a=alpha, b=beta, c=cdf: levy.levy(
                        X_CROSSOVER, a, b, cdf=c
                    )
                ),
            )

    # Known-corrupt CDF cells (alpha=0.58, beta=+-0.74). Pinned deliberately so
    # that the load-time repair in "fix/corrupt-cdf-table-cells" shows up as an
    # explicit, reviewable golden diff rather than a silent change.
    for beta in (-0.74, 0.74):
        yield Case(
            id=f"levy/cdf/corrupt_cells/b{beta}",
            group="levy",
            run=(
                lambda b=beta: levy.levy(
                    np.array([-1.0, 0.0, 1.0]), 0.58, b, cdf=True
                )
            ),
        )


# --------------------------------------------------------------------------
# neglog_levy()
# --------------------------------------------------------------------------


def _neglog_cases():
    for alpha in ALPHAS:
        for beta in BETAS:
            yield Case(
                id=f"neglog_levy/a{alpha}/b{beta}",
                group="levy",
                run=(
                    lambda a=alpha, b=beta: levy.neglog_levy(X, a, b, 0.0, 1.0)
                ),
            )
    yield Case(
        id="neglog_levy/loc_scale",
        group="levy",
        run=lambda: levy.neglog_levy(X, 1.5, -0.4, 2.0, 0.75),
    )
    for i, (alpha, beta) in enumerate(OFF_GRID):
        yield Case(
            id=f"neglog_levy/off_grid/{i}/a{alpha}/b{beta}",
            group="levy",
            run=(
                lambda a=alpha, b=beta: levy.neglog_levy(
                    X_CROSSOVER, a, b, 0.0, 1.0
                )
            ),
        )


# --------------------------------------------------------------------------
# Parameters.convert() -- all 20 ordered pairs
# --------------------------------------------------------------------------


def _convert_cases():
    for i, point in enumerate(CONVERT_POINTS):
        arr = np.array(point)
        for par_in in PARS:
            # Move the point into par_in first so every pair starts from a
            # parametrization-consistent vector rather than reinterpreting the
            # same four numbers five different ways.
            try:
                start = levy.Parameters.convert(arr, "1", par_in)
            except Exception:  # pragma: no cover - defensive
                continue
            for par_out in PARS:
                if par_in == par_out:
                    continue
                yield Case(
                    id=f"convert/p{i}/{par_in}_to_{par_out}",
                    group="convert",
                    run=(
                        lambda s=start, a=par_in, b=par_out: levy.Parameters.convert(
                            s, a, b
                        )
                    ),
                )


def _parameters_cases():
    yield Case(
        id="Parameters/get/par1_to_B",
        group="convert",
        run=lambda: levy.Parameters(
            par="1", alpha=1.5, beta=0.5, mu=0, sigma=1.2
        ).get("B"),
    )
    for par in PARS:
        yield Case(
            id=f"Parameters/defaults/{par}",
            group="convert",
            run=(
                lambda p=par: levy.Parameters(
                    par=p, **{k: None for k in levy.par_names[p]}
                ).get()
            ),
        )


# --------------------------------------------------------------------------
# random() -- exact on the legacy MT19937 stream
# --------------------------------------------------------------------------


def _random_cases():
    combos = (
        (1.5, 0.0, 0.0, 1.0),
        (1.5, 0.5, 0.0, 1.0),
        (0.7, -1.0, 2.0, 0.5),
        # alpha == 1 with beta == 0 is stable: _phi() multiplies tan(pi/2) by
        # zero, so the pole never reaches the sample.
        (1.0, 0.0, 0.0, 1.0),
        # Deliberately NOT covered here: alpha ~ 1 with beta != 0. random()
        # nudges alpha to 1.0 + 1e-15, which lands tan(pi*alpha/2) on its pole
        # (-5.83e+14), where a single ULP moves _phi by 11.4%. The resulting
        # samples are not reproducible across libm builds, so pinning them
        # would be pinning noise. The defect itself is asserted in
        # test_known_bugs.py::test_alpha_1_skewed_sampler_is_well_conditioned.
        (1.9, 1.0, -3.0, 2.0),
        (2.0, 0.0, 0.0, 1.0),  # normal branch; ignores mu/sigma today (bug a)
        (2.0, 0.0, 5.0, 3.0),
    )
    for seed in (0, 12345):
        for alpha, beta, mu, sigma in combos:
            yield Case(
                id=f"random/seed{seed}/a{alpha}/b{beta}/mu{mu}/s{sigma}",
                group="random",
                run=(
                    lambda a=alpha, b=beta, m=mu, s=sigma, sd=seed: (
                        np.random.seed(sd),
                        levy.random(a, b, m, s, shape=(64,)),
                    )[1]
                ),
            )


# --------------------------------------------------------------------------
# fit_levy() -- all five parametrizations (slow)
# --------------------------------------------------------------------------


def _fit_data(seed=0, n=200):
    np.random.seed(seed)
    return levy.random(1.5, 0.0, 0.0, 1.0, shape=(n,))


def _encode_fit(result):
    parameters, neglog = result
    return (
        np.asarray(parameters.get(), dtype=np.float64),
        np.asarray(parameters.get("0"), dtype=np.float64),
        float(neglog),
    )


def _fit_cases():
    # Every parametrization, unconstrained. PR "fix/parametrization-aware-fit-
    # bounds" changes results for par='A'/'B'/'M'; without these cases that PR
    # is unreviewable.
    for par in PARS:
        yield Case(
            id=f"fit_levy/free/par{par}",
            group="fit",
            slow=True,
            run=(lambda p=par: _encode_fit(levy.fit_levy(_fit_data(), par=p))),
        )

    # Fixed-parameter fits, mirroring the four original doctests.
    fixed = (
        ("alpha", {"alpha": 1.5}),
        ("beta", {"beta": 0.0}),
        ("beta_mu", {"beta": 0.0, "mu": 0.0}),
        ("cauchy", {"alpha": 1.0, "beta": 0.0}),
    )
    for name, kwargs in fixed:
        yield Case(
            id=f"fit_levy/fixed/{name}",
            group="fit",
            slow=True,
            run=(
                lambda kw=kwargs: _encode_fit(
                    levy.fit_levy(_fit_data(), par="0", **kw)
                )
            ),
        )

    # A second data seed, so a change that happens to be benign on one sample
    # does not slip through.
    yield Case(
        id="fit_levy/free/par0/seed7",
        group="fit",
        slow=True,
        run=lambda: _encode_fit(levy.fit_levy(_fit_data(seed=7), par="0")),
    )


# --------------------------------------------------------------------------


def iter_cases():
    """Yield every characterization case, in a stable order."""
    yield from _levy_cases()
    yield from _neglog_cases()
    yield from _convert_cases()
    yield from _parameters_cases()
    yield from _random_cases()
    yield from _fit_cases()


def all_cases():
    cases = list(iter_cases())
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise AssertionError(f"duplicate case ids: {sorted(duplicates)}")
    return cases
