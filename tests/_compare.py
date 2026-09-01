"""Comparison tolerances for the characterization suite.

These tolerances are the contract. They are deliberately tight where the
computation is deterministic and loose only where it provably is not, so that a
failure means "the math moved", not "the CPU differs".

Every number below was measured by running the suite on macOS/arm64 and on
Linux/x86_64 (the CI matrix) and taking the worst observed disagreement. An
earlier revision used ``atol=0.0`` and ``rtol=1e-14`` and failed on Linux; the
notes record why.

``levy`` / ``convert``
    ``_interpolate`` is elementwise arithmetic plus ``np.take`` -- no BLAS, no
    reductions -- so the interpolation itself is bit-reproducible. The 1e-13
    slack absorbs platform ``libm`` differences in ``np.tan``, ``np.arctan`` and
    ``scipy.special.gamma``, which the parametrization conversions go through.

    ``atol=1e-15`` is not optional. Several conversions produce a *numerical
    zero* by cancellation: converting ``(1.99, 1.0, 0.0, 3.0)`` from M/A/B back
    to parametrization 1 gives a third component of ``-6.9e-18`` on macOS,
    ``-1.4e-17`` on Linux and exactly ``0.0`` elsewhere. All three mean zero,
    but a purely relative comparison calls that a 100% difference. The floor is
    15 orders of magnitude below the O(1) values everywhere else, so it costs
    nothing where the numbers are real.

``random``
    Seeded through the legacy ``np.random.seed`` MT19937 global stream, whose
    reproducibility is a documented NumPy backwards-compatibility guarantee.
    **The stream is exact; the transform is not.** The Chambers-Mallows-Stuck
    step chains ``tan``, ``log``, ``cos`` and a fractional power, and those
    differ by an ULP or two between libm builds -- measured worst case 1.09e-14
    relative between macOS/arm64 and Linux/x86_64. 1e-12 leaves ~100x margin.

``fit``
    ``scipy.optimize.minimize(method='L-BFGS-B')`` terminates on a gradient
    tolerance, so the optimum itself legitimately moves with the SciPy version,
    the BLAS backend and the CPU. **Never compare fits exactly.** That is
    precisely what broke the package's original doctests: they pinned
    ``402.37150603509247`` and now get ``402.3715060350985``.

    The 1e-3 figure is measured, not guessed. Running every fit case under
    numpy 1.26.4 / scipy 1.11.4 and under numpy 2.5.2 / scipy 1.18.1 gives a
    worst-case relative deviation of 3.87e-05 (``fit_levy/free/parA``), so 1e-3
    leaves roughly 25x margin. It is deliberately the loosest tolerance here;
    a genuine regression in the fitter moves parameters far more than this.
"""

from __future__ import annotations

import numpy as np

TOLERANCES = {
    "levy": dict(rtol=1e-13, atol=1e-15),
    "convert": dict(rtol=1e-13, atol=1e-15),
    "random": dict(rtol=1e-12, atol=1e-15),
    "fit": dict(rtol=1e-3, atol=1e-8),
}

#: Groups that regenerate byte-identically *on a fixed platform and dependency
#: set*. Used only by ``generate.py --strict-bytes``, which is a local
#: convenience: byte-identity does not survive a change of libm, so it is not a
#: CI gate. Confirmed by the CI run that produced these tolerances -- goldens
#: generated on macOS/arm64 are not byte-identical on Linux/x86_64.
DETERMINISTIC_GROUPS = ("levy", "convert", "random")


def assert_matches(group, actual, expected, case_id):
    """Assert ``actual`` matches ``expected`` within the tolerance for ``group``."""
    tol = TOLERANCES[group]
    np.testing.assert_allclose(
        _as_comparable(actual),
        _as_comparable(expected),
        equal_nan=True,
        err_msg=f"characterization case {case_id!r} ({group}) changed",
        **tol,
    )


def matches(group, actual, expected):
    """Non-raising form of :func:`assert_matches`."""
    tol = TOLERANCES[group]
    return np.allclose(
        _as_comparable(actual),
        _as_comparable(expected),
        equal_nan=True,
        **tol,
    )


def _as_comparable(obj):
    """Flatten a result into a float array so tuples compare elementwise."""
    if isinstance(obj, tuple):
        return np.concatenate([np.atleast_1d(np.asarray(o, dtype=np.float64)).ravel() for o in obj])
    return np.asarray(obj, dtype=np.float64)
