"""Shared pytest configuration and fixtures.

Note that the whole suite runs in about a second: nothing here downloads data,
and the property/interpolation tests build tiny synthetic grids rather than
touching the 12 MB tables shipped in the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Import the package under test from src/ (the layout keeps the working tree
# from shadowing an installed copy), and let the test helper modules
# (_cases, _compare, _encode) be imported by name.
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

sys.path.insert(0, str(TESTS_DIR / "golden"))


@pytest.fixture(scope="session")
def golden():
    """The committed golden records, keyed by case id."""
    import generate

    return generate.load()


@pytest.fixture
def tiny_grid():
    """Build a small analytic grid for exercising the interpolator directly.

    Returns ``(grid, lower, upper, f)`` where ``f(x, y, z)`` is the closed-form
    function the grid samples. Shape is (20, 8, 11) -- three orders of magnitude
    smaller than the shipped (200, 76, 101) tables, so interpolation behaviour
    can be tested without loading 12 MB from disk.

    ``f`` is quadratic on purpose. Catmull-Rom takes its tangents from centred
    differences, which are exact for a quadratic but not for a cubic, so it
    reproduces quadratics to machine precision (measured 1.3e-15) while a cubic
    is off by ~8.8e-04. Only the quadratic is a true oracle.
    """
    lower = np.array([-1.0, 0.5, -1.0])
    upper = np.array([1.0, 2.0, 1.0])
    shape = (20, 8, 11)

    def f(x, y, z):
        return x**2 - 2.0 * x + 0.5 * y**2 + 2.0 * z**2 + 0.25 * z + 1.0

    axes = [np.linspace(lower[i], upper[i], shape[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return f(gx, gy, gz), lower, upper, f


@pytest.fixture
def seeded():
    """Seed the legacy global RNG that levy.random() uses."""

    def _seed(value=0):
        np.random.seed(value)

    return _seed
