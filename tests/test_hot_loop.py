"""Pydantic must stay at the boundary and out of the likelihood loop.

This is the test that makes the new dependency defensible. A stable fit runs
thousands of density evaluations; if a model were constructed inside that loop,
`pip install pylevy` would have bought a validation library and paid for it in
the one place where this package's whole reason to exist is speed.

The invariant is not "few constructions" but "a number that does not grow with
the data". That is what the scaling test below measures.
"""

from __future__ import annotations

import numpy as np
import pytest

import levy

pytest.importorskip("pydantic")

from levy import api  # noqa: E402


class ConstructionCounter:
    """Count StableParams constructions inside a block."""

    def __init__(self, monkeypatch):
        self.count = 0
        original = api.StableParams.__init__

        def counting_init(model_self, **data):
            self.count += 1
            original(model_self, **data)

        monkeypatch.setattr(api.StableParams, "__init__", counting_init)


def _sample(n, seed=0):
    np.random.seed(seed)
    return levy.random(1.5, 0.0, 0.0, 1.0, shape=(n,))


def test_fit_levy_never_constructs_a_model(monkeypatch):
    # The 1.x path must not acquire a pydantic cost at all: someone who never
    # imports levy.api should not pay for it.
    counter = ConstructionCounter(monkeypatch)
    levy.fit_levy(_sample(1000))
    assert counter.count == 0


def test_levy_and_neglog_levy_never_construct_a_model(monkeypatch):
    counter = ConstructionCounter(monkeypatch)
    x = np.linspace(-10.0, 10.0, 5000)
    levy.levy(x, 1.5, 0.0)
    levy.levy(x, 1.5, 0.0, cdf=True)
    levy.neglog_levy(x, 1.5, 0.0, 0.0, 1.0)
    assert counter.count == 0


def test_api_fit_constructs_a_bounded_number_of_models(monkeypatch):
    counter = ConstructionCounter(monkeypatch)
    api.fit(_sample(1000))
    # One for the result. The FitResult wrapper revalidates nothing, because a
    # BaseModel instance passed for a BaseModel field is accepted as is.
    assert counter.count <= 2, (
        f"{counter.count} constructions for a single fit; pydantic has leaked "
        "into the optimizer's loop"
    )


@pytest.mark.parametrize("n", [100, 1000, 4000])
def test_construction_count_does_not_grow_with_the_data(monkeypatch, n):
    counter = ConstructionCounter(monkeypatch)
    api.fit(_sample(n))
    assert counter.count <= 2


def test_pdf_constructs_exactly_one_model_per_call(monkeypatch):
    counter = ConstructionCounter(monkeypatch)
    api.pdf(np.linspace(-5.0, 5.0, 10000), alpha=1.5, beta=0.0)
    assert counter.count == 1


def test_evaluating_with_preconstructed_params_still_costs_one(monkeypatch):
    # Splatting a StableParams back in re-validates: cheap, and it keeps the
    # functions' contract identical whether or not a model was involved.
    params = api.StableParams(alpha=1.5, beta=0.0)
    counter = ConstructionCounter(monkeypatch)
    api.pdf(np.linspace(-5.0, 5.0, 1000), **params.as_kwargs())
    assert counter.count == 1


def test_importing_levy_does_not_import_pydantic():
    # `levy.api` is resolved lazily through the module __getattr__, so a user
    # of the 1.x surface never pays pydantic's import time.
    import os
    import pathlib
    import subprocess
    import sys

    code = (
        "import sys, numpy, levy; "
        "levy.levy(numpy.array([1.0]), 1.5, 0.0); "
        "levy.fit_levy(numpy.array([0.1, -0.4, 1.2, 0.3, -1.1])); "
        "print('pydantic' in sys.modules)"
    )
    env = dict(os.environ)
    # Works whether the package is installed or only present in src/.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(levy.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=env,
    )
    assert result.stdout.strip() == "False", result.stdout
