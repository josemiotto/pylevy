# Contributing

Thanks for looking. This is a small numerical package, and the thing that makes
it maintainable is that every change can be checked. Most of what follows is
about how to check yours.

## Getting set up

[uv](https://docs.astral.sh/uv/) is the recommended toolchain — it resolves and
installs in seconds, which matters when the optional extras include torch:

```console
uv venv
uv pip install -e ".[dev,lint,docs]"
uv run pytest
```

Plain pip works identically:

```console
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,lint,docs]"
pytest
```

There is deliberately **no `uv.lock` in the repository**. This is a library, not
an application: its dependencies are declared as floors (`numpy>=1.20`) rather
than pins, and the useful thing to test is that it works across that range —
which the CI matrix does — not that it works against one frozen resolution. A
lock file would add churn and hide exactly the breakage it looks like it
prevents.

## Running the checks

```console
pytest                          # the suite, minus the slow table build
pytest -m build                 # regenerates tiny tables by quadrature, ~70s
pytest --doctest-modules src/levy
ruff check .
mypy
numpydoc lint src/levy/*.py src/levy/_build/*.py src/levy/backends/*.py
sphinx-build -b html -W docs/source docs/_build/html
```

`.pre-commit-config.yaml` runs the fast ones on commit:

```console
pre-commit install
```

## The golden file

`tests/golden/golden_v1.jsonl` holds 251 records pinning the numerical output of
`levy`, `neglog_levy`, `random` and `fit_levy`, stored as exact hex floats so
that a change to them shows up as a readable diff rather than a binary blob.

**It is the centre of everything.** Every pull request either leaves it
untouched, or changes it deliberately and says why.

If your change moves a golden record, the pull request has to answer three
questions in its body:

1. **Which records moved, and by how much?** A per-case maximum relative change.
2. **Why is the new value better?** Not "the test now passes" — a comparison
   against `levy._build.calculate_levy`, the quadrature ground truth, showing
   the new values are closer to it.
3. **Was the move intended?** If the answer is "no, but the new values look
   fine", stop and find out what actually changed.

Regenerate with:

```console
python tests/golden/generate.py          # rewrite
python tests/golden/generate.py --check  # verify without writing
```

A pull request that regenerates the goldens without that justification should be
rejected, however green CI is. That discipline is the only thing standing
between this package and a silent numerical regression — which is precisely what
happened before: the last substantive commit of the 1.x line changed the
numerics and regenerated both lookup tables with nothing protecting it.

## Proving a refactor changed nothing

For changes that move code without intending to move numbers:

```console
python scripts/compare_installs.py /path/to/venv-before/bin/python \
                                  /path/to/venv-after/bin/python
```

It builds the same 99,312 values in both interpreters and diffs them as exact
hex floats. "All values match bit for bit" is a much stronger claim than a green
test suite, and it is cheap to produce.

## Coding guidelines

See [AGENTS.md](AGENTS.md). Briefly:

- NumPy-style docstrings, enforced by `ruff` and `numpydoc`.
- No `print()` in library code. The package has a logger.
- Pydantic validates at the API boundary and never inside a likelihood
  evaluation. `tests/test_hot_loop.py` enforces this by counting.
- Optional dependencies are never imported unless they are used.
- A comment explaining *why* is worth more than one explaining *what*. Several
  of the constants in this package look arbitrary and are not; where that is
  true, the reasoning is written down next to them.

## Pull requests

Small and single-purpose, please. The 2.0 work was delivered as a stack of
focused pull requests rather than one large one, and that is the shape that gets
reviewed.

A good pull request body says what changed, what it was verified against, and
whether the goldens moved. If it fixes a bug, it comes with a regression test
that fails before and passes after.

## Reporting a bug

Numerical bugs need a reproducer with concrete numbers: the parameters, the
input, what came out, and what you expected. `levy._build.calculate_levy` is the
ground truth if you need something to compare against — slow, but independent of
the interpolation tables.
