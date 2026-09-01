# Coding guidelines

For humans and for coding agents alike. These are the conventions this codebase
actually follows; a change that breaks one of them will be caught by CI, so this
file is mostly here to save you the round trip.

## The one rule that matters

**Numbers do not move by accident.**

`tests/golden/golden_v1.jsonl` pins the output of `levy`, `neglog_levy`,
`random` and `fit_levy` at 251 points as exact hex floats. If your change moves
any of them, either it was not supposed to and you have a bug, or it was and you
owe the reviewer evidence — a comparison against `levy._build.calculate_levy`
showing the new values are closer to the truth. See
[CONTRIBUTING.md](CONTRIBUTING.md#the-golden-file).

Never regenerate the golden file to make a test pass. That inverts the entire
point of having it.

## Docstrings

NumPy style, checked two ways in CI:

- `ruff` runs pydocstyle under the numpy convention, so the section grammar is
  enforced;
- `numpydoc lint` checks the docstring against the signature — every parameter
  documented, in order, with a type.

Practically:

- Summary on the first line, after the opening quotes, imperative mood, ending
  in a period.
- Document parameters in signature order with types.
- `Returns` for anything that returns.
- `Raises` for anything that raises deliberately.
- Examples that run. They are executed by `pytest --doctest-modules` and by
  Sphinx's doctest builder, both gated. Round floats (`np.round(..., 6)`) rather
  than pasting 17 digits — the 1.x doctests failed precisely because they pinned
  the last digits of an L-BFGS-B optimum.

## Comments

Explain the reasoning, not the mechanics. Several constants here look arbitrary
and are not: `_ALPHA_1_RADIUS = 1e-8` has fifteen lines above it explaining what
1e-15 did wrong and why every value from 1e-10 to 1e-6 behaves identically. When
you make a numerical choice, write down what you measured.

Conversely, do not narrate the obvious. `# increment i` earns nothing.

## Logging

A library does not write to stdout. `levy._logging` provides a logger with a
`NullHandler`; use it. The only `print()` in the package is in the `levy-tables`
command, which is a CLI and is supposed to print.

## Validation and the hot loop

Pydantic validates **once, at the API boundary**, and never inside a likelihood
evaluation. A fit runs thousands of density evaluations; a model construction in
that loop would make the package slow in the one place it exists to be fast.

`tests/test_hot_loop.py` counts `StableParams` constructions during a fit and
fails if the count grows with the sample size. If you add a code path, keep it
outside the loop.

## Optional dependencies

`pandas` and `torch` are extras. Neither may be imported unless it is being
used, and "is it available" is answered by looking in `sys.modules`
(`levy._compat.loaded`) — not by a `try: import` in a hot path.

`tests/test_no_pandas.py` and `tests/test_no_torch.py` run the whole API in a
subprocess with each blocked at the import system. If you add a third optional
dependency, add the matching pair of test files.

When an optional dependency really is required for something, raise through
`levy._compat.require`, which names the extra: `pip install "pylevy[torch]"`,
not an `ImportError` from three frames down.

## Types

`levy/api.py`, `levy/_typing.py`, `levy/_compat.py` and `levy/_pandas.py` are
checked with `mypy --strict` and shipped with `py.typed`. The numerical core is
not annotated and is deliberately excluded — it is checked by tests instead.

If you add to the typed surface, keep it strict. If you need a value from the
untyped core, narrow it with a checked `isinstance` (see `api._narrow`) rather
than a bare `cast`.

Annotations must work on Python 3.9: PEP 585 builtin generics (`list[str]`) are
fine, `X | Y` unions are not.

## Deprecations

The 1.x names resolve through a PEP 562 module `__getattr__` and warn **on
access, not on import**. `import levy` under `-W error::DeprecationWarning` must
keep succeeding; there is a test for it.

A deprecation message says three things: that the name is deprecated and since
when, when it goes away, and what to use instead.

Nothing is a wrapper — a deprecated name resolves to the same object, so numbers
cannot drift between the old spelling and the new one.

## Tests

- Name them after the claim: `test_random_ignores_loc_scale_at_alpha_2`, not
  `test_random_3`.
- A bug fix comes with a regression test that fails before it.
- State tolerances as constants with a reason. Do not tighten one to the last
  digit you happened to observe on your machine — `libm` differs across
  platforms, and CI will find out before you do.
- Tests may use the 1.x names; that is how the goldens prove the shims return
  identical numbers.

## Commits and pull requests

One purpose per pull request. The body says what changed, what it was verified
against, and whether the goldens moved. Include the numbers — a table of
measured deviations is what makes a numerical change reviewable at all.

If you tried something, measured it, and it was worse, say so and say by how
much. Two changes in this package's history were reverted on measurement, and
recording that is more useful than quietly dropping them.
