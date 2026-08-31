# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - unreleased

The package could not be built on Python 3.12, had no tests and no CI, and
carried several defects that returned wrong numbers silently. 2.0 fixes that
and adds a typed, validated API. **No 1.x call changes its result**: every
public name still works and still returns the same floats, bit for bit.

### Added

- `levy.api` — a typed, keyword-only API: `pdf`, `cdf`, `logpdf`, `rvs`, `fit`,
  with a frozen `StableParams` carrying validated parameters. Out-of-range
  values are rejected where they are written, naming the field. Pydantic is used
  only at this boundary; `tests/test_hot_loop.py` counts model constructions
  during a fit and fails if the number grows with the data.
- `py.typed`, so the annotations are visible to a downstream type checker.
  `mypy --strict` runs in CI over the typed surface.
- A characterization test suite: 251 golden records, stored as exact hex floats,
  covering `levy`, `neglog_levy`, `random` and `fit_levy` in all five
  parametrizations. Every change since is measured against it.
- CI on Linux × Python 3.9–3.13, plus a NumPy 1.x leg, a table-generation leg, a
  golden-reproducibility leg, a lint/docstring leg and a doctest leg.
- `levy-tables`, a console script that regenerates the lookup tables into a user
  cache directory, with a `manifest.json` recording grid size, library versions
  and per-array SHA256.
- `levy/__main__.py`, so the documented `python -m levy build` works. It never
  did: for a package, `-m` requires `__main__.py`.
- NumPy-style docstrings throughout, enforced by `ruff` (pydocstyle, numpy
  convention) and `numpydoc`, both gated in CI.
- Documentation that builds from a checkout, published to GitHub Pages and
  built on every pull request with `-W`, so a broken cross-reference fails
  where it was introduced. New narrative pages: **How it works** (the tan-space
  grid, Catmull-Rom interpolation, the tail crossover and its known
  discontinuity, why float32 suffices) and **Migrating from 1.x**. A committed
  `.readthedocs.yaml` replaces build settings that existed only in the Read the
  Docs web interface, where nobody but the account owner could see them.
- An optional `pandas` extra. `pdf`, `cdf` and `logpdf` accept a `Series` or a
  `DataFrame` and return one carrying the same index; `fit` accepts a `Series`
  or a single-column `DataFrame`, and `FitResult.to_series()` reports the
  parameters under that parametrization's own names. The core never sees a
  pandas object, and an install without the extra never imports pandas --
  support is detected by looking in `sys.modules`, so the fast path is one
  dictionary lookup. `tests/test_no_pandas.py` runs the whole API in a
  subprocess with pandas blocked at the import system.
- An optional `torch` extra: `levy.backends`, with a torch implementation of
  the interpolation written to be differentiable. Hand `pdf`, `cdf` or `logpdf`
  a tensor -- or select the backend with `levy.set_backend('torch')` or
  `levy.using('torch')` -- and gradients flow to `alpha`, `beta`, `mu` and
  `sigma`, so a stable distribution can sit inside a larger torch model and be
  fitted by gradient descent. `torch.autograd.gradcheck` passes for all four
  parameters, for pdf, cdf and the negative log density, in float64. NumPy
  remains the default, and an install without the extra never imports torch.

### Changed

- Packaging moved from `distutils` to PEP 621 `pyproject.toml`. The package
  builds and installs on 3.9–3.13 again; it could not be built on 3.12+ at all.
- The 800-line `__init__.py` was split by concern into `constants`,
  `interpolation`, `tables`, `parametrization`, `distribution`, `fitting`,
  `sampling` and `_build`, under a `src/` layout. Verified as a pure move:
  99,312 values bit-identical across wheels built before and after.
- The lookup tables are stored as `float32` and the two crossover-limit tables
  are merged: **24.7 MB → 10.3 MB**. Measured cost, worst case 1.7e-07 relative
  — three orders of magnitude below the interpolation error that already
  dominates.
- `print()` on error paths replaced with a module logger carrying a
  `NullHandler`.

### Deprecated

Every name below still works and still returns exactly what it returned in 1.1.
Accessing one through `levy.` emits a `DeprecationWarning` naming its
replacement; the warning fires on *access*, not on `import levy`. They are
scheduled for removal in 3.0.

| 1.x | 2.0 | note |
|---|---|---|
| `levy.levy(x, a, b, cdf=False)` | `levy.api.pdf(x, alpha=a, beta=b)` | |
| `levy.levy(x, a, b, cdf=True)` | `levy.api.cdf(x, alpha=a, beta=b)` | the `cdf=` flag is gone |
| `levy.neglog_levy(...)` | `levy.api.logpdf(...)` | **opposite sign** — `logpdf` returns `log(pdf)` |
| `levy.fit_levy(x)` | `levy.api.fit(x)` | returns a `FitResult`; rejects a misspelt parameter name |
| `levy.random(..., shape=)` | `levy.api.rvs(..., size=)` | |
| `levy.Parameters` | `levy.api.StableParams` | or `levy.parametrization.Parameters` for the fitting wrapper |
| `levy.convert_to_par0` | `levy.api.StableParams.from_par` | validates the result |
| `levy.convert_from_par0` | `levy.api.StableParams.to_par` | |
| `levy.size` | `levy.constants.size` | |
| `levy.par_bounds` | `levy.constants.par_bounds` | |
| `levy.par_names` | `levy.constants.par_names` | |
| `levy.default` | `levy.constants.default` | |
| `levy.f_bounds` | `levy.constants.f_bounds` | |

Each 1.x name also remains importable from its own module without a warning, if
you want the old behaviour and no deprecation noise.

### Fixed

- `random(alpha=2.0, mu=..., sigma=...)` ignored `mu` and `sigma`. The Gaussian
  branch returned before reaching the line that applied them, so
  `random(2.0, 0.0, mu=100, sigma=5)` came back centred on zero.
- `random(1.0, ±1.0)` produced **NaN for about 0.9% of draws**, and the
  surviving samples were from the wrong distribution (Kolmogorov–Smirnov against
  this package's own CDF: p = 3e-07 over 200k draws). The α = 1 nudge sat 1e-15
  from the pole of the tangent, where rounding the argument becomes an ~11%
  error in the result. Moved to 1e-8, in the middle of a plateau where every
  radius from 1e-10 to 1e-6 behaves identically.
- `alpha` below 0.5 produced a *negative* grid index — a perfectly valid Python
  index — and silently returned values for `alpha ≈ 1.94`. `alpha` and `beta`
  are now validated against the tabulated range.
- Four cells of the shipped `cdf.npz` held `5.72e+307` instead of a probability:
  quadrature failures baked into the table. `levy(0.0, 0.58, 0.74, cdf=True)`
  returned `6.44e+307`. They are repaired at load time by interpolating along x,
  logged once at `WARNING`.
- `Parameters.x = [...]` raised `UnboundLocalError` instead of `TypeError`; the
  setter dispatched on `__class__.__name__`.
- `_reflect` was an unbounded `while 1:` loop. With the σ bounds `(1e-6, 1e10)`,
  folding `1e30` needs ~1e20 iterations and never returns. Replaced by the
  closed-form fold, identical output in the covered range.
- `np.Inf` (removed in NumPy 2.0) made table generation fail on modern NumPy.
- `levy.size` was hardcoded into the index arithmetic, so tables at any other
  resolution raised `IndexError` — which defeated the point of `--size`.
- The doctests: 4 of the 21 had been failing, and CI now runs them as a gate.

### Known issues

- The CDF is discontinuous at the tail crossover, stepping down by up to
  2.6e-03. Fixing it means regenerating the crossover limits or reconciling the
  interpolated and asymptotic branches; tracked separately.
- `par_bounds` uses parametrization-0 semantics for all five parametrizations,
  so a fit in `'B'` searches the wrong feasible region. Tracked separately.

## Versioning

`levy.__version__` is the single source of truth, and tags are `vX.Y.Z` from
here on. The existing tags `v0.5`, `1.1` and `1.2` are inconsistent with that
and are left as they are — note that `1.2` was tagged while `__version__` still
read `"1.1"`. Release automation asserts that a tag matches `__version__` before
publishing.

## [1.1] - 2020-08

The last release of the 1.x line. See the git history; there was no changelog.
