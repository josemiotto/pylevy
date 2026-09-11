# 📈 pylevy

Lévy alpha-stable distributions for Python: density, distribution function, sampling, and maximum-likelihood fitting. Stable distributions are the heavy-tailed generalisation of the normal, and computing their density directly means a slow numerical integration for every point. pylevy interpolates a precomputed table instead, which is what makes fitting them by maximum likelihood fast enough to be practical.

[![CI](https://github.com/josemiotto/pylevy/actions/workflows/ci.yml/badge.svg)](https://github.com/josemiotto/pylevy/actions/workflows/ci.yml)
[![Docs](https://github.com/josemiotto/pylevy/actions/workflows/docs.yml/badge.svg)](https://github.com/josemiotto/pylevy/actions/workflows/docs.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## ✨ Features

- 📐 **Density, CDF and log density** for `alpha` in `[0.5, 2]` and any skewness `beta` in `[-1, 1]`
- 🎲 **Exact sampling** by the Chambers–Mallows–Stuck transform, no table involved
- 🎯 **Maximum-likelihood fitting** with L-BFGS-B, any subset of parameters pinned, from a start derived from your data
- 🔁 **Five parametrizations**: Nolan's 0 and 1, Zolotarev's M, A and B, converted for you
- 🛡️ **Validated at the boundary**: a bad parameter raises where you wrote it, instead of returning a wrong number
- 🐼 **pandas in, pandas out**: a `Series` or `DataFrame` comes back with its index (optional extra)
- 🔥 **Differentiable torch backend**: the same numbers, with gradients, for fitting inside a larger model (optional extra)
- ⚡ **Fast**: a fit is thousands of density evaluations, and not one of them touches the integrator
- 🔬 **Pinned numerics**: 280 golden records of exact hex floats guard every output across Linux, macOS and Windows
- 🧰 **Rebuildable tables**: `levy-tables` regenerates the lookup tables at any resolution, into a cache, never into the install

## 🚀 Quick Start

### Installation

> **Note:** the `pylevy` name on PyPI belongs to the 2005 package this one descends from, so `pip install pylevy` does **not** install this. Install from a clone until the name is settled ([docs/proposals/pypi-name.md](https://github.com/josemiotto/pylevy/blob/master/docs/proposals/pypi-name.md)).

```bash
git clone https://github.com/josemiotto/pylevy.git
cd pylevy
pip install .
```

Optional extras:

```bash
pip install ".[pandas]"          # labelled input and output
pip install ".[torch]"           # differentiable backend
pip install ".[pandas,torch]"    # both
```

Requires Python 3.9 or newer, NumPy, SciPy and pydantic. pandas and torch are never imported unless you install the extra and use it.

### Thirty seconds

```python
import numpy as np
from levy import api

x = np.array([-1.0, 0.0, 1.0])

api.pdf(x, alpha=1.5, beta=0.0)      # array([0.202038, 0.287353, 0.202038])
api.cdf(x, alpha=1.5, beta=0.0)      # array([0.243658, 0.5     , 0.756342])

sample = api.rvs(alpha=1.5, beta=0.0, size=1000, random_state=0)
result = api.fit(sample)
result.params                        # StableParams(alpha=1.44, beta=0.098, mu=-0.072, sigma=0.962)
result.negative_log_likelihood       # 2066.164
```

## 📖 Usage

### Evaluating the distribution

```python
api.pdf(x, alpha=1.7, beta=0.3, mu=0.5, sigma=2.0)      # density
api.cdf(x, alpha=1.7, beta=0.3, mu=0.5, sigma=2.0)      # distribution function
api.logpdf(x, alpha=1.7, beta=0.3, mu=0.5, sigma=2.0)   # log density, floored so it is never -inf
```

All four parameters are keyword-only. `mu` defaults to 0 and `sigma` to 1.

### Sampling

```python
api.rvs(alpha=1.5, beta=0.0, size=1000)                    # shape can be an int or a tuple
api.rvs(alpha=1.5, beta=0.0, size=(10, 100), random_state=0)
```

Seeded draws are reproducible and leave the surrounding NumPy random stream exactly where it was.

### Fitting

```python
result = api.fit(sample)                       # everything free
result = api.fit(sample, beta=0.0)             # symmetric: beta pinned
result = api.fit(sample, alpha=1.0, beta=0.0)  # Cauchy: alpha and beta pinned

result.params.alpha                            # a float
result.params.as_tuple()                       # (alpha, beta, mu, sigma)
result.negative_log_likelihood
```

The search starts from a point derived from your data's median and interquartile range as well as from the historical constant, and keeps the better optimum. That is what makes fitting data at a scale of `0.005` work where it used to stall at a boundary.

### Parametrizations

Everything runs internally in Nolan's parametrization 0. Pass `par=` to work in another, or convert explicitly:

```python
api.pdf(x, alpha=1.6, beta=0.5, mu=0.3, sigma=1.2, par='1')

params = api.StableParams.from_par(1.6, 0.5, 0.3, 1.2, par='1')
params.to_par('B')                             # (1.6, 0.5546, 0.246, 1.4243)

api.fit(sample, par='M').as_par('0')
```

| `par` | Notation | Third and fourth parameters |
|---|---|---|
| `'0'`, `'1'` | Nolan | `mu`, `sigma` (location, scale) |
| `'M'`, `'A'`, `'B'` | Zolotarev | `gamma`, `lambda` |

### Validation

Parameters are checked where you write them, once, and never inside the likelihood loop:

```python
api.pdf(x, alpha=0.2, beta=0.0)
# ValidationError: alpha -- Input should be greater than or equal to 0.5
```

`alpha` must lie in `[0.5, 2]`, which is what the tables cover; `beta` in `[-1, 1]`; `sigma` strictly positive and finite.

### 🐼 pandas

```python
import pandas as pd

prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)["close"]
returns = np.log(prices).diff().dropna()

api.pdf(returns, alpha=1.6, beta=0.0)          # a Series, same index
api.fit(returns).to_series()
# alpha    1.63
# beta    -0.07
# mu       0.00
# sigma    0.01
```

`pdf`, `cdf` and `logpdf` take a `Series` or a `DataFrame` and return one with the same index. `fit` takes a `Series` or a single-column `DataFrame`; a wider one is refused rather than silently pooled.

### 🔥 torch

Hand the functions tensors and the result carries gradients, so the log likelihood can be minimised by gradient descent inside a larger model:

```python
import torch

sample = torch.tensor(observations)
params = torch.tensor([1.4, 0.0, 0.0, 1.0], requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.03)

for _ in range(400):
    optimizer.zero_grad()
    # Keep the optimizer's raw values inside the domain: clamp is
    # differentiable in the interior and stops the gradient at the edges.
    loss = -api.logpdf(sample,
                       alpha=params[0].clamp(0.55, 1.95),
                       beta=params[1].clamp(-0.95, 0.95),
                       mu=params[2],
                       sigma=params[3].clamp(min=1e-3)).sum()
    loss.backward()
    optimizer.step()
```

The backend is chosen automatically when any argument is a tensor. `levy.set_backend('torch')` or `with levy.using('torch'):` selects it explicitly. The torch implementation is checked against the NumPy one at `rtol=1e-6` across the interpolated region and both tails, and its gradients against finite differences.

### 🧰 Rebuilding the tables

The shipped tables are enough for normal use. To rebuild them, at the default resolution or another one:

```bash
levy-tables build --jobs 8                 # ~55 CPU-minutes at the default 200x76x101
levy-tables build --size 40,16,21 --what pdf,cdf,limits
levy-tables where                          # which tables are in use, and whether any is missing
```

Tables go to a user cache directory, never into the installed package, and every build writes a `manifest.json` with the grid size, library versions and a checksum per file. `$LEVY_DATA_DIR` points the library at a directory of your own.

### ⬆️ Upgrading from 1.x

Every 1.x name still works. Away from the bugs 2.0 fixes, each returns exactly the same numbers, and each emits a `DeprecationWarning` naming its replacement:

| 1.x | 2.0 |
|---|---|
| `levy.levy(x, a, b)` | `api.pdf(x, alpha=a, beta=b)` |
| `levy.levy(x, a, b, cdf=True)` | `api.cdf(x, alpha=a, beta=b)` |
| `levy.neglog_levy(x, a, b, m, s)` | `-api.logpdf(x, alpha=a, beta=b, mu=m, sigma=s)` |
| `levy.random(a, b, m, s, shape=n)` | `api.rvs(alpha=a, beta=b, mu=m, sigma=s, size=n)` |
| `levy.fit_levy(x)` | `api.fit(x)` |

The full table, and the list of what changed on purpose, is in the [migration guide](https://github.com/josemiotto/pylevy/blob/master/docs/source/migration.md).

## 🔬 How it works

The density is tabulated on a `200 × 76 × 101` grid over `(x, alpha, beta)`, with `x` stored in arctan space so the nodes reach out to `|x| ≈ 637` and sit densest near the mode. Evaluation is a Catmull-Rom cubic interpolation over the 64 surrounding nodes, and beyond a per-`(alpha, beta)` crossover point the interpolant hands over to the analytical power-law tail. Sampling never touches the table. [How it works](https://github.com/josemiotto/pylevy/blob/master/docs/source/how_it_works.md) has the derivation and the measured accuracy.

## 🏗️ Architecture

```text
src/levy/
├── api.py               # the typed, validated 2.0 API: pdf, cdf, logpdf, rvs, fit
├── distribution.py      # evaluation: interpolation in the middle, power law in the tails
├── fitting.py           # maximum likelihood with L-BFGS-B
├── sampling.py          # Chambers-Mallows-Stuck sampler
├── parametrization.py   # the five parametrizations and the conversions between them
├── interpolation.py     # Catmull-Rom interpolation and bound folding
├── tables.py            # locating, loading, caching and repairing the lookup tables
├── constants.py         # grid geometry, fit bounds, parametrization metadata
├── backends/            # NumPy, and the optional torch implementation
├── _build/              # levy-tables: regenerating the tables by quadrature
├── _pandas.py           # labels in, labels out
└── data/                # the shipped tables (float32, 10 MB) and their manifest
```

Everything under `api.py`, `_typing.py`, `_compat.py` and `_pandas.py` is checked with `mypy --strict` and shipped with `py.typed`. The numerical core is checked by tests instead.

## 🧪 Development

### Setup

```bash
git clone https://github.com/josemiotto/pylevy.git
cd pylevy
pip install -e ".[dev,lint,pandas,torch,docs]"

pip install pre-commit
pre-commit install
```

### Running the checks

```bash
pytest -m "not build"                 # the suite, minus the slow table build (~1,070 tests)
pytest -m build                       # regenerates tiny tables by quadrature, ~70s
pytest --doctest-modules src/levy     # every docstring example runs
ruff check .
mypy
numpydoc lint src/levy/*.py src/levy/_build/*.py src/levy/backends/*.py
sphinx-build -b html -W docs/source docs/_build/html
```

All of these are gated in CI, on Linux, macOS and Windows, with NumPy 1.x and 2.x.

### 🔒 The rule that matters

**Numbers do not move by accident.** `tests/golden/golden_v1.jsonl` pins the output of every function at 280 points as exact hex floats, and CI regenerates it from scratch on every push. If your change moves any of them, either it was not supposed to and you have a bug, or it was and you owe the reviewer evidence. Never regenerate the golden file to make a test pass. [CONTRIBUTING.md](https://github.com/josemiotto/pylevy/blob/master/CONTRIBUTING.md) has the details.

## 📊 Example Output

```text
>>> result = api.fit(sample)
>>> result.params
StableParams(alpha=1.4399998850328382, beta=0.09847779011605386, mu=-0.07193641709460895, sigma=0.9620643517415507)
>>> result.to_series()
alpha    1.440000
beta     0.098478
mu      -0.071936
sigma    0.962064
Name: levy, dtype: float64

$ levy-tables where
tables in use : /.../site-packages/levy/data
packaged      : /.../site-packages/levy/data
user cache    : /Users/you/Library/Caches/pylevy
LEVY_DATA_DIR : (unset)
  pdf.npz              5.14 MB
  cdf.npz              5.16 MB
  limits.npz           0.01 MB
  manifest.json        0.00 MB
```

## 🤝 Contributing

Contributions are welcome. [CONTRIBUTING.md](https://github.com/josemiotto/pylevy/blob/master/CONTRIBUTING.md) has the setup, the checks, and what a pull request needs to say: what changed, what it was verified against, and whether the goldens moved.

1. Fork the repository
2. Create a branch for one purpose (`git checkout -b fix/one-thing`)
3. Run the checks above
4. Open a pull request with the numbers in it

### Project documentation

| Document | Contents |
| --- | --- |
| [CONTRIBUTING.md](https://github.com/josemiotto/pylevy/blob/master/CONTRIBUTING.md) | Development setup, the checks, the golden-file rule |
| [CHANGELOG.md](https://github.com/josemiotto/pylevy/blob/master/CHANGELOG.md) | What 2.0 changed, fixed and added |
| [AGENTS.md](https://github.com/josemiotto/pylevy/blob/master/AGENTS.md) | Coding conventions, for humans and coding agents alike |
| [CODE_OF_CONDUCT.md](https://github.com/josemiotto/pylevy/blob/master/CODE_OF_CONDUCT.md) | Community standards |
| [SECURITY.md](https://github.com/josemiotto/pylevy/blob/master/SECURITY.md) | Supported versions, how to report a vulnerability, and what the package does with its inputs |
| [docs/source/how_it_works.md](https://github.com/josemiotto/pylevy/blob/master/docs/source/how_it_works.md) | The grid, the interpolation, the tails, and the measured accuracy |
| [docs/source/migration.md](https://github.com/josemiotto/pylevy/blob/master/docs/source/migration.md) | Moving from 1.x |
| [docs/proposals/](https://github.com/josemiotto/pylevy/tree/master/docs/proposals) | Repository governance, the PyPI name, and the state of every upstream issue |

The rendered documentation is at <https://pylevy.readthedocs.io/en/latest/>.

## 📚 Citing

If pylevy contributes to your research, cite it via the repository's [CITATION.cff](https://github.com/josemiotto/pylevy/blob/master/CITATION.cff); GitHub renders it under **Cite this repository**. The methods it builds on:

- J. P. Nolan, *Univariate Stable Distributions*, Springer, 2020.
- J. M. Chambers, C. L. Mallows and B. W. Stuck, "A Method for Simulating Stable Random Variables", *Journal of the American Statistical Association*, 71(354), 1976.
- V. M. Zolotarev, *One-dimensional Stable Distributions*, AMS, 1986.

## 📄 License

GPL-3.0-or-later. See [LICENSE](https://github.com/josemiotto/pylevy/blob/master/LICENSE).

## 🔗 Related Resources

- [John Nolan's stable distribution pages](https://edspace.american.edu/jpnolan/stable/), the reference for the parametrizations
- [`scipy.stats.levy_stable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html), which computes the density by integration and is the natural cross-check

## 🙏 Acknowledgements

- **Paul Harrison** wrote the original package and the table-interpolation approach, in 2005
- **José María Miotto** picked it up in 2016 and has maintained it since
- **Esteban Carisimo** wrote 2.0: the typed API, the extras, the test suite, and the CI

And the users who reported the bugs 2.0 fixes, on the issue tracker, with reproductions.
