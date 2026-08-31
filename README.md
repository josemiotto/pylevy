# pylevy

Levy alpha-stable distributions for Python: density, distribution function,
sampling, and maximum-likelihood fitting.

Direct computation of a stable density requires a lengthy numerical
integration. pylevy interpolates a precomputed table instead, which is what
makes fitting by maximum likelihood fast enough to be practical.

## Install

    pip install .

Requires Python 3.9 or newer.

## Use

```python
import numpy as np
from levy import api

x = np.array([-1.0, 0.0, 1.0])

api.pdf(x, alpha=1.5, beta=0.0)          # density
api.cdf(x, alpha=1.5, beta=0.0)          # distribution function
api.logpdf(x, alpha=1.5, beta=0.0)       # log density

sample = api.rvs(alpha=1.5, beta=0.0, size=1000, random_state=0)

result = api.fit(sample)
result.params                            # StableParams(alpha=..., beta=..., mu=..., sigma=...)
result.negative_log_likelihood
```

Parameters are validated where you write them:

```python
api.pdf(x, alpha=0.2, beta=0.0)
# ValidationError: alpha -- Input should be greater than or equal to 0.5
```

`alpha` must lie in `[0.5, 2]`, which is what the lookup tables cover.

## Parametrizations

Parametrizations 0 and 1 in the notation of Nolan, and M, A and B from
Zolotarev. Everything runs internally in parametrization 0; pass `par=` to give
parameters in another one, or convert explicitly:

```python
params = api.StableParams.from_par(1.6, 0.5, 0.3, 1.2, par='1')
params.to_par('B')
```

## pandas

    pip install ".[pandas]"

`pdf`, `cdf` and `logpdf` accept a `Series` or a `DataFrame` and return one
with the same index; `fit` accepts a `Series`, and `FitResult.to_series()`
reports the parameters under their names.

```python
prices = pd.Series(..., index=dates)
returns = np.log(prices).diff().dropna()

api.fit(returns).to_series()
# alpha    1.63
# beta    -0.07
# mu       0.00
# sigma    0.01
```

Without the extra, pandas is never imported.

## torch

    pip install ".[torch]"

Hand the functions tensors and the result carries gradients, so the log
likelihood can be minimised by gradient descent inside a larger model:

```python
import torch
from levy import api

sample = torch.tensor(observations)
params = torch.tensor([1.4, 0.0, 0.0, 1.0], requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.03)

for _ in range(400):
    optimizer.zero_grad()
    loss = -api.logpdf(sample, alpha=params[0], beta=params[1],
                       mu=params[2], sigma=params[3]).sum()
    loss.backward()
    optimizer.step()
```

`levy.set_backend('torch')` or `with levy.using('torch'):` selects it
explicitly. NumPy is the default, and without the extra torch is never
imported.

## Regenerating the tables

The shipped tables are enough for normal use. To rebuild them, at the default
or any other resolution:

    levy-tables build --jobs 8
    levy-tables where

They are written to a user cache directory, never into the installed package.
A full rebuild is roughly 55 CPU-minutes.

## Upgrading from 1.x

Every 1.x name still works and returns exactly the same numbers; each emits a
`DeprecationWarning` naming its replacement. See
[CHANGELOG.md](CHANGELOG.md) for the migration table and for the bugs 2.0
fixes.

## Documentation

<https://pylevy.readthedocs.io/en/latest/index.html>

[How it works](docs/source/how_it_works.md) explains the interpolation scheme
and where the accuracy comes from.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the checks, and the rule that
matters: the golden file pins this package's numerical output, and a change that
moves it needs evidence, not a regeneration. [AGENTS.md](AGENTS.md) has the
coding conventions.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE).

Written by Paul Harrison and José María Miotto, with contributions from
Esteban Carisimo.
