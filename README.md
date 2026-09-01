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

## License

GPL-3.0-or-later; see [LICENSE](LICENSE).

Written by Paul Harrison and José María Miotto, with contributions from
Esteban Carisimo.
