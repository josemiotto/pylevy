# Migrating from 1.x

**Nothing breaks.** Every 1.x name still works and still returns exactly the
same numbers — bit for bit, verified by installing wheels built from 1.1 and
from 2.0 into two environments and comparing 99,312 values.

What changed is that each 1.x name now emits a `DeprecationWarning` when reached
through `levy.`, naming its replacement. They are scheduled for removal in 3.0.

If you want to keep the 1.x behaviour and silence the warning, import from the
module the function actually lives in — `levy.distribution.levy` rather than
`levy.levy`. No warning, same function object.

## The table

| 1.x | 2.0 |
|---|---|
| `levy.levy(x, a, b, cdf=False)` | `levy.api.pdf(x, alpha=a, beta=b)` |
| `levy.levy(x, a, b, cdf=True)` | `levy.api.cdf(x, alpha=a, beta=b)` |
| `levy.neglog_levy(x, a, b, m, s)` | `-levy.api.logpdf(x, alpha=a, beta=b, mu=m, sigma=s)` |
| `levy.fit_levy(x)` | `levy.api.fit(x)` |
| `levy.random(a, b, m, s, shape=n)` | `levy.api.rvs(alpha=a, beta=b, mu=m, sigma=s, size=n)` |
| `levy.Parameters` | `levy.api.StableParams` |
| `levy.convert_to_par0[p](v)` | `levy.api.StableParams.from_par(*v, par=p)` |
| `levy.convert_from_par0[p](v)` | `levy.api.StableParams(*v).to_par(p)` |
| `levy.size` | `levy.constants.size` |
| `levy.par_bounds` | `levy.constants.par_bounds` |
| `levy.par_names` | `levy.constants.par_names` |
| `levy.default` | `levy.constants.default` |
| `levy.f_bounds` | `levy.constants.f_bounds` |

## Three differences worth reading

### `logpdf` has the opposite sign to `neglog_levy`

`neglog_levy` returns `-log(pdf(x))`. `logpdf` returns `log(pdf(x))`, following
`scipy.stats`. If you are porting a likelihood, the sign flip is the one thing
that will silently give you a maximum where you wanted a minimum.

```python
# 1.x
total = levy.neglog_levy(x, alpha, beta, mu, sigma).sum()

# 2.0
total = -levy.api.logpdf(x, alpha=alpha, beta=beta, mu=mu, sigma=sigma).sum()
```

### The new functions are keyword-only

`api.pdf(x, 1.5, 0.0)` is a `TypeError`. This is deliberate: `levy(x, 1.5, 0.0,
0.0, 1.0, True)` is not something anyone should have to decode, and the six
positional arguments were easy to get out of order.

### Bad parameters raise instead of returning a number

```python
levy.levy(x, 0.4, 0.0)            # 1.x: values for alpha ~ 1.94, silently
levy.api.pdf(x, alpha=0.4, beta=0.0)   # 2.0: ValidationError, naming alpha
```

Along the same lines, `api.fit` rejects a keyword that is not a parameter name.
`fit_levy` took `**kwargs` and ignored anything it did not recognise, so
`fit_levy(x, beta_=0.0)` fitted `beta` freely and said nothing.

## Fits give you an object, not a tuple

```python
# 1.x
parameters, nll = levy.fit_levy(x)
alpha, beta, mu, sigma = parameters.get('0')

# 2.0
result = levy.api.fit(x)
result.params.alpha, result.params.beta      # frozen and validated
result.negative_log_likelihood
result.params.to_par('B')                    # in another parametrization
result.to_series()                           # with pylevy[pandas]
```

## Bugs that were fixed

If you relied on any of these, your numbers will change — for the better. All
are listed in the [changelog](changelog.md).

- `random(alpha=2.0, ...)` ignored `mu` and `sigma` entirely.
- `random(1.0, ±1.0)` returned NaN for about 0.9% of draws, and the rest came
  from the wrong distribution.
- `alpha < 0.5` silently returned values for `alpha ≈ 1.94`.
- Four cells of the shipped CDF table held `5.72e+307` instead of a probability;
  `levy(0.0, 0.58, 0.74, cdf=True)` returned `6.44e+307`.
- `_reflect` could hang: with the sigma bounds, folding `1e30` needed ~1e20
  iterations.

## Checking your own code

Run your test suite with deprecations promoted to errors to find every 1.x call
site at once:

```console
python -m pytest -W error::DeprecationWarning
```

`import levy` on its own never warns — the warning fires when a deprecated name
is *used*, not when the package is imported.
