# How it works

A Levy alpha-stable distribution has no closed-form density except in three
special cases (Gaussian, Cauchy, Lévy). Everywhere else the density is defined
through its characteristic function, and getting a number out of it means a
numerical integration — one that is slow and, near the origin for some
parameters, badly conditioned.

Maximum-likelihood fitting evaluates the density once per observation per
optimizer step. At a few milliseconds per evaluation, fitting a thousand points
is not something you do interactively.

pylevy's answer is to do the integration once, offline, on a grid, and
interpolate. That is the whole design, and everything below is a consequence of
it.

## The grid

The tables are `(200, 76, 101)` — 200 points in `x`, 76 in `alpha`, 101 in
`beta`:

| axis | from | to | nodes |
|---|---|---|---|
| `x` | `-pi/2 * 0.999` | `+pi/2 * 0.999` | 200 |
| `alpha` | 0.5 | 2.0 | 76 |
| `beta` | -1.0 | 1.0 | 101 |

`alpha` and `beta` are stored directly. `x` is not: the grid holds the density
at `tan(x_axis)`, so `x` runs over the **whole real line** in 200 nodes rather
than over some arbitrary truncation. The `0.999` keeps the endpoints off the
pole of the tangent.

Working in arctan space is what makes a fixed-size table cover an unbounded
domain, and it puts the nodes where they are needed: dense near the mode, sparse
far out in the tails where the density is smooth and nearly a power law.

`alpha < 0.5` is not tabulated, so it is not supported. Values outside the grid
raise `ValueError` — in 1.x they produced a negative array index, which Python
accepts, and returned the values for `alpha ≈ 1.94` without complaint.

## Interpolation

Evaluation is a Catmull-Rom cubic interpolation in all three dimensions at once:
a weighted sum over the 4×4×4 = 64 grid nodes surrounding the query point, with
weights that are cubic polynomials in the fractional offsets.

Catmull-Rom takes its tangents from centred differences. That makes it exact for
quadratics — measured at 1.3e-15 on a quadratic test grid — and it is C¹ across
cell boundaries, which matters because the fit differentiates the likelihood
numerically.

It is not positivity-preserving. The cubic weights have negative lobes, so a
grid of non-negative values can interpolate slightly below zero. That is why
`neglog_levy` floors the density at 1e-100 before taking a logarithm.

The weights are `float64` and the grid is `float32` (see below); multiplying
them promotes to `float64`, so the accumulation happens at full precision.

## The tails

Interpolation covers the middle. Far enough out, the density is replaced by its
analytical asymptote:

```
pdf(x) ~ alpha * sin(pi*alpha/2) * Gamma(alpha) / pi * (1 ± beta) * |x|^(-alpha-1)
```

Two more tables, `lower_limit` and `upper_limit`, say where to switch, one
crossover pair per `(alpha, beta)` cell. They were found by searching for the
point where the asymptote best matches the interpolated CDF.

**A known defect lives here.** The two branches do not meet: at the crossover
the CDF steps down by up to 2.6e-03. So the choice of crossover cell mostly
decides which side of a seam you land on, which is why an apparently obvious
improvement — rounding to the nearest cell instead of truncating — measured
*worse* on the median, and was reverted:

| strategy | median rel err | mean | p90 |
|---|---|---|---|
| truncate (current) | 1.1164e-02 | 3.5488e-01 | 1.4560 |
| round to nearest | 1.4552e-02 | 4.3251e-01 | 1.4656 |

Fixing it properly means regenerating the limit tables or reconciling the two
branches, not adjusting the index arithmetic.

## Why float32 is enough

The tables ship as `float32`, which halved the wheel from 24.7 MB to 10.3 MB.
That is only defensible with numbers, so here they are: over a dense sweep of
`(alpha, beta, x)`, the worst relative deviation from the `float64` tables is
**5.96e-08** for the pdf and **5.70e-08** at the 99.9th percentile for the cdf.

That is `float32`'s epsilon, and it is three to four orders of magnitude below
the interpolation error that already dominates — which is itself far below the
crossover discontinuity above. Compared against the quadrature ground truth, the
`float32` and `float64` tables are indistinguishable.

Compression was not the lever: gzip recovers only 9% on these arrays. Precision
was.

## Parametrizations

Five, and they disagree about what the four numbers mean.

- **Nolan 0** — what the tables are built in, and what everything converts to.
  `E(X) = delta_0 - beta*gamma*tan(pi*alpha/2)`.
- **Nolan 1** — easier to interpret, since `E(X) = delta_1`. Discontinuous in
  `alpha` at 1, which is why 0 is preferred for anything numerical.
- **Zolotarev M, A, B** — from *One-dimensional Stable Distributions*.

Everything routes through 0. `StableParams.from_par` converts on the way in and
then validates, which is the only place a `beta_B` that maps outside `[-1, 1]`
can be caught: `beta_0 = tan(beta_B * psi(alpha)) / tan(alpha*pi/2)` depends on
`alpha`, so a box constraint on `beta_B` does not keep `beta_0` in range.

```{note}
`par_bounds` still applies parametrization-0 box constraints to all five
parametrizations, so a fit in `'B'` searches the wrong feasible region. This is
a known open defect, listed in the changelog.
```

## Sampling

`rvs` does not invert the interpolated CDF. It uses the Chambers–Mallows–Stuck
transform, which turns two uniforms into an exact stable variate — no table
involved, so sampling accuracy does not depend on the grid at all.

The transform divides by `tan(pi*alpha/2)`, which has a pole at `alpha = 1`, so
`alpha` within 1e-8 of 1 is nudged aside. The nudge used to be 1e-15, close
enough to the pole that rounding the argument became an ~11% error in the
result: at `beta = ±1` that produced NaN for about 0.9% of draws, and the
surviving samples failed a Kolmogorov–Smirnov test against this package's own
CDF at p = 3e-07.

## Regenerating the tables

```console
levy-tables build --jobs 8      # into the user cache directory
levy-tables where               # which tables are actually in use
```

A full rebuild is roughly 55 CPU-minutes: 25 for the densities, 30 for the
crossover limits, which depend on the CDF table and must be built after it.
Output goes to a user cache directory — never into the installed package — along
with a `manifest.json` recording the grid size, the NumPy and SciPy versions,
and a SHA256 per array.

`$LEVY_DATA_DIR` overrides the search path, and the loader prefers the cache
directory over the packaged tables when it holds a complete set.

## The differentiable backend

With `pylevy[torch]` installed, the same interpolation runs on tensors and
carries gradients, so a stable distribution can be fitted by gradient descent
inside a larger model.

Two things in it are deliberately not differentiable: the integer grid cell used
to look up the crossover limits, and the mask it implies. Both are *choices*
rather than quantities — the crossover is piecewise constant in `alpha` — so the
gradient is right everywhere except exactly at a cell boundary, where a step
function has no derivative anyway.

## References

1. J. P. Nolan, *Univariate Stable Distributions*, Springer, 2020.
2. V. M. Zolotarev, *One-dimensional Stable Distributions*, AMS, 1986.
3. J. M. Chambers, C. L. Mallows and B. W. Stuck, "A Method for Simulating
   Stable Random Variables", *JASA* **71**(354), 340–344, 1976.
