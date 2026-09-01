# What this stack does about the open upstream issues

A record of what was measured against each existing issue in
`josemiotto/pylevy`, so that the pull requests can point at evidence instead of
re-arguing from scratch, and so that nothing found here is lost between now and
when they are opened.

Two rules were applied throughout:

- **Nothing here corrects anyone.** Where a maintainer or a reporter reached a
  conclusion, it is quoted and credited. In every case they were reading the
  symptom correctly; what was missing was a mechanism.
- **Where a position has already been taken, it is not overridden quietly.**
  Issue #15 is the clearest case: it is closed, deliberately, and this stack
  does not touch it. See below for what would have to be true to reopen it.

---

## #20 — "La estimación de los parámetros a veces falla" (open)

Reported 2024-09-30 by @hurtadillero. **Fixed by this stack.**

### The report

Fitting `alpha=1.6, beta=0, mu=0, sigma=0.005` over 1000 samples of 10,000
points, most fits were fine, but some returned

```
(par=0, alpha=2.00, beta=1.00, mu=-0.00, sigma=0.01, -27450.39157380032)
(0.513090465, -0.0279757810, 0.000122625, 0.0130914520)
```

The report is unusually good and did most of the diagnostic work. They ruled out
fat-tail outliers by plotting the log-log survival function against the
theoretical curve, and cross-checked with scipy's fit, which succeeded on both
samples. Their conclusion — *"el problema está en el método numérico que resuelve
o en algún bug del código"* — is correct.

@josemiotto's reply is also correct:

> efectivamente parece alguna inestabilidad que te lleva alpha a algún extremo
> (.5 o 2 (cuando alpha=2, beta es irrelevante por que es la gaussiana))

The note about `beta` being irrelevant at `alpha=2` is what identifies the
reported `beta=1.00` as an artefact rather than a second symptom.

### It is not a local optimum

Worth establishing before fixing, because the remedy differs. Taking the
reported stopping point `(alpha=2.0, beta=1.0, mu=0, sigma=0.01)` and moving one
coordinate at a time:

| perturbation | change in negative log likelihood |
|---|---|
| `alpha` 2.0 → 1.98 | **−3657.79** (downhill) |
| `sigma` 0.01 → 0.009 | **−104.60** (downhill) |
| `sigma` 0.01 → 0.011 | **−394.10** (downhill) |
| `beta` 1.0 → 0.90 | +0.56 (uphill; irrelevant at alpha=2) |

The optimizer stopped at a point that is not stationary.

### Mechanism

`Parameters` starts every fit at the constant `[1.5, 0, 0, 1]`, in every
parametrization, without looking at the data. For `sigma = 0.005` that start is
200× too wide and L-BFGS-B does not recover. This is why the failure is
scale-dependent, and why scipy — which derives a starting scale from the data —
was unaffected.

### Reproduction

400 samples of 10,000 points at the reported parameters, `numpy.random.seed(10000 + i)`:

```
9 / 400 = 2.25% of fits stopped early
```

| seed | alpha | beta | sigma | nll left unclaimed |
|---|---|---|---|---|
| 10007 | 0.5738 | +0.061 | 0.0279 | 8312.3 |
| 10028 | 0.9415 | +0.530 | 0.0046 | 1290.3 |
| 10033 | 0.5977 | −0.104 | 0.0311 | 9054.1 |
| 10052 | 0.9911 | +0.200 | 0.0084 | |
| 10178 | 0.5192 | +0.046 | 0.0074 | |
| 10208 | 0.6332 | +0.040 | 0.0131 | |
| 10241 | 0.5823 | +0.073 | 0.0287 | |
| 10251 | 0.5986 | +0.115 | 0.0317 | |
| 10317 | 0.8865 | +0.060 | 0.0039 | |

**Caveat to state in the PR:** every failure here pinned `alpha` *low*
(0.52–0.99). The `alpha=2.00` mode from the report was not reproduced. It is the
same failure — a non-stationary stop near a boundary, at the same rate and
scale — but the claim should not be overstated.

### Fix

A second starting point from the sample **median** and **interquartile range**
— not the mean and standard deviation, neither of which exists for `alpha < 2` —
tried alongside the historical constant, keeping whichever optimum is better.

Calibration: the IQR of a standard `sigma=1` stable variate is 1.945 at
`alpha=1.5` and stays within 1.90–2.35 across the supported range, so dividing
by 2 is sufficient for a starting point. Unit-scale data then starts at 0.973,
within 3% of the 1.0 it replaces.

**An approach that was tried and rejected:** *substituting* the data-derived
start. That regressed M, A and B from +0.6 to +22.1, because converting a
location-scale start into parametrizations whose third and fourth parameters are
not a location and a scale leaves the optimizer worse conditioned than the
constant did. Adding the start instead makes the result an argmin over
candidates, so the returned likelihood can only stay the same or improve.

### Results

Gap to the likelihood at the true parameters on the issue-20 sample; lower is
better, negative means the fit beat the truth:

| parametrization | before | after |
|---|---|---|
| `'0'` | **+8312.3** | **−2.1** |
| `'1'` | −2.6 | −2.6 |
| `'M'` | +0.6 | +0.6 |
| `'A'` | +0.6 | +0.6 |
| `'B'` | **+994.9** | **+22.1** |

Over the ten golden fit cases, comparing old code against new **on the same
machine and library versions** — not against the committed file, which carries
platform ULP drift — three improve strictly (up to 2.0e-08), seven are
identical, none get worse.

Repeating the 400-sample sweep with the fix in place:

```
before:  9 / 400 = 2.25%
after:   0 / 400 = 0.00%
```

Residual, stated rather than tuned away: `'B'` at +22.1 is still short of the
optimum `'0'` reaches. That one does look like a genuine local optimum, and no
attempt was made to tune it away on a single sample.

### CI guard — required, not optional

`tests/test_fit_starting_scale.py`, 41 tests, pins the nine seeds above plus the
never-worse guarantee, pinned-parameter preservation in all five
parametrizations, bound compliance from data scales 1e-4 to 1e4, and fallback on
degenerate input.

These are **ordinary tests, not behind the `build` marker**, so they run on all
nine platform and version legs of the matrix. `.github/workflows/ci.yml` carries
a comment at the selector saying so, because excluding `slow` to save two
minutes is the obvious future edit that would silently retire them.

---

## #15 — "Computation of cdf, tail approximation, and limits is incorrect" (closed)

Reported 2020-07-20 by @ragibson. **Deliberately not addressed by this stack.**

@josemiotto closed it:

> The upper and lower limits are what they are, they are not incorrect. Those
> "extreme" discontinuities are in the order of 5e-3 in the plot you shown, and
> they are intended to be there; not only that, I guess they are unavoidable if
> you want to use the tail approximation.

What was measured here, for the record:

- **The discontinuity is real and quantified.** The CDF steps down by up to
  **2.57e-03** at the crossover between the interpolated branch and the
  power-law tail.
- **It is the dominant error term** in that region. It explains why an
  apparently obvious improvement — rounding the crossover grid index to nearest
  instead of truncating — measured *worse* (median relative error 1.4552e-02
  against 1.1164e-02), and was reverted. That measurement is recorded in the
  docstring of `_grid_index`.
- **One concrete defect from that thread is already gone.** ragibson reported
  `levy.levy(-1000, alpha=0.5, beta=0.0, cdf=True)` returning `1.0126156626`,
  a CDF above 1. It now returns `0.0126156626`, and that value is *correct*: an
  `alpha=0.5` tail genuinely places 1.26% of its mass below −1000. Swept across
  the supported range, the largest far-left-tail CDF value is 0.068 at
  `alpha=0.5, beta=−1`. No CDF above 1 remains.

**Position taken here:** the maintainer's judgement stands, and this stack does
not reopen it. His stated reason — *"I guess they are unavoidable"* — is a
hypothesis rather than a demonstration, and ragibson's technical points about
`_get_closest_approx` (the all-NaN `argmin` returning 0, discarding 95% of the
computed points) were substantive. So the issue is not closed on the merits
forever; it is closed until someone brings a **demonstration that the
discontinuity is avoidable**, which is the only argument that actually answers
what he wrote.

Doing that means regenerating the crossover limit tables (~30 CPU-minutes) and
moving the golden file. It is a deliberate decision, not a drive-by, and it is
out of scope for this stack.

---

## #14 — "Functions should warn about valid parameter ranges" (closed)

Reported 2020-07-17 by @ragibson. **This stack does what he asked, against the
maintainer's stated preference — so the PR needs to answer it directly.**

@josemiotto closed it with:

> To answer your concern, I'm not raising an error, because when calling the
> function thousands of times you don't want that.

That is a performance objection, and it is measurable:

| | |
|---|---|
| `levy()` on 100,000 points | 110.196 ms |
| one `_check_alpha_beta` call | 828 ns |
| validation as a share of a call | **0.00075%** |

The check is once per call, not once per point, so its cost does not scale with
the data. For the typed API the argument is stronger still: pydantic validates
once at the boundary and never inside the likelihood loop, and
`tests/test_hot_loop.py` enforces that by counting model constructions during a
fit and failing if the count grows with sample size.

ragibson's own follow-up conceded the performance point and suggested the check
belong to `Parameters`. Worth quoting in the PR, since it shows the two
positions were closer than the thread's ending suggests.

Related: the exactness of that check caused its own bug, found and fixed in this
stack — converting from Zolotarev's B overshoots `beta = 1` by up to 44 ULP,
and an exact check turned that into a `ValueError` that aborted 9 of 36 measured
`par='B'` fits. Values within 1e-12 of an endpoint are now snapped.

---

## Other open items

| | |
|---|---|
| **#16** (open PR, 2020-11) | "Added optional weight vector to `fit_levy`" — never reviewed. Not addressed here; worth acknowledging when the stack opens, since it touches `fit_levy` and will conflict. |
| **#17** (open, 2021-12) | "Bibtex" — addressed: the stack adds `CITATION.cff`, which GitHub renders as a citation widget and exports to BibTeX. |
| **#18, #19** (open dependabot PRs) | Bump numpy / scipy in `requirements.txt`, which the packaging PR deletes. Superseded; can be closed. |
| **#33** (open, filed while preparing this stack) | `_crossover_cell` runs `argmin` over the *masked* subarray but maps the index back through `li1`. The mask is strict, so the first masked sample is one step past the bound and every crossover limit is off by exactly `dx = 0.1`. The expression is byte-identical on `master`, so the shipped tables carry it. Fixing it means regenerating the limit tables and re-pinning goldens, so it needs its own PR. Plausibly a contributing cause of the two crossover `xfail`s still open in `test_known_bugs.py`. |
| **`dev` branch** (7 commits) | Do not merge. See [repo-governance.md](repo-governance.md). |

---

## Measurement provenance

Everything above was measured on 2026-08-31 against the stack at
`fix/fit-starting-scale`, on macOS with NumPy 2.x, except the upstream-master
checks in #14 and #15, which were run against `origin/master` directly. The
per-parametrization local-optimum rates quoted in `CHANGELOG.md` come from a
separate sweep of 12 truths × 3 seeds × 5 parametrizations.

Numbers that came from a single sample are labelled as such. Where a claim is
weaker than it looks — the unreproduced `alpha=2.00` mode in #20 — that is said
in the text rather than left for a reviewer to discover.
