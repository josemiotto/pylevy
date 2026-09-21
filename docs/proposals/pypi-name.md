# The distribution name

The package is published on PyPI as **`levy-stable`**. The import name is
`levy`, and the repository is `josemiotto/pylevy`. Three names, on purpose;
this note records why, so nobody reopens the question by accident.

## Why not `pylevy`

The `pylevy` name on PyPI belongs to Paul Harrison's original PyLevy (2005),
the package this repository descends from. Its three releases carry no files;
nothing was ever installable under the name.

He was asked to add a maintainer, on 1 September 2026 and again on 19
September. He replied on 20 September and declined, preferring that this
project use a name of its own. That is his call: he has had no part in this
repository, and a name is a claim of association. A PEP 541 request filed on
11 September while waiting (pypi/support#12225) was withdrawn the day he
replied, as it had promised.

He remains credited as the author of the 2005 package: in the README, in
`CITATION.cff` as a reference, and in the copyright notice at the top of
`levy/__init__.py`.

## Why `levy-stable`

It names what the package computes, and it is the name of the distribution in
`scipy.stats`, so it is what anyone looking for this will type. It owes nothing
to anyone. Verified free on 20 September 2026; `pylevy2`, `pylevy-ng`,
`alphastable` and `levyfit` were free too, and the `pylevy*` spellings were
set aside because they lean on a name whose owner had just said no.

Two things a user may notice:

- `pip install levy-stable` but `import levy`, as with `scikit-learn`/`sklearn`
  or `Pillow`/`PIL`. The README says so on its first line.
- The unrelated PyPI package `levy` (a configuration parser, last release
  2021) also installs a top-level `levy` package. The two cannot coexist in one
  environment. This has been true for as long as this package has used
  `import levy`; changing the import name would break every 1.x user for the
  sake of that one collision, so it stays.

The repository is not renamed: the URL is what issues, citations and the Read
the Docs project point at, and it is not this package's to rename. A full
rename is a separate decision that loses nothing by waiting.

## Where the name is spelled

The release workflow reads the distribution name from `pyproject.toml`, so
renaming again is one line there plus the places that spell it for humans:

- `src/levy/_compat.py`: the install hint in the missing-extra error, and the
  tests that assert its wording (`tests/test_compat.py`,
  `tests/test_no_pandas.py`, `tests/test_no_torch.py`);
- the same hint in `AGENTS.md`, `docs/source/how_it_works.md` and
  `docs/source/migration.md`;
- the install sections of `README.md` and `docs/source/index.rst`;
- the release note at the top of `CHANGELOG.md`.

All findable with `grep -rn 'levy-stable'`.
