# Which name to publish under

## The situation

**This package has never been published to PyPI.** `setup.py` declared
`name='PyLevy'`, and `pyproject.toml` still does, but no release was ever
uploaded.

The `pylevy` slot on PyPI is taken. It belongs to Paul Harrison's original 2005
package — version 0.3, maintainer `pfh` — which is the ancestor this repository
was forked from, and which Harrison is still credited as an author of here.

So `pip install pylevy` today installs twenty-year-old code, and there is
currently no way to install this repository except from source. That is worth
stating plainly, because it is the single largest reason the package has no
traction: there is nothing to install.

## Two paths

### (a) Take over the `pylevy` name

Strong case, and it costs one email:

- Same lineage. This *is* the continuation of that package.
- Paul Harrison is credited as an author here, so there is no dispute about
  provenance.
- The PyPI project has been dormant for twenty years.
- The name is what people already type.

Two routes, in order of preference:

1. **Ask.** Email `pfh` and ask to be added as an owner of the PyPI project.
   Twenty years dormant, same lineage, friendly request — this usually just
   works, and it is far faster than the alternative.
2. **PEP 541.** If there is no reply after a reasonable interval, file a
   [PEP 541 name-transfer request](https://peps.python.org/pep-0541/) at
   `pypi/support`. The criteria — abandoned project, requester is continuing the
   same work — are met about as clearly as they ever are.

### (b) Publish as `levy-stable`

Verified free at the time of writing, as are `pylevy2` and `pylevy-ng`.

`levy-stable` is the better fallback: it says what the package computes, it is
what someone searching for this would plausibly type, and it does not read as a
fork-of-a-fork the way `pylevy2` does.

The cost is real, though: the import name would stay `levy` while the
distribution name is `levy-stable`, which is a small permanent papercut for
users, and search traffic for "pylevy" would keep landing on the 2005 package.

## Recommendation

Pursue (a) first, fall back to (b). Start with the email, not the PEP 541
request.

## This does not block anything

The release workflow reads the distribution name from `pyproject.toml`, so the
decision can be made at merge time — or after — by editing one line:

```toml
[project]
name = "PyLevy"        # or "levy-stable"
```

Nothing else in the repository refers to the distribution name. The import name
is `levy` either way.

## While it is unresolved

The README should not tell people to `pip install pylevy`, because that installs
the 2005 package. It currently says `pip install .`, which is correct for a
source checkout and should stay that way until the name is settled.
