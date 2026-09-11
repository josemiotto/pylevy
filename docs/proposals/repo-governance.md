# Repository governance: a checklist for the owner

Everything in this file needs permissions a pull request does not have. It is
written down here so that the work is visible and can be done in one sitting,
rather than discovered piecemeal.

Nothing here changes any code. All of it makes the repository harder to break by
accident.

## 1. Protect `master`

Settings → Branches → Add branch protection rule, pattern `master`:

- [ ] Require a pull request before merging
- [ ] Require status checks to pass, and mark these as required:
      every `test` leg -- `py3.9` through `py3.13`, `macos py3.10`,
      `macos py3.13`, `windows py3.9`, `windows py3.13` -- plus `numpy 1.x`,
      `numpy 2.x (pinned)`, `lint and docstrings`, `doctests`,
      `golden file is reproducible`, `package builds`, `optional extras`,
      `table generation`, and the docs workflow's `build` (the Sphinx build
      with `-W`; its `links` job is informational and stays optional).
      Everything that gates, in other words: a check that is not required is
      one a platform-specific or documentation failure can slip past.
- [ ] Require branches to be up to date before merging
- [ ] Do not allow force pushes
- [ ] Do not allow deletions

And a second ruleset for tags, pattern `v*`, so that a release tag cannot be
moved or removed once it exists:

- [ ] Restrict creations to maintainers
- [ ] Do not allow updates (no force-moving a tag)
- [ ] Do not allow deletions

The release workflow checks on its own that a tag is a tag and that the
tagged commit is on `master`, and builds every job from that one commit; the
tag ruleset closes what a workflow cannot, which is the tag being moved after
the fact.

The `golden file is reproducible` check is the important one. It regenerates all
251 golden records from scratch and compares them against the committed file, so
a pull request cannot quietly edit the thing that pins the package's numerical
behaviour.

## 2. `CODEOWNERS`

A `CODEOWNERS` file does nothing without branch protection, which is why it is
proposed here rather than added by a pull request. Once the rule above exists,
add `.github/CODEOWNERS`:

```
# Everything, by default.
*                       @josemiotto

# The numerical core and the file that pins its output. A change here is a
# change to the package's results.
/src/levy/distribution.py    @josemiotto
/src/levy/interpolation.py   @josemiotto
/src/levy/sampling.py        @josemiotto
/tests/golden/               @josemiotto

# Release and publishing.
/.github/workflows/release.yml  @josemiotto
```

Then tick **Require review from Code Owners** in the branch rule.

## 3. Reviewing policy

Proposed, to go in `CONTRIBUTING.md` once agreed:

- One approving review to merge; two for anything that moves a golden record.
- The reviewer's job on a numerical change is to check the *evidence*, not to
  re-derive the mathematics: which records moved, by how much, and the
  comparison against `calculate_levy` ground truth.
- CI being green is necessary and not sufficient. A pull request that
  regenerates the goldens without justification is rejected however green it is.
- Anything that changes the public API needs a changelog entry in the same pull
  request.

## 4. Trusted Publishing on PyPI

`.github/workflows/release.yml` publishes via OIDC, which needs one
configuration on PyPI and no stored secret:

- [ ] On PyPI, project → Publishing → Add a new pending publisher
      - Owner: `josemiotto`, repository: `pylevy`
      - Workflow: `release.yml`, environment: `pypi`
- [ ] In GitHub: Settings → Environments → New environment `pypi`, and add
      yourself as a required reviewer so a publish cannot happen unattended
- [ ] In the same environment, set deployment branches and tags to
      **selected**: `master` and `v*`. The workflow refuses a manual run
      started anywhere but `master` and a tag whose commit is not on
      `master`; this setting makes the platform refuse them too, so the OIDC
      credential is never issued to a run that started from a branch.

See [pypi-name.md](pypi-name.md) for which project name this applies to.

## 5. GitHub Pages

`.github/workflows/docs.yml` builds the site on every pull request and
uploads it as an artifact, but does not publish it: the publish job was
removed on 2026-09-11 because Pages is not enabled and only an admin can
enable it, so the job failed on every push to `master`. Two steps, in order:

- [ ] Settings → Pages → Source: **GitHub Actions**
- [ ] Put the `publish` job back in `docs.yml` -- the workflow header holds
      the job, ready to paste -- and switch the artifact step to
      `actions/upload-pages-artifact`.

## 6. Fix the repository "About"

The sidebar currently has no description, no topics and no website, so the
repository is effectively unfindable by search. Suggested:

- **Description:** *Levy alpha-stable distributions for Python: density,
  distribution function, sampling, and maximum-likelihood fitting by
  interpolation.*
- **Website:** the GitHub Pages URL, once §5 is done
- **Topics:** `python`, `statistics`, `probability-distributions`,
  `levy-distribution`, `alpha-stable`, `heavy-tails`, `maximum-likelihood`,
  `scipy`, `numpy`
- [ ] Tick "Releases" and "Packages" in the sidebar

## 7. Stale branches

- [ ] **`dev` (7 commits) — close with credit, do not merge.** It regresses
      three things: it moves the `.npz` files to the repository root, outside
      the installed package, which breaks `package_data` entirely; it deletes
      all of `docs/`; and it flattens `levy/` to a single `levy.py`, the
      opposite of the module split. It also rewrote both 12 MB tables, which is
      part of why `.git` is 55 MB. Its one good idea — merging the two limit
      tables into one archive — was salvaged and credited in the float32 pull
      request.
- [ ] **`dependabot/pip/numpy-1.22.0` — close, superseded.** Targets
      `requirements.txt`, which no longer exists.
- [ ] **`dependabot/pip/scipy-1.10.0` — close, superseded.** Same reason.

## 8. Issue labels

The defaults are fine except that a numerical package wants one more:

- [ ] `numerics` — for anything where the disagreement is about a value rather
      than about behaviour. These need a different kind of review.
