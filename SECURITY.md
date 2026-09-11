# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 2.x     | Yes |
| 1.x     | No. 1.x was never published to PyPI; move to 2.0, where every 1.x name still works |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately by email to the maintainer listed under `maintainers` in
`pyproject.toml` (José María Miotto, <josemiotto@gmail.com>). Include the
version, what the problem is, and a minimal reproduction if you have one. You
will get an acknowledgement within a week. A fix is released as a patch version
and recorded in `CHANGELOG.md`.

Once GitHub's private vulnerability reporting is enabled for this repository
(an administrator setting, listed in `docs/proposals/repo-governance.md`),
reports can also go through
<https://github.com/josemiotto/pylevy/security/advisories/new>.

## What this package does and does not do

pylevy is a numerical library. It reads no network, runs no service, and
executes no code from its inputs. The only files it reads are the lookup tables
(`.npz` archives) and a `manifest.json`, from the installed package or from a
directory the user chooses (`$LEVY_DATA_DIR` or the user cache); they are loaded
with `numpy.load` **without** `allow_pickle`, so an archive cannot execute code,
and an archive that does not parse is reported as an error naming the file.

The `levy-tables` command writes tables to the user cache directory, never into
the installation.

The main risk areas are therefore malformed table files and the optional
dependencies (pandas, torch, pydantic, NumPy, SciPy), whose own advisories are
tracked by Dependabot.
