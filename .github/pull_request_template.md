## What this changes

<!-- One or two sentences. -->

## How it was verified

<!-- Commands run, and what they said. Not "tests pass" -- which tests, and what
     new ones were added. -->

## Did the golden file move?

<!-- Delete whichever does not apply. -->

- [ ] **No.** `tests/golden/golden_v1.jsonl` is untouched.
- [ ] **Yes**, deliberately. Below: which records moved, by how much, and a
      comparison against `levy._build.calculate_levy` showing the new values are
      closer to the truth.

<!-- A pull request that regenerates the goldens without that evidence should be
     rejected however green CI is. See CONTRIBUTING.md. -->

## Checklist

- [ ] A bug fix comes with a regression test that fails before it
- [ ] `ruff check .` and `mypy` pass
- [ ] Docstrings are NumPy style, and any examples run
- [ ] `CHANGELOG.md` updated if the change is user-visible
