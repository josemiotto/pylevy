"""Characterization tests: pin the current numerical output of the package.

These tests assert nothing about whether pylevy is *correct*. They assert that
it still produces what it produced when the golden file was generated. That is
the point: the package ships a 12 MB interpolation table and a maximum-likelihood
fitter with no test coverage at all, so any refactor -- moving code between
modules, changing the table dtype, touching the interpolator -- currently has no
way to demonstrate it did not move the math.

If one of these fails, either the change is a regression, or it is a deliberate
numerical improvement. In the second case, regenerate the golden file with

    python tests/golden/generate.py

and justify the diff in the pull request. Do not regenerate to make CI green.
"""

from __future__ import annotations

import pytest

from _cases import all_cases
from _compare import assert_matches
from _encode import decode

CASES = all_cases()


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.id, marks=[pytest.mark.slow] if c.slow else []) for c in CASES],
)
def test_matches_golden(case, golden):
    assert case.id in golden, (
        f"case {case.id!r} has no golden entry; "
        "run `python tests/golden/generate.py` to add it"
    )
    record = golden[case.id]
    assert record["group"] == case.group
    assert_matches(case.group, case.run(), decode(record["value"]), case.id)


def test_golden_file_has_no_stale_entries(golden):
    """Every golden entry corresponds to a live case."""
    stale = sorted(set(golden) - {c.id for c in CASES})
    assert not stale, f"golden file has entries for cases that no longer exist: {stale}"


def test_case_count_is_stable(golden):
    """Guard against a refactor silently dropping coverage."""
    assert len(CASES) == len(golden)
    assert len(CASES) >= 245, "characterization coverage shrank unexpectedly"
