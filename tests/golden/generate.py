#!/usr/bin/env python
"""Generate (or check) the characterization golden file.

    python tests/golden/generate.py            # rewrite golden_v1.jsonl
    python tests/golden/generate.py --check     # verify the committed file

``--check`` is what CI runs on every matrix leg. It regenerates every case in
memory and compares against the committed file within the suite's tolerances, so
a reviewer can trust ``golden_v1.jsonl`` without reading 250 lines of hex.

``--check --strict-bytes`` additionally requires the deterministic groups
(``levy``, ``convert``, ``random``) to regenerate byte-identically. That is a
real guarantee but only on a fixed platform and dependency set -- ``np.tan`` and
``scipy.special.gamma`` may differ by an ULP across libms -- so it runs in one
pinned job rather than across the whole matrix.

The file is JSONL sorted by case id: one case per line, floats as ``float.hex()``
strings. That keeps it exact, greppable, and reviewable in a diff -- unlike a
``.npz`` blob, which this project already has too many of.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_PATH = HERE / "golden_v1.jsonl"

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from _cases import all_cases  # noqa: E402
from _compare import DETERMINISTIC_GROUPS, matches  # noqa: E402
from _encode import decode, encode  # noqa: E402


def build():
    """Run every case and return {id: {"group": ..., "value": <encoded>}}."""
    out = {}
    for case in all_cases():
        out[case.id] = {"group": case.group, "value": encode(case.run())}
    return out


def serialize(records):
    lines = []
    for case_id in sorted(records):
        entry = {"id": case_id, **records[case_id]}
        lines.append(json.dumps(entry, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def load(path=GOLDEN_PATH):
    """Load the golden file into {id: {"group": ..., "value": <encoded>}}."""
    records = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            records[entry["id"]] = {"group": entry["group"], "value": entry["value"]}
    return records


def check(strict_bytes=False):
    if not GOLDEN_PATH.exists():
        print(f"FAIL: {GOLDEN_PATH} does not exist", file=sys.stderr)
        return 1

    committed = load()
    regenerated = build()

    missing = sorted(set(committed) - set(regenerated))
    extra = sorted(set(regenerated) - set(committed))
    if missing or extra:
        for case_id in missing:
            print(f"FAIL: {case_id}: in golden file but no longer generated", file=sys.stderr)
        for case_id in extra:
            print(f"FAIL: {case_id}: generated but missing from golden file", file=sys.stderr)
        return 1

    failures = 0
    drifted = []
    identical = 0
    for case_id in sorted(committed):
        group = committed[case_id]["group"]
        want = committed[case_id]["value"]
        got = regenerated[case_id]["value"]

        if got == want:
            identical += 1
            continue

        if not matches(group, decode(got), decode(want)):
            print(f"FAIL: {case_id} ({group}) differs beyond tolerance", file=sys.stderr)
            failures += 1
        elif strict_bytes and group in DETERMINISTIC_GROUPS:
            print(
                f"FAIL: {case_id} ({group}) is not byte-reproducible; "
                "this group must regenerate identically under --strict-bytes",
                file=sys.stderr,
            )
            failures += 1
        else:
            drifted.append((group, case_id))

    print(f"checked {len(committed)} cases; {identical} byte-identical")
    if drifted:
        print(
            f"{len(drifted)} case(s) differ in the last digits but are within "
            "tolerance (expected across SciPy/BLAS/libm versions):"
        )
        for group, case_id in drifted:
            print(f"  ~ [{group}] {case_id}")
    if failures:
        print(f"{failures} case(s) FAILED", file=sys.stderr)
        return 1
    print("golden file is consistent with the current code")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed golden file instead of rewriting it",
    )
    parser.add_argument(
        "--strict-bytes",
        action="store_true",
        help="also require deterministic groups to regenerate byte-identically",
    )
    args = parser.parse_args()

    if args.check:
        return check(strict_bytes=args.strict_bytes)

    GOLDEN_PATH.write_text(serialize(build()))
    print(f"wrote {GOLDEN_PATH} ({len(load())} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
