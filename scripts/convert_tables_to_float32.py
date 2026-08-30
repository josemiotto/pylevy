#!/usr/bin/env python
"""Convert the shipped float64 lookup tables to the float32 layout in levy/data/.

    python scripts/convert_tables_to_float32.py [--dry-run]

This *converts* rather than regenerates, deliberately. Rebuilding by quadrature
would take ~55 CPU-minutes and would bake in whatever SciPy happens to be
installed, changing values for reasons unrelated to storage. Converting keeps
the shipped numbers exactly, to float32 precision, so the only difference is the
rounding this script measures and prints.

It reads the legacy float64 archives from an explicit source directory (the
package directory, where they lived; --source overrides) rather than through
data_dir(), which would resolve to whatever is in use -- a complete user cache,
$LEVY_DATA_DIR, or the very levy/data/ this script writes -- and applies the
load-time repair, so the four cells where quadrature failed are fixed at the
source rather than carried into the new files. Those four are also the only
cells in either table that exceed the float32 maximum, so the repair has to
happen first: a naive .astype(np.float32) would turn a wrong number into inf.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import levy  # noqa: E402
from levy._build.tables import _write_table, write_manifest  # noqa: E402

OUT_DIR = os.path.join(levy.ROOT, "data")
LEGACY_DIR = levy.ROOT


def load_legacy(directory, key):
    """The float64 table `key` from `directory`, repaired the way loading does."""
    return levy._repair_table(key, levy._load_table(directory, key))


def report(name, original):
    converted = original.astype(np.float32)
    back = converted.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(back - original) / np.maximum(np.abs(original), 1e-300)
    finite = np.isfinite(relative)
    print(
        f"  {name:<12} max rel {relative[finite].max():.3e}"
        f"   p99.9 rel {np.percentile(relative[finite], 99.9):.3e}"
        f"   max abs {np.abs(back - original).max():.3e}"
    )
    assert np.isfinite(converted).all(), f"{name} produced non-finite float32 values"
    positive_lost = int(((original > 0) & (converted == 0)).sum())
    assert positive_lost == 0, f"{name}: {positive_lost} positive values underflowed"
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="measure but do not write")
    parser.add_argument("--source", default=LEGACY_DIR,
                        help="directory holding the float64 archives (default: the package "
                             "directory, where the legacy tables lived)")
    args = parser.parse_args()

    for name in ("pdf", "cdf", "lower_limit", "upper_limit"):
        if not os.path.exists(os.path.join(args.source, f"{name}.npz")):
            print(f"no {name}.npz in {args.source}; point --source at the float64 archives",
                  file=sys.stderr)
            return 2
    print(f"reading float64 tables from {args.source}")
    pdf = load_legacy(args.source, "pdf")
    cdf = load_legacy(args.source, "cdf")
    lower = load_legacy(args.source, "lower_limit")
    upper = load_legacy(args.source, "upper_limit")

    assert cdf.max() <= 1.0 + 1e-6, "cdf still holds unusable cells; repair first"

    print("float32 conversion error:")
    pdf32 = report("pdf", pdf)
    cdf32 = report("cdf", cdf)
    lower32 = report("lower_limit", lower)
    upper32 = report("upper_limit", upper)

    if args.dry_run:
        print("dry run; nothing written")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    # Through the builder's atomic writer: a conversion that dies mid-write
    # must not leave a truncated archive that data_dir() then takes for part
    # of a complete set. The manifest is written last, once every table is.
    _write_table(os.path.join(OUT_DIR, "pdf.npz"), pdf32)
    _write_table(os.path.join(OUT_DIR, "cdf.npz"), cdf32)
    # lower_limit.npz and upper_limit.npz become one file with two named
    # arrays. Merging them was proposed on the unmerged `dev` branch (3ab5d8e).
    _write_table(os.path.join(OUT_DIR, "limits.npz"), lower=lower32, upper=upper32)

    write_manifest(
        OUT_DIR,
        extra={"source": "converted from the float64 tables", "dtype": "float32"},
    )

    print(f"\nwrote {OUT_DIR}")
    before = sum(
        os.path.getsize(os.path.join(args.source, f"{n}.npz"))
        for n in ("pdf", "cdf", "lower_limit", "upper_limit")
        if os.path.exists(os.path.join(args.source, f"{n}.npz"))
    )
    after = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
        if f.endswith(".npz")
    )
    if before:
        print(f"  {before / 1e6:.2f} MB -> {after / 1e6:.2f} MB  "
              f"({100 * (1 - after / before):.0f}% smaller)")
    else:
        # `before` is a filtered sum, so it is 0 once the float64 originals are
        # gone -- on a tree where this already ran, or one checked out after the
        # conversion landed. Nothing to compare against.
        print(f"  wrote {after / 1e6:.2f} MB "
              "(no float64 originals here to compare against)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
