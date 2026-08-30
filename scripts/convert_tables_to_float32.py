#!/usr/bin/env python
"""Convert the shipped float64 lookup tables to the float32 layout in levy/data/.

    python scripts/convert_tables_to_float32.py [--dry-run]

This *converts* rather than regenerates, deliberately. Rebuilding by quadrature
would take ~55 CPU-minutes and would bake in whatever SciPy happens to be
installed, changing values for reasons unrelated to storage. Converting keeps
the shipped numbers exactly, to float32 precision, so the only difference is the
rounding this script measures and prints.

It reads the tables through levy._read_from_cache, which applies the load-time
repair, so the four cells where quadrature failed are fixed at the source rather
than carried into the new files. Those four are also the only cells in either
table that exceed the float32 maximum, so the repair has to happen first: a
naive .astype(np.float32) would turn a wrong number into inf.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import levy  # noqa: E402
from levy._build.tables import write_manifest  # noqa: E402

OUT_DIR = os.path.join(levy.ROOT, "data")


def report(name, original):
    converted = original.astype(np.float32)
    back = converted.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(back - original) / np.maximum(np.abs(original), 1e-300)
    finite = np.isfinite(relative)
    print(
        "  {:<12} max rel {:.3e}   p99.9 rel {:.3e}   max abs {:.3e}".format(
            name,
            relative[finite].max(),
            np.percentile(relative[finite], 99.9),
            np.abs(back - original).max(),
        )
    )
    assert np.isfinite(converted).all(), "{} produced non-finite float32 values".format(name)
    positive_lost = int(((original > 0) & (converted == 0)).sum())
    assert positive_lost == 0, "{}: {} positive values underflowed".format(name, positive_lost)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="measure but do not write")
    args = parser.parse_args()

    print("reading tables from {}".format(levy.data_dir()))
    pdf = levy._read_from_cache("pdf")
    cdf = levy._read_from_cache("cdf")
    lower = levy._read_from_cache("lower_limit")
    upper = levy._read_from_cache("upper_limit")

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
    np.savez_compressed(os.path.join(OUT_DIR, "pdf.npz"), pdf32)
    np.savez_compressed(os.path.join(OUT_DIR, "cdf.npz"), cdf32)
    # lower_limit.npz and upper_limit.npz become one file with two named
    # arrays. Merging them was proposed on the unmerged `dev` branch (3ab5d8e).
    np.savez_compressed(os.path.join(OUT_DIR, "limits.npz"), lower=lower32, upper=upper32)

    write_manifest(OUT_DIR, extra={"source": "converted from the float64 tables", "dtype": "float32"})

    print("\nwrote {}".format(OUT_DIR))
    before = sum(
        os.path.getsize(os.path.join(levy.ROOT, "{}.npz".format(n)))
        for n in ("pdf", "cdf", "lower_limit", "upper_limit")
        if os.path.exists(os.path.join(levy.ROOT, "{}.npz".format(n)))
    )
    after = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
        if f.endswith(".npz")
    )
    if before:
        print("  {:.2f} MB -> {:.2f} MB  ({:.0f}% smaller)".format(
            before / 1e6, after / 1e6, 100 * (1 - after / before)))
    else:
        # `before` is a filtered sum, so it is 0 once the float64 originals
        # are gone -- on a tree where this already ran, or one checked out
        # after the conversion landed. Nothing to compare against.
        print("  wrote {:.2f} MB (no float64 originals here to compare "
              "against)".format(after / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
