#!/usr/bin/env python
"""Compare two installed copies of pylevy over a large random grid.

    python scripts/compare_installs.py /path/to/venv-a/bin/python /path/to/venv-b/bin/python

Each interpreter is asked to evaluate levy(), neglog_levy() and random() at the
same points and print the results as exact hex floats; this process then diffs
them. Used to demonstrate that a refactor which moves code without intending to
change numbers really did not change any.

Exit status is 0 when every value matches bit for bit.
"""

import argparse
import json
import subprocess
import sys

PROBE = r"""
import json, sys
import numpy as np
import levy

rng = np.random.RandomState(12345)
out = {}

alphas = rng.uniform(0.5, 2.0, 40)
betas = rng.uniform(-1.0, 1.0, 40)
xs = np.concatenate([
    np.linspace(-5000.0, 5000.0, 401),
    np.linspace(-50.0, 50.0, 401),
])

values = []
for alpha, beta in zip(alphas, betas):
    for cdf in (False, True):
        values.append(levy.levy(xs, alpha, beta, cdf=cdf))
    values.append(levy.neglog_levy(xs, alpha, beta, 0.0, 1.0))
for seed in (0, 1, 2):
    for alpha, beta in ((1.5, 0.0), (0.7, -1.0), (1.0, 0.5), (2.0, 0.3)):
        np.random.seed(seed)
        values.append(levy.random(alpha, beta, 0.0, 1.0, shape=(256,)))

flat = np.concatenate([np.ravel(v) for v in values])
out["count"] = int(flat.size)
out["hex"] = [float(v).hex() for v in flat]
out["version"] = levy.__version__
json.dump(out, sys.stdout)
"""


def probe(interpreter):
    completed = subprocess.run(
        [interpreter, "-c", PROBE], capture_output=True, text=True, timeout=600
    )
    if completed.returncode != 0:
        raise SystemExit("{} failed:\n{}".format(interpreter, completed.stderr))
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interpreter_a")
    parser.add_argument("interpreter_b")
    args = parser.parse_args()

    a = probe(args.interpreter_a)
    b = probe(args.interpreter_b)

    print("A: pylevy {} ({} values)".format(a["version"], a["count"]))
    print("B: pylevy {} ({} values)".format(b["version"], b["count"]))

    if a["count"] != b["count"]:
        print("FAIL: different number of values")
        return 1

    mismatches = [i for i, (x, y) in enumerate(zip(a["hex"], b["hex"])) if x != y]
    if not mismatches:
        print("identical: all {} values match bit for bit".format(a["count"]))
        return 0

    print("FAIL: {} of {} values differ".format(len(mismatches), a["count"]))
    for index in mismatches[:10]:
        print("  [{}] {} != {}".format(index, a["hex"][index], b["hex"][index]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
