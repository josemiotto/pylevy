"""Exact, text-diffable encoding of numerical results for the golden file.

Every float is stored via :meth:`float.hex`, which round-trips bit-exactly and is
stable across NumPy versions. This matters: the package's original doctests broke
on NumPy 2 for two independent reasons -- last-digit drift in the L-BFGS-B result
*and* the new ``np.float64(...)`` scalar repr. Storing hex strings removes the
repr from the equation entirely, so a golden diff always means the numbers moved.

``float.hex()`` handles ``nan`` and ``inf`` (round-tripping via
:meth:`float.fromhex`), so degenerate results are pinned rather than skipped.
"""

from __future__ import annotations

import numpy as np


def _f(value):
    """Encode a single float as a hex string."""
    return float(value).hex()


def encode(obj):
    """Encode a result into JSON-serializable, bit-exact form."""
    if isinstance(obj, np.ndarray):
        return {
            "t": "array",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "v": [_f(x) for x in obj.ravel(order="C")],
        }
    if isinstance(obj, (bool, np.bool_)):
        return {"t": "bool", "v": bool(obj)}
    if isinstance(obj, (int, np.integer)):
        return {"t": "int", "v": int(obj)}
    if isinstance(obj, (float, np.floating)):
        return {"t": "float", "v": _f(obj)}
    if isinstance(obj, tuple):
        return {"t": "tuple", "v": [encode(o) for o in obj]}
    if isinstance(obj, list):
        return {"t": "list", "v": [encode(o) for o in obj]}
    raise TypeError(f"cannot encode {type(obj).__name__}")


def decode(obj):
    """Inverse of :func:`encode`."""
    kind = obj["t"]
    if kind == "array":
        flat = np.array([float.fromhex(x) for x in obj["v"]], dtype=np.float64)
        return flat.reshape(obj["shape"]).astype(obj["dtype"])
    if kind == "bool":
        return obj["v"]
    if kind == "int":
        return obj["v"]
    if kind == "float":
        return float.fromhex(obj["v"])
    if kind == "tuple":
        return tuple(decode(o) for o in obj["v"])
    if kind == "list":
        return [decode(o) for o in obj["v"]]
    raise TypeError(f"cannot decode kind {kind!r}")
