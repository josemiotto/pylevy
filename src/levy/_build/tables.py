# -*- encoding: utf-8 -*-
"""Builders for the four lookup tables, with provenance and validation.

Differences from the previous ``_make_dist_data_file`` / ``_make_limit_data_files``:

* they take an output directory instead of writing into the installed package;
* they can run across processes, because a full rebuild is ~25 CPU-minutes for
  the densities and ~30 more for the crossover limits;
* they validate what quadrature returns instead of storing it blindly, which is
  how four unusable cells ended up in the shipped cdf.npz;
* they write a manifest recording how the tables were made.
"""

import hashlib
import json
import logging
import os
import sys

import numpy as np

from levy import _approximate, _lower, _upper, size
from levy._build.quadrature import calculate_levy, interpolated_levy

logger = logging.getLogger(__name__)

__all__ = [
    "build_crossover_tables",
    "build_density_tables",
    "grid_axes",
    "write_manifest",
]

#: A CDF outside this band, or a non-finite density, means quadrature failed.
_CDF_BOUND = 1e-6


def grid_axes(grid_size=None):
    """ The x, alpha and beta axes of the lookup grid.

    x is stored in arctan space, so the grid covers the whole real line; the
    tables hold values at ``tan(x_axis)``.
    """
    grid_size = tuple(grid_size or size)
    return [
        np.linspace(_lower[i], _upper[i], grid_size[i], endpoint=True)
        for i in range(3)
    ]


def _density_column(args):
    """One (alpha, beta) column of the table. Top level, so it can be pickled."""
    alpha, beta, ts, cdf = args
    return np.array([calculate_levy(t, alpha, beta, cdf) for t in ts])


def _validate_column(column, cdf, alpha, beta):
    """Flag values quadrature could not compute, rather than storing them."""
    if cdf:
        bad = ~np.isfinite(column) | (column < -_CDF_BOUND) | (column > 1.0 + _CDF_BOUND)
    else:
        bad = ~np.isfinite(column)
    if bad.any():
        good = ~bad
        kind = "cdf" if cdf else "pdf"
        # Two good points are the minimum np.interp needs. Below that the
        # column cannot be filled at all, and saying "filling by
        # interpolation" would be a lie: the bad values are written as they
        # are. Say which of the two actually happened.
        if good.sum() >= 2:
            logger.warning(
                "quadrature failed for %d of %d points at alpha=%.4f beta=%.4f (%s); "
                "filling by interpolation along x",
                int(bad.sum()), column.size, alpha, beta, kind,
            )
            column = column.copy()
            column[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), column[good])
        else:
            logger.error(
                "quadrature failed for %d of %d points at alpha=%.4f beta=%.4f (%s) "
                "with fewer than 2 usable points; the column is UNFILLED and the "
                "table it is written into is not usable",
                int(bad.sum()), column.size, alpha, beta, kind,
            )
    return column, int(bad.sum())


def _map(function, items, jobs):
    if jobs and jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            for result in pool.imap(function, items):
                yield result
    else:
        for item in items:
            yield function(item)


def build_density_tables(out_dir, grid_size=None, jobs=1, what=("pdf", "cdf")):
    """ Generate pdf.npz and/or cdf.npz into `out_dir`.

    Returns {name: (array, number_of_repaired_cells)}.
    """
    grid_size = tuple(grid_size or size)
    x_axis, alphas, betas = grid_axes(grid_size)
    ts = np.tan(x_axis)
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name in what:
        cdf = name == "cdf"
        logger.info("Generating %s.npz at %s ...", name, "x".join(map(str, grid_size)))
        table = np.zeros(grid_size, 'float64')
        columns = [(alpha, beta, ts, cdf) for alpha in alphas for beta in betas]
        repaired = 0
        for index, column in enumerate(_map(_density_column, columns, jobs)):
            i, j = divmod(index, len(betas))
            column, bad = _validate_column(column, cdf, alphas[i], betas[j])
            repaired += bad
            table[:, i, j] = column
            if index % 500 == 0:
                logger.info("  %d/%d columns", index, len(columns))
        np.savez_compressed(os.path.join(out_dir, '{}.npz'.format(name)), table)
        logger.info("Wrote %s.npz (%d cell(s) repaired)", name, repaired)
        results[name] = (table, repaired)
    return results


def _crossover_cell(args):
    """Where the power-law asymptote best matches the interpolated CDF."""
    alpha, beta, upper, table = args
    n = 100000
    x1, x2 = -50.0, 1e4 - 50.0
    li1, li2 = 10, 500
    if upper is False:
        x1, x2 = -1e4 + 50, 50
        li1, li2 = -500, -10
    dx = (x2 - x1) / n
    x = np.linspace(x1, x2, num=n + 1, endpoint=True)
    y = 1.0 - interpolated_levy(x, alpha, beta, cdf=True, table=table)
    z = 1.0 - _approximate(x, alpha, beta, cdf=True)
    mask = (li1 < x) & (x < li2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return li1 + dx * np.argmin((np.log(z[mask]) - np.log(y[mask])) ** 2.0)


def build_crossover_tables(out_dir, grid_size=None, jobs=1, cdf_table=None):
    """ Generate lower_limit.npz and upper_limit.npz into `out_dir`.

    These say where levy() should stop interpolating and switch to the
    power-law tail, so they depend on the CDF table and must be rebuilt after it.
    """
    from levy import _read_from_cache

    grid_size = tuple(grid_size or size)
    _, alphas, betas = grid_axes(grid_size)
    os.makedirs(out_dir, exist_ok=True)
    if cdf_table is None:
        # The crossover is a property of a particular CDF table: it says where
        # that table stops matching the tail approximation. Reading the
        # installed one would describe a table other than the one being written
        # here, which is wrong whenever the two differ -- a different grid size,
        # or simply a rebuild. Prefer the CDF just produced in out_dir.
        built = os.path.join(out_dir, 'cdf.npz')
        if os.path.exists(built):
            with np.load(built) as archive:
                cdf_table = archive[archive.files[0]]
        else:
            cdf_table = _read_from_cache('cdf')

    results = {}
    for upper in (True, False):
        name = 'upper' if upper else 'lower'
        logger.info("Generating %s_limit.npz ...", name)
        cells = [(alpha, beta, upper, cdf_table) for alpha in alphas for beta in betas]
        limits = np.zeros(grid_size[1:], 'float64')
        for index, value in enumerate(_map(_crossover_cell, cells, jobs)):
            i, j = divmod(index, len(betas))
            limits[i, j] = value
            if index % 500 == 0:
                logger.info("  %d/%d cells", index, len(cells))
        np.savez_compressed(os.path.join(out_dir, '{}_limit.npz'.format(name)), limits)
        results['{}_limit'.format(name)] = limits
    return results


def write_manifest(out_dir, grid_size=None, extra=None):
    """ Record how the tables in `out_dir` were produced.

    Without this there is no way to tell what resolution a table was built at,
    which library versions produced it, or whether a file has been altered.
    """
    import scipy

    grid_size = tuple(grid_size or size)
    entries = {}
    # Both layouts: the two separate crossover files, and the merged
    # limits.npz that replaces them. Missing names are skipped below, so
    # listing all of them keeps the manifest correct across the change.
    for name in ('pdf', 'cdf', 'lower_limit', 'upper_limit', 'limits'):
        path = os.path.join(out_dir, '{}.npz'.format(name))
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        with np.load(path) as archive:
            key = archive.files[0]
            entries[name] = {
                'sha256': digest,
                'bytes': os.path.getsize(path),
                'shape': list(archive[key].shape),
                'dtype': str(archive[key].dtype),
            }

    manifest = {
        'grid_size': list(grid_size),
        'lower': list(map(float, _lower)),
        'upper': list(map(float, _upper)),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'python': sys.version.split()[0],
        'tables': entries,
    }
    manifest.update(extra or {})
    path = os.path.join(out_dir, 'manifest.json')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
