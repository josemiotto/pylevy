"""Tests for the table-generation CLI and the data-directory search order.

The builders are exercised at a tiny grid size -- the shipped (200, 76, 101)
tables take about 55 CPU-minutes to regenerate, so the full build is not
something a test suite can run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import levy
import levy.tables
from levy._build.cli import main
from levy._build.tables import build_crossover_tables, build_density_tables, write_manifest

TINY = (24, 10, 13)


# --------------------------------------------------------------------------
# Where tables come from
# --------------------------------------------------------------------------


def test_data_dir_defaults_to_the_packaged_tables(monkeypatch, tmp_path):
    monkeypatch.delenv("LEVY_DATA_DIR", raising=False)
    monkeypatch.setattr(levy.tables, "user_cache_dir", lambda: str(tmp_path / "empty"))
    assert levy.data_dir() == levy.PACKAGED_DATA


def test_data_dir_honours_the_environment_override(monkeypatch):
    monkeypatch.setenv("LEVY_DATA_DIR", "/somewhere/else")
    assert levy.data_dir() == "/somewhere/else"


def test_data_dir_prefers_a_complete_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("LEVY_DATA_DIR", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(levy.tables, "user_cache_dir", lambda: str(cache))

    # An incomplete cache must be ignored, not half-used.
    (cache / "pdf.npz").write_bytes(b"")
    assert levy.data_dir() == levy.PACKAGED_DATA

    # pdf + cdf + the merged limits file is a complete set.
    (cache / "cdf.npz").write_bytes(b"")
    assert levy.data_dir() == levy.PACKAGED_DATA
    (cache / "limits.npz").write_bytes(b"")
    assert levy.data_dir() == str(cache)


def test_data_dir_accepts_the_legacy_split_limit_files(monkeypatch, tmp_path):
    """Tables built before limits.npz existed must still be usable."""
    monkeypatch.delenv("LEVY_DATA_DIR", raising=False)
    cache = tmp_path / "legacy"
    cache.mkdir()
    monkeypatch.setattr(levy.tables, "user_cache_dir", lambda: str(cache))
    for name in ("pdf", "cdf", "lower_limit", "upper_limit"):
        (cache / "{}.npz".format(name)).write_bytes(b"")
    assert levy.data_dir() == str(cache)


def test_writable_data_dir_is_never_the_installed_package(monkeypatch, tmp_path):
    """Building must not write into site-packages, as `python -m levy build` did."""
    monkeypatch.delenv("LEVY_DATA_DIR", raising=False)
    monkeypatch.setattr(levy.tables, "user_cache_dir", lambda: str(tmp_path / "cache"))
    assert levy.data_dir(writable=True) != levy.ROOT


def test_user_cache_dir_is_platform_appropriate():
    path = levy.user_cache_dir()
    assert path.endswith("pylevy")
    assert os.path.isabs(path)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


@pytest.mark.build
def test_build_density_tables_at_a_tiny_grid(tmp_path):
    results = build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf", "cdf"))

    for name in ("pdf", "cdf"):
        table, _ = results[name]
        assert table.shape == TINY
        assert np.isfinite(table).all()
        assert (tmp_path / "{}.npz".format(name)).exists()

    cdf = results["cdf"][0]
    assert cdf.min() >= -1e-6 and cdf.max() <= 1.0 + 1e-6
    # the CDF must increase along x at every (alpha, beta)
    assert np.all(np.diff(cdf, axis=0) > -1e-6)


@pytest.mark.build
def test_generated_table_matches_quadrature(tmp_path):
    """The builder must store what calculate_levy returns, at the right places."""
    results = build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf",))
    table = results["pdf"][0]
    x_axis, alphas, betas = levy._build.tables.grid_axes(TINY)
    ts = np.tan(x_axis)
    for i in (0, 4, 9):
        for j in (0, 6, 12):
            expected = levy._calculate_levy(ts[5], alphas[i], betas[j], False)
            assert table[5, i, j] == pytest.approx(expected, rel=1e-12)


@pytest.mark.build
def test_build_writes_nothing_into_the_installed_package(tmp_path):
    def snapshot():
        # st_mtime_ns and st_size, not getmtime: getmtime is seconds-resolution
        # on some filesystems, so a rewrite inside the same second would leave
        # this test green while the build clobbered the installed package.
        out = {}
        for name in sorted(os.listdir(levy.PACKAGED_DATA)):
            st = os.stat(os.path.join(levy.PACKAGED_DATA, name))
            out[name] = (st.st_mtime_ns, st.st_size)
        return out

    before = snapshot()
    build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf",))
    assert before == snapshot()


@pytest.mark.build
def test_manifest_records_provenance(tmp_path):
    build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf",))
    manifest = write_manifest(str(tmp_path), TINY)

    assert manifest["grid_size"] == list(TINY)
    assert manifest["numpy"] == np.__version__
    assert "pdf" in manifest["tables"]
    entry = manifest["tables"]["pdf"]
    assert entry["shape"] == list(TINY)
    assert len(entry["sha256"]) == 64

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest


@pytest.mark.build
def test_generated_tables_are_usable_via_the_environment_override(tmp_path, monkeypatch):
    """End-to-end: build tiny tables, point levy at them, evaluate."""
    build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf", "cdf"))
    build_crossover_tables(str(tmp_path), TINY, jobs=1,
                           cdf_table=np.load(str(tmp_path / "cdf.npz"))["arr_0"])

    script = (
        "import numpy as np, levy;"
        "print(float(levy.levy(np.array([1.0]), 1.5, 0.0)[0]))"
    )
    # Derived from the package location, not os.getcwd(): pytest can be
    # invoked from anywhere, and after the src/ move the importable root is
    # src/ rather than the repository root.
    env = dict(os.environ, LEVY_DATA_DIR=str(tmp_path),
               PYTHONPATH=os.path.dirname(levy.__path__[0]))
    completed = subprocess.run([sys.executable, "-c", script], env=env,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    # A 24x10x13 grid is far too coarse for accuracy; we are checking the
    # plumbing, so only sanity is asserted.
    assert 0.0 < float(completed.stdout.strip()) < 1.0


# --------------------------------------------------------------------------
# Command-line surface
# --------------------------------------------------------------------------


def test_cli_where_reports_the_search_path(capsys):
    assert main(["where"]) == 0
    output = capsys.readouterr().out
    assert "tables in use" in output
    assert "LEVY_DATA_DIR" in output
    assert "pdf.npz" in output and "cdf.npz" in output
    assert "limits.npz" in output


def test_cli_with_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "levy-tables" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["1,2", "200,76,101,5", "4,4,4"])
def test_cli_rejects_bad_sizes(bad):
    with pytest.raises(SystemExit):
        main(["build", "--size", bad])


def test_cli_rejects_unknown_table_names():
    with pytest.raises(SystemExit):
        main(["build", "--what", "pdf,nonsense"])


@pytest.mark.build
def test_cli_build_writes_to_the_requested_directory(tmp_path):
    assert main(["build", "--out", str(tmp_path), "--size", "24,10,13", "--what", "pdf"]) == 0
    assert (tmp_path / "pdf.npz").exists()
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.build
def test_tables_of_a_different_resolution_are_usable(tmp_path, monkeypatch):
    """Grid indices must come from the loaded table, not the `size` constant.

    `size` is hardcoded to the shipped (200, 76, 101). Before this was fixed,
    pointing LEVY_DATA_DIR at tables built at any other resolution -- the whole
    reason `levy-tables build --size` exists -- raised
    ``IndexError: index 50 is out of bounds for axis 0 with size 10`` from
    inside levy().
    """
    build_density_tables(str(tmp_path), TINY, jobs=1, what=("pdf", "cdf"))
    with np.load(str(tmp_path / "cdf.npz")) as archive:
        cdf_table = archive["arr_0"]
    build_crossover_tables(str(tmp_path), TINY, jobs=1, cdf_table=cdf_table)

    monkeypatch.setenv("LEVY_DATA_DIR", str(tmp_path))
    levy._data_cache.clear()
    try:
        assert levy._grid_shape() == TINY
        assert levy.size != TINY, "the constant still says the shipped resolution"
        result = levy.levy(np.array([0.5, 1.0, 2.0]), 1.5, 0.0)
        assert np.all(np.isfinite(result))
    finally:
        levy._data_cache.clear()


def test_python_dash_m_levy_works(tmp_path):
    """`python -m levy` needs levy/__main__.py, which never existed.

    The module docstring documented `python -m levy build` as the way to
    regenerate the tables, but for a package that requires __main__.py; the
    `if __name__ == "__main__"` block in __init__.py is only reachable by
    running the file directly. Verified against master: it failed there too.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "levy", "where"],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONPATH=os.path.dirname(levy.__path__[0])),
    )
    assert completed.returncode == 0, completed.stderr
    assert "tables in use" in completed.stdout
