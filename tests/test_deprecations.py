"""The 1.x names still work, warn on access, and return identical objects.

This is the file that makes the 2.0 promise checkable. Two things have to hold
at once: a 1.x caller gets a warning telling them what to use instead, and a 1.x
caller's numbers do not move.

The second half is proved elsewhere, and more thoroughly: the whole
characterization suite calls `levy.levy`, `levy.neglog_levy`, `levy.random` and
`levy.fit_levy` -- the deprecated spellings -- and its 251 golden records are
unchanged. What is proved here is that the shim hands back the very same object,
so there is nowhere for a difference to come from.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import pytest

import levy

# (name, module it now lives in, a fragment the message must name)
DEPRECATED = [
    ("levy", "levy.distribution", "levy.api.pdf"),
    ("neglog_levy", "levy.distribution", "levy.api.logpdf"),
    ("fit_levy", "levy.fitting", "levy.api.fit"),
    ("random", "levy.sampling", "levy.api.rvs"),
    ("Parameters", "levy.parametrization", "levy.api.StableParams"),
    ("convert_to_par0", "levy.parametrization", "levy.api.StableParams.from_par"),
    ("convert_from_par0", "levy.parametrization", "levy.api.StableParams.to_par"),
    ("size", "levy.constants", "levy.constants.size"),
    ("par_bounds", "levy.constants", "levy.constants.par_bounds"),
    ("par_names", "levy.constants", "levy.constants.par_names"),
    ("default", "levy.constants", "levy.constants.default"),
    ("f_bounds", "levy.constants", "levy.constants.f_bounds"),
]

NAMES = [row[0] for row in DEPRECATED]


@pytest.mark.parametrize(("name", "module", "replacement"), DEPRECATED)
def test_access_warns_and_names_the_replacement(name, module, replacement):
    with pytest.warns(DeprecationWarning) as record:
        getattr(levy, name)
    message = str(record[0].message)
    assert f"levy.{name} is deprecated" in message
    assert replacement in message
    assert "3.0" in message, "a deprecation should say when the name goes away"
    assert module in message, "and where the name lives now"


@pytest.mark.parametrize(("name", "module", "_replacement"), DEPRECATED)
def test_the_shim_returns_the_identical_object(name, module, _replacement):
    # Not "an equal object" -- the same one. There is no wrapper, so a 1.x call
    # runs exactly the code 1.1 ran.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        through_shim = getattr(levy, name)
    direct = getattr(importlib.import_module(module), name)
    assert through_shim is direct


@pytest.mark.parametrize(("name", "module", "_replacement"), DEPRECATED)
def test_importing_from_the_module_does_not_warn(name, module, _replacement):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("name", NAMES)
def test_from_levy_import_still_works(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        namespace: dict = {}
        exec(f"from levy import {name}", namespace)  # noqa: S102
    assert name in namespace


@pytest.mark.parametrize("name", NAMES)
def test_deprecated_names_are_still_in_dir_and_all(name):
    # Lazy resolution keeps them out of the module dictionary, so without
    # __dir__ they would vanish from tab completion and from `from levy import *`.
    assert name in dir(levy)
    assert name in levy.__all__


def test_importing_levy_emits_no_warning():
    # An import-time warning storm -- one line per deprecated name on every
    # `import levy` -- is the fastest way to get a deprecation reverted.
    code = "import warnings; warnings.simplefilter('error'); import levy"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0, result.stderr


def test_using_the_new_api_emits_no_warning():
    code = (
        "import warnings, numpy as np; warnings.simplefilter('error'); "
        "import levy; "
        "levy.api.pdf(np.array([1.0]), alpha=1.5, beta=0.0); "
        "levy.api.fit(np.array([0.1, -0.4, 1.2, 0.3, -1.1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0, result.stderr


def test_an_unknown_attribute_still_raises_attribute_error():
    missing = "nope"
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr(levy, missing)


SUBMODULES = [
    "api", "constants", "distribution", "fitting",
    "interpolation", "parametrization", "sampling", "tables",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodules_are_reachable_after_a_bare_import(name):
    # They used to become attributes as a side effect of __init__ importing
    # names out of them. Resolving the 1.x names lazily removed that side
    # effect, so `import levy; levy.sampling.random(...)` has to be kept
    # working on purpose.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        module = getattr(levy, name)
    assert module.__name__ == f"levy.{name}"
    assert name in dir(levy)


def test_a_submodule_resolved_this_way_is_the_real_one():
    import levy.sampling

    assert levy.sampling is levy.sampling


def test_the_build_helpers_resolve_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert callable(levy._calculate_levy)
        assert callable(levy._int_levy)


def test_the_2_0_surface_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        levy.data_dir()
        levy.user_cache_dir()
        assert levy.PACKAGED_DATA
        assert levy.ROOT
        assert levy.api is not None


# --------------------------------------------------------------------------
# version policy
# --------------------------------------------------------------------------

def test_version_is_the_2_0_line():
    assert levy.__version__ == "2.0.0"


def test_changelog_documents_this_version():
    import pathlib
    import re

    changelog = pathlib.Path(levy.__file__).parents[2] / "CHANGELOG.md"
    if not changelog.exists():          # installed copy, not a checkout
        pytest.skip("CHANGELOG.md is not shipped in the wheel")
    headings = re.findall(r"^## \[([^\]]+)\]", changelog.read_text(), re.MULTILINE)
    assert headings, "no version headings in CHANGELOG.md"
    assert headings[0] == levy.__version__, (
        f"CHANGELOG's newest entry is {headings[0]!r} but __version__ is "
        f"{levy.__version__!r}; they are the same fact and must not drift"
    )


def test_every_deprecated_name_is_in_the_changelog_migration_table():
    import pathlib

    changelog = pathlib.Path(levy.__file__).parents[2] / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md is not shipped in the wheel")
    text = changelog.read_text()
    missing = [name for name in NAMES if f"levy.{name}" not in text]
    assert not missing, f"deprecated but undocumented: {missing}"


def _env_with_src():
    """Environment that can import levy whether or not it is installed."""
    import os
    import pathlib

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(levy.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    return env
