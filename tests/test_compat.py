"""levy._compat: optional dependencies are looked up, never imported blindly."""

from __future__ import annotations

import pytest

from levy._compat import require


def test_require_names_the_extra_when_the_module_is_not_installed():
    expected = r'not installed.*pip install "pylevy\[no_such_module_xyz\]"'
    with pytest.raises(ImportError, match=expected):
        require("no_such_module_xyz")


def test_require_does_not_blame_the_install_when_the_module_is_broken(tmp_path, monkeypatch):
    """A module that is present but fails to import used to be reported as
    "not installed", with an install command that would have changed nothing.
    """
    package = tmp_path / "brokenpkg_for_levy"
    package.mkdir()
    (package / "__init__.py").write_text("import a_dependency_that_is_missing\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ImportError, match="installed but failed to import") as info:
        require("brokenpkg_for_levy")
    assert "a_dependency_that_is_missing" in str(info.value)
    assert "not installed" not in str(info.value)
