"""An install without pandas never imports it, and never pays for it.

`pandas` is an optional extra. The tests here run in a subprocess with pandas
blocked at the import system, so they hold whether or not pandas happens to be
installed in the environment running the suite -- which is the only way to
check this claim on a developer machine that has pandas for other reasons.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import levy

# Refuses to import pandas, or anything under it, no matter who asks.
BLOCK_PANDAS = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        if name == 'pandas' or name.startswith('pandas.'):
            return self
    def load_module(self, name):
        raise ImportError('pandas is blocked for this test')
    def find_spec(self, name, path=None, target=None):
        if name == 'pandas' or name.startswith('pandas.'):
            raise ImportError('pandas is blocked for this test')
        return None

sys.meta_path.insert(0, _Blocked())
"""


def _run(body):
    """Run `body` in a fresh interpreter with pandas blocked."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(levy.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", BLOCK_PANDAS + textwrap.dedent(body)],
        capture_output=True, text=True, env=env,
    )


def test_the_whole_api_works_without_pandas():
    result = _run("""
        import numpy as np
        from levy import api

        x = np.linspace(-5.0, 5.0, 101)
        api.pdf(x, alpha=1.5, beta=0.0)
        api.cdf(x, alpha=1.5, beta=0.0)
        api.logpdf(x, alpha=1.5, beta=0.0)
        sample = api.rvs(alpha=1.5, beta=0.0, size=200, random_state=0)
        api.fit(sample)
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_pandas_is_never_imported():
    result = _run("""
        import sys
        import numpy as np
        from levy import api

        api.pdf(np.array([1.0]), alpha=1.5, beta=0.0)
        api.fit(api.rvs(alpha=1.5, beta=0.0, size=100, random_state=0))
        print('pandas' in sys.modules)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_the_1x_surface_works_without_pandas():
    result = _run("""
        import warnings
        import numpy as np
        import levy

        warnings.simplefilter('ignore', DeprecationWarning)
        levy.levy(np.array([1.0]), 1.5, 0.0)
        levy.fit_levy(levy.random(1.5, 0.0, shape=(100,)))
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_to_series_fails_with_an_instruction_not_a_traceback():
    result = _run("""
        from levy import api

        sample = api.rvs(alpha=1.5, beta=0.0, size=100, random_state=0)
        try:
            api.fit(sample).to_series()
        except ImportError as error:
            print(error)
        else:
            print('NO ERROR')
    """)
    assert result.returncode == 0, result.stderr
    message = result.stdout.strip()
    assert "pandas" in message
    assert 'pip install "pylevy[pandas]"' in message, message


def test_labels_of_is_a_dictionary_lookup_when_pandas_is_absent():
    # The fast path has to stay fast: no import attempt, no exception handling,
    # just a miss in sys.modules.
    from levy._pandas import labels_of

    if "pandas" in sys.modules:
        pytest.skip("pandas is imported in this process; covered in a subprocess above")
    assert labels_of(object()) is None
