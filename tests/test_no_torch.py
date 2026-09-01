"""An install without torch never imports it, and never pays for it.

`torch` is a ~200 MB optional extra. The claim that it costs nothing when
absent is worth checking rather than asserting, and checking it needs a
subprocess with torch blocked at the import system -- otherwise the test would
pass for the wrong reason on any machine that happens not to have torch.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import levy

BLOCK_TORCH = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        if name == 'torch' or name.startswith('torch.'):
            return self
    def load_module(self, name):
        raise ImportError('torch is blocked for this test')
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('torch is blocked for this test')
        return None

sys.meta_path.insert(0, _Blocked())
"""


def _run(body):
    """Run `body` in a fresh interpreter with torch blocked."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pathlib.Path(levy.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", BLOCK_TORCH + textwrap.dedent(body)],
        capture_output=True, text=True, env=env,
    )


def test_the_whole_api_works_without_torch():
    result = _run("""
        import numpy as np
        from levy import api

        x = np.linspace(-5.0, 5.0, 101)
        api.pdf(x, alpha=1.5, beta=0.0)
        api.cdf(x, alpha=1.5, beta=0.0)
        api.logpdf(x, alpha=1.5, beta=0.0)
        api.fit(api.rvs(alpha=1.5, beta=0.0, size=200, random_state=0))
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_torch_is_never_imported():
    result = _run("""
        import sys
        import numpy as np
        import levy
        from levy import api, backends

        api.pdf(np.array([1.0]), alpha=1.5, beta=0.0)
        backends.get(None, np.array([1.0]))
        print('torch' in sys.modules)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_importing_the_backend_package_does_not_import_torch():
    # levy.backends is imported by levy.api on every call. If merely importing
    # it pulled torch in, the extra would not be optional in any useful sense.
    result = _run("""
        import sys
        import levy.backends
        print('torch' in sys.modules)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_selecting_torch_fails_with_an_instruction_not_a_traceback():
    result = _run("""
        import levy

        try:
            levy.set_backend('torch')
        except ImportError as error:
            print(error)
        else:
            print('NO ERROR')
    """)
    assert result.returncode == 0, result.stderr
    message = result.stdout.strip()
    assert "torch" in message
    assert 'pip install "pylevy[torch]"' in message, message


def test_the_failure_comes_at_selection_not_at_the_first_evaluation():
    # set_backend loads the backend eagerly so the error arrives where the
    # mistake was made, rather than several calls later.
    result = _run("""
        import numpy as np
        import levy
        from levy import api

        try:
            levy.set_backend('torch')
            print('SELECTED')
        except ImportError:
            print('failed at selection')
        api.pdf(np.array([1.0]), alpha=1.5, beta=0.0)
        print('numpy still works')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[0] == "failed at selection"
    assert "numpy still works" in result.stdout


def test_an_explicit_backend_argument_also_explains_itself():
    result = _run("""
        import numpy as np
        from levy import api

        try:
            api.pdf(np.array([1.0]), alpha=1.5, beta=0.0, backend='torch')
        except ImportError as error:
            print('pylevy[torch]' in str(error))
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_the_1x_surface_works_without_torch():
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
