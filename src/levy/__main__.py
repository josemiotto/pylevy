"""Entry point for ``python -m levy``.

This file did not exist before, which means ``python -m levy build`` -- the
command the module docstring gave for regenerating the lookup tables -- never
actually worked: for a package, ``-m`` requires ``__main__.py``, and the
``if __name__ == "__main__"`` block in ``__init__.py`` is only reachable by
running that file directly (``python levy/__init__.py build``).

Prefer the ``levy-tables`` console script; this exists so the documented
invocation does what it says.
"""

import sys

from levy._build.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    # `python -m levy build` used to be the documented spelling; keep it working.
    if argv and argv[0] == "build":
        sys.exit(main(argv))
    sys.exit(main(argv or ["where"]))
