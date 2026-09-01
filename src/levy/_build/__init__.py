"""Offline generation of the lookup tables pylevy interpolates from.

Nothing here is needed to *use* the package. It is imported only by the
``levy-tables`` command and by tests, which is the point: the previous
arrangement kept the quadrature code in ``levy/__init__.py``, so importing the
library pulled in ``scipy.integrate`` and ~90 lines that only a maintainer
regenerating the tables would ever run.

The tables take roughly 25 minutes of CPU to rebuild at the shipped
(200, 76, 101) resolution, and the crossover limits another 30, so the builders
here support parallel execution.
"""

from levy._build.quadrature import calculate_levy, interpolated_levy
from levy._build.tables import (
    build_crossover_tables,
    build_density_tables,
    write_manifest,
)

__all__ = [
    "build_crossover_tables",
    "build_density_tables",
    "calculate_levy",
    "interpolated_levy",
    "write_manifest",
]
