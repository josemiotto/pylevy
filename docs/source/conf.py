#
# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The package lives in src/levy. This used to point three levels up, which is
# outside the repository entirely -- the docs only ever built because
# Read the Docs installed the package first, so autodoc imported the installed
# copy and this line did nothing. Pointing it at src/ means `sphinx-build` works
# in a plain checkout, which is what lets CI build the docs on every PR.
sys.path.insert(0, os.path.abspath('../../src'))

from levy import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = 'pylevy'
copyright = '2026, Paul Harrison, José María Miotto and contributors'
author = 'Paul Harrison, José María Miotto and contributors'

# Read from the package rather than repeated here. The two had already drifted:
# this file said 1.1 while the tag said 1.2.
version = '.'.join(__version__.split('.')[:2])
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    # The docstrings are NumPy style; without napoleon, autodoc renders the
    # section underlines as literal text.
    'sphinx.ext.napoleon',
    # So the narrative pages can be Markdown, and CHANGELOG.md can be included
    # rather than duplicated.
    'myst_parser',
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# The 1.x names emit DeprecationWarning on access. autodoc touches every
# documented attribute, and the build runs with -W, so without this a
# deprecation notice would fail the docs build -- documenting a deprecated name
# is exactly what we want to keep doing.
autodoc_default_options = {'show-inheritance': True}

templates_path = ['_templates']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
exclude_patterns = []
pygments_style = 'sphinx'

myst_enable_extensions = ['deflist', 'colon_fence']
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# Furo, rather than alabaster: it has a working dark mode and a sidebar that
# copes with a module of this size. No third-party service is involved -- it is
# an ordinary pip package, so the build is reproducible from this file alone.
html_theme = 'furo'
html_static_path = ['_static']
html_title = f'pylevy {release}'

htmlhelp_basename = 'pylevydoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements: dict = {}
latex_documents = [
    (master_doc, 'pylevy.tex', 'pylevy Documentation', author, 'manual'),
]

man_pages = [(master_doc, 'pylevy', 'pylevy Documentation', [author], 1)]

texinfo_documents = [
    (master_doc, 'pylevy', 'pylevy Documentation', author, 'pylevy',
     'Levy alpha-stable distributions for Python.', 'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# The old mapping used the pre-Sphinx-1.7 two-argument form against
# docs.scipy.org, which has not served numpy's objects.inv for years; every
# cross-reference into NumPy silently failed to resolve.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

nitpicky = False

# pytest's --doctest-modules runs a docstring example in its own module's
# namespace; Sphinx's doctest builder starts from an empty one. This makes the
# two agree, so an example cannot pass under one runner and fail under the
# other. Same approach NumPy takes for its own `np`.
doctest_global_setup = """
import sys
import numpy as np

from levy import api
from levy._compat import have, loaded
from levy.api import FitResult, StableParams, cdf, fit, logpdf, pdf, rvs
from levy.backends import get, set_backend, using
from levy.distribution import levy, neglog_levy
from levy.fitting import fit_levy
from levy.interpolation import _interpolate, _reflect
from levy.parametrization import Parameters
from levy.sampling import random
"""

linkcheck_ignore = [
    # Rejects the checker's requests, and is the canonical citation anyway.
    r'https://doi\.org/.*',
]
linkcheck_timeout = 20
