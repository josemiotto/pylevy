pylevy
======

Levy alpha-stable distributions for Python: density, distribution function,
sampling, and maximum-likelihood fitting.

Direct computation of a stable density requires a lengthy numerical
integration. pylevy interpolates a precomputed table instead, which is what
makes fitting by maximum likelihood fast enough to be practical.
:doc:`how_it_works` explains how, and where the accuracy comes from.

.. image:: _static/levy_distributions.png
   :width: 600
   :alt: Levy stable densities for several values of alpha and beta

Install
-------

.. code-block:: console

   pip install pylevy

Optional extras: ``pylevy[pandas]`` for labelled input and output,
``pylevy[torch]`` for a differentiable backend. Neither is imported unless it
is installed and used.

At a glance
-----------

.. code-block:: python

   import numpy as np
   from levy import api

   x = np.array([-1.0, 0.0, 1.0])

   api.pdf(x, alpha=1.5, beta=0.0)
   api.cdf(x, alpha=1.5, beta=0.0)

   sample = api.rvs(alpha=1.5, beta=0.0, size=1000, random_state=0)
   result = api.fit(sample)
   result.params

Parameters are validated where you write them, so ``alpha=0.2`` raises rather
than quietly returning a wrong number from a clamped table index.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   how_it_works
   api
   migration
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
