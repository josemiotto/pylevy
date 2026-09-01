API reference
=============

The 2.0 API
-----------

.. automodule:: levy.api
   :members: pdf, cdf, logpdf, rvs, fit, StableParams, FitResult
   :member-order: bysource

Backends
--------

.. automodule:: levy.backends
   :members: get, set_backend, using

Parametrizations
----------------

.. automodule:: levy.parametrization
   :members: Parameters

Table management
----------------

.. automodule:: levy.tables
   :members: data_dir, user_cache_dir

Grid constants
--------------

.. automodule:: levy.constants

The 1.x API
-----------

These still work and still return the same numbers. Each emits a
``DeprecationWarning`` naming its replacement when reached through ``levy.``;
reaching them through their own modules, as below, does not warn. See
:doc:`migration`.

.. autofunction:: levy.distribution.levy

.. autofunction:: levy.distribution.neglog_levy

.. autofunction:: levy.fitting.fit_levy

.. autofunction:: levy.sampling.random
