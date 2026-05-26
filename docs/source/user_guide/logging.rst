.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _logging:

=======
Logging
=======

OpenSTEF uses Python's standard :mod:`logging` library and attaches no handlers
by default. You stay in control: configure logging in your application before
importing OpenSTEF modules and choose your own format, destinations, and levels.

This page covers the logger naming convention, what each level emits, and
common configuration patterns for development and production.

The Default Behavior
====================

Each top-level OpenSTEF package installs a :class:`logging.NullHandler` on its
root logger at import time. Until you configure logging yourself, OpenSTEF
produces no output, not even uncaught warnings. This is the standard practice
for Python libraries: the application, not the library, decides where logs go.

To start seeing output, configure logging once at the start of your script,
notebook, or service entry point:

.. code-block:: python

   import logging

   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
   )

   # Subsequent imports will produce log output
   from openstef_models.presets import create_forecasting_workflow

Logger Naming
=============

OpenSTEF loggers follow Python's hierarchical naming, with one logger per
module. The three top-level prefixes correspond to the three packages:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Logger prefix
     - What it covers
   * - ``openstef_core.*``
     - Dataset construction, validation, type coercion
   * - ``openstef_models.*``
     - Model fitting and prediction, transform pipelines, workflow callbacks, MLflow integration
   * - ``openstef_beam.*``
     - Backtesting, evaluation, benchmarking, analysis pipelines

Because loggers are hierarchical, you can configure each subsystem independently.
For example, ``logging.getLogger("openstef_beam").setLevel(logging.WARNING)``
quiets only the backtesting/evaluation chatter, leaving model training logs at
their previous level.

What You'll See at Each Level
=============================

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Level
     - Typical messages
   * - ``INFO``
     - Workflow lifecycle events. *"Created MLflow run X for model Y."* *"Stored trained model for run Z."* *"Skipping model fitting: <reason>."*
   * - ``WARNING``
     - Degraded but recoverable conditions. *"No validation metrics found in fit results. Skipping performance evaluation."* Missing optional columns, dropped empty features, weight-normalization edge cases.
   * - ``ERROR``
     - Unrecoverable failures that prevent a stage from completing.
   * - ``DEBUG``
     - Verbose internal state. Off by default; enable per-subsystem only when investigating a specific issue.

OpenSTEF is conservative about ``INFO`` chatter: a successful ``fit()``
followed by ``predict()`` typically emits only a handful of lines. If a
subsystem feels noisy, raise its level rather than tuning the global root.

Common Configurations
=====================

Minimal setup
-------------

For most local development and one-off scripts:

.. code-block:: python

   import logging

   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S",
   )

Quieting noisy subsystems
-------------------------

When you want to keep training logs but silence backtest iteration progress,
configure the package-level loggers after ``basicConfig``:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)

   # Keep openstef_models at INFO, drop openstef_beam to WARNING
   logging.getLogger("openstef_beam").setLevel(logging.WARNING)

   # Or drill in further: only the MLflow callback at DEBUG
   logging.getLogger(
       "openstef_models.integrations.mlflow.mlflow_storage_callback"
   ).setLevel(logging.DEBUG)

Notebooks
---------

In Jupyter, ``basicConfig`` is sometimes a no-op because the kernel has already
attached a handler to the root logger. Pass ``force=True`` to replace it:

.. code-block:: python

   import logging

   logging.basicConfig(
       level=logging.INFO,
       format="%(name)-25s | %(levelname)-8s | %(message)s",
       force=True,
   )

Production: structured logs
---------------------------

OpenSTEF emits unstructured log records via stdlib logging: no JSON fields,
no contextvars, no opinions about your log shipping pipeline. If you want
structured output (e.g., one JSON object per line, fields parsed by your log
aggregator), wrap the root logger using a structured-logging library.

`structlog <https://www.structlog.org/>`_ is a well-maintained option that
integrates cleanly with stdlib logging. Use its
``structlog.stdlib.LoggerFactory`` adapter so OpenSTEF's loggers flow through
structlog's processors:

.. code-block:: python

   import logging
   import structlog

   structlog.configure(
       processors=[
           structlog.stdlib.add_logger_name,
           structlog.stdlib.add_log_level,
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.processors.JSONRenderer(),
       ],
       logger_factory=structlog.stdlib.LoggerFactory(),
       cache_logger_on_first_use=True,
   )
   logging.basicConfig(format="%(message)s", level=logging.INFO)

OpenSTEF does not require any specific structured-logging library; adapt the
standard logger hierarchy described above to whatever your stack uses.

Troubleshooting
===============

No log output appearing
-----------------------

If you call OpenSTEF functions and see nothing:

1. Check that you actually called :func:`logging.basicConfig` (or set up handlers
   another way) *before* the OpenSTEF call.
2. In notebooks, retry with ``logging.basicConfig(..., force=True)`` to override
   the kernel's preconfigured root handler.
3. Inspect the logger state directly:

   .. code-block:: python

      import logging

      logger = logging.getLogger("openstef_models")
      print("level:", logger.level, "effective:", logger.getEffectiveLevel())
      print("handlers:", logger.handlers)
      print("root handlers:", logging.getLogger().handlers)

Too much log output
-------------------

OpenSTEF subsystems can be quieted individually:

.. code-block:: python

   import logging

   logging.getLogger("openstef_models").setLevel(logging.WARNING)
   logging.getLogger("openstef_beam").setLevel(logging.WARNING)

If a specific module is the culprit (e.g., a transform that warns on every
batch), set ``WARNING`` or ``ERROR`` on that module's logger by its full
dotted name.

.. seealso::

   - `Python logging documentation <https://docs.python.org/3/library/logging.html>`_
   - `structlog standard-library integration <https://www.structlog.org/en/stable/standard-library.html>`_
   - :doc:`/user_guide/guides/deployment` for callback-based monitoring hooks
     in production deployments.
