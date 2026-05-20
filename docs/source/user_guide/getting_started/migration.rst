Migrating from OpenSTEF 3
=========================

This page guides users of the legacy ``openstef`` v3 package through the conceptual
and practical changes in OpenSTEF v4. If you are starting fresh, skip this page and
begin with :doc:`/user_guide/getting_started/installation`.

.. note::

   V4 is a ground-up redesign. While the forecasting *goals* are the same, the API
   surface is intentionally different. Plan for a rewrite of integration code, not a
   find-and-replace.

Why the Split Happened
----------------------

V3 shipped as a single ``openstef`` package that mixed ML logic, database access
(``openstef-dbc``), orchestration ("tasks"), and data types in one repository. This
created tight coupling: you couldn't use the models without pulling in database
opinions, and testing required standing up infrastructure.

V4 separates concerns into focused packages:

.. list-table:: Package Mapping
   :header-rows: 1
   :widths: 30 30 40

   * - V3
     - V4
     - Responsibility
   * - ``openstef`` (core ML)
     - ``openstef-models``
     - Model training, prediction, feature engineering
   * - ``openstef`` (data classes)
     - ``openstef-core``
     - Types, datasets, validation, testing utilities
   * - ``openstef`` (evaluation)
     - ``openstef-beam``
     - Backtesting, evaluation, experiment tracking
   * - (not available)
     - ``openstef-meta``
     - Ensemble forecasting, metalearning
   * - ``openstef-dbc``
     - *No equivalent*
     - See :ref:`migration_dbc` below

.. mermaid:: /diagrams/user_guide/getting_started/migration_diagram_1.mmd

Configuration: Dicts → Pydantic Models
---------------------------------------

In v3, a "prediction job" was a plain dictionary (or a thin dataclass wrapper) with
loosely typed fields. Typos in keys were silent bugs, and required fields were only
discovered at runtime deep in a pipeline.

**Before (V3):**

.. code-block:: python

   pj = {"id": 287, "model": "xgb", "quantiles": [10, 30, 50, 70, 90],
         "forecast_type": "demand", "resolution_minutes": 15}
   pj = PredictionJobDataClass(**pj)

**After (V4):**

.. code-block:: python

   from openstef_models.presets import ForecastingWorkflowConfig
   config = ForecastingWorkflowConfig(model_id="loc_287", model="xgboost", ...)

Key differences:

- :class:`~openstef_models.presets.ForecastingWorkflowConfig` (the v4 successor to
  ``PredictionJobDataClass``) is a full Pydantic model with validation, defaults, and
  type checking at construction time.
- Fields use explicit types (``Literal["xgboost", "lgbm", ...]``) instead of free-form
  strings.
- Location, horizon, and feature settings are structured sub-objects rather than flat
  keys.

See :doc:`/user_guide/concepts/configuration` for the full configuration model.

Data: DataFrames → TimeSeriesDataset
-------------------------------------

V3 passed raw ``pandas.DataFrame`` objects between pipeline stages. Metadata like
sample interval, availability windows, and column roles lived *outside* the data — in
the prediction job dict or as implicit conventions.

V4 introduces :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`,
which carries the data **and** its metadata together:

- **Sample interval** — validated on construction; no more silent resampling bugs.
- **Availability windows** — tracks when each observation became available (critical
  for correct backtesting without lookahead).
- **Versioning** — supports horizon-aware or ``available_at``-aware slicing.

**Before (V3):**

.. code-block:: python

   input_data = pd.read_csv("data.csv", index_col="index", parse_dates=True)
   train_model_pipeline(pj, input_data)

**After (V4):**

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset
   dataset = TimeSeriesDataset.from_csv("data.csv", sample_interval=timedelta(minutes=15))

The dataset is then passed to workflows which can introspect its properties without
external configuration. See :doc:`/user_guide/concepts/datasets` for details.

Pipelines/Tasks → Workflows/Presets
------------------------------------

V3 had two abstraction layers:

- **Tasks** — fetched data from a database, ran a pipeline, wrote results back.
  Provided by ``openstef-dbc``.
- **Pipelines** — pure ML logic (``train_model_pipeline``, ``create_forecast``).

V4 replaces both with a single concept: **Workflows**.

.. list-table:: API Mapping
   :header-rows: 1
   :widths: 40 40

   * - V3
     - V4
   * - ``train_model_pipeline(pj, data)``
     - ``workflow.train(dataset)``
   * - ``create_forecast(pj, data)``
     - ``workflow.predict(dataset)``
   * - Task (fetch → pipeline → store)
     - User code + workflow (you own I/O)

A **Preset** is a factory that builds a fully configured workflow from a config object:

.. code-block:: python

   from openstef_models.presets import create_forecasting_workflow, ForecastingWorkflowConfig
   workflow = create_forecasting_workflow(config)

:func:`~openstef_models.presets.create_forecasting_workflow` assembles preprocessing,
the forecaster, postprocessing, and callbacks into a
:class:`~openstef_models.presets.forecasting_workflow.CustomForecastingWorkflow`.

For ensemble approaches, ``openstef-meta`` provides
:func:`~openstef_meta.presets.forecasting_workflow.create_ensemble_forecasting_workflow`.

See :doc:`/user_guide/guides/forecasting` for a complete walkthrough.

.. _migration_dbc:

Database Connector (openstef-dbc)
---------------------------------

``openstef-dbc`` provided MySQL/InfluxDB integration: fetching prediction jobs,
reading time series, writing forecasts. **V4 has no equivalent package.** This is
intentional — v4 is a pure ML library with no opinions on data storage.

If your v3 code relied on ``openstef-dbc``:

1. **Data ingestion** — write your own adapter that loads data into
   :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`.
2. **Result storage** — extract forecasts from the workflow output and write them to
   your database of choice.
3. **Configuration storage** — serialize
   :class:`~openstef_models.presets.ForecastingWorkflowConfig` to/from your config
   store (JSON, YAML, database row).

See :doc:`/user_guide/deployment/index` for integration patterns and reference
architectures.

.. warning::

   There is no automated migration path for ``openstef-dbc`` schemas. The v3 database
   tables (``prediction_jobs``, ``predictions_systems``, etc.) are specific to the v3
   architecture.

Model Type Names
----------------

Some model identifiers changed for clarity:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - V3
     - V4
   * - ``"xgb"``
     - ``"xgboost"``
   * - ``"xgb_quantile"``
     - ``"xgboost"`` (quantiles configured separately)
   * - ``"lgb"``
     - ``"lgbm"``

Quantile configuration is now part of the workflow config rather than the model name.

Migration Checklist
-------------------

- ☐ Replace ``openstef`` import with ``openstef-models``, ``openstef-core``, and
  (if needed) ``openstef-beam`` / ``openstef-meta``.
- ☐ Convert prediction job dicts to :class:`~openstef_models.presets.ForecastingWorkflowConfig`.
- ☐ Wrap input DataFrames in :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`.
- ☐ Replace ``train_model_pipeline`` / ``create_forecast`` calls with workflow
  ``.train()`` / ``.predict()``.
- ☐ Remove ``openstef-dbc`` dependency; implement your own data I/O layer.
- ☐ Update model type strings (``"xgb"`` → ``"xgboost"``).
- ☐ Review feature engineering — v4 uses composable transform pipelines rather than
  implicit feature selection from column names.

Next Steps
----------

- :doc:`/user_guide/getting_started/installation` — install the v4 packages
- :doc:`/user_guide/concepts/datasets` — understand TimeSeriesDataset in depth
- :doc:`/user_guide/concepts/configuration` — the new configuration model
- :doc:`/user_guide/guides/forecasting` — end-to-end forecasting workflow
- :doc:`/user_guide/deployment/index` — patterns for production integration