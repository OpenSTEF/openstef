Migrating from OpenSTEF 3
=========================

This page helps existing OpenSTEF v3 users understand what changed in v4 and how to
adapt their code. If you are starting fresh with v4, you can skip this page and go
directly to :doc:`/user_guide/getting_started/installation`.

Why the Architecture Changed
----------------------------

In v3, ``openstef`` was a single package containing ML models, data classes, evaluation
logic, and pipeline orchestration. A companion package, ``openstef-dbc``, provided
database connectors. This coupling meant that users who only needed the forecasting
models still pulled in database dependencies, and changes to evaluation code could
inadvertently affect training pipelines.

V4 separates these concerns into independent packages. Each package has a focused
responsibility, its own release cadence, and minimal dependencies on the others.

.. mermaid:: /diagrams/user_guide/getting_started/migration_diagram_1.mmd

Package Mapping
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - V3 Package
     - V4 Package
     - Responsibility
   * - ``openstef`` (core ML)
     - ``openstef-models``
     - Forecasting models, transforms, workflows
   * - ``openstef`` (data classes)
     - ``openstef-core``
     - ``TimeSeriesDataset``, base types, exceptions
   * - ``openstef`` (evaluation)
     - ``openstef-beam``
     - Backtesting, metrics, analysis plots
   * - (new in v4)
     - ``openstef-meta``
     - Metalearning, ensemble presets
   * - ``openstef-dbc``
     - No equivalent
     - Users provide their own I/O layer

Configuration
-------------

V3 used :class:`PredictionJobDataClass`, a plain Python dataclass with loosely-typed
fields passed as a dictionary. V4 replaces it with
:class:`~openstef_models.presets.forecasting_workflow.ForecastingWorkflowConfig`, a
Pydantic model that validates field types and values at construction time.

**Before (V3):**

.. code-block:: python

   from openstef.data_classes.prediction_job import PredictionJobDataClass

   pj = PredictionJobDataClass(**{"id": 287, "model": "xgb", "quantiles": [10, 30, 50, 70, 90],
                                   "resolution_minutes": 15, "forecast_type": "demand"})

**After (V4):**

.. code-block:: python

   from openstef_models.presets.forecasting_workflow import ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(model_id="287", model="xgboost",
                                       quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])

Key differences:

- Quantiles are now expressed as fractions (0.1) rather than percentages (10).
- The ``model`` field uses full names (see :ref:`model-type-renames` below).
- Validation errors surface immediately at construction, not deep inside a pipeline.

Data Representation
-------------------

V3 passed raw ``pandas.DataFrame`` objects to pipelines. Users had to ensure the
correct columns, index frequency, and metadata were present by convention.

V4 introduces :class:`~openstef_core.datasets.TimeSeriesDataset`, which wraps a
DataFrame and carries metadata the pipeline needs (sample interval, availability
windows, sorted index guarantees). This eliminates an entire class of silent bugs where
mismatched frequencies or missing columns only surfaced mid-pipeline.

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset

   dataset = TimeSeriesDataset(data=df, sample_interval_minutes=15)

Specialized subclasses such as ``ForecastInputDataset`` add further validation (e.g.,
requiring a target column).

API: Pipelines to Workflows
----------------------------

V3's API followed a "pipeline/task" pattern where ``openstef-dbc`` fetched data, ran a
pipeline function, and stored results:

**Before (V3):**

.. code-block:: python

   from openstef.pipeline.train_model import train_model_pipeline

   train, val, test = train_model_pipeline(pj, train_data, check_old_model_age=False,
                                            mlflow_tracking_uri="./models")

**After (V4):**

.. code-block:: python

   from openstef_models.workflows import CustomForecastingWorkflow

   result = workflow.fit(train_dataset)
   forecast = workflow.predict(predict_dataset, forecast_start=train_end)

In v4, :class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`
owns the orchestration logic (callbacks, model selection, lifecycle events) while you
own the I/O. This means you decide where data comes from and where results go.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - V3 Function
     - V4 Equivalent
   * - ``train_model_pipeline(pj, data)``
     - ``workflow.fit(dataset)``
   * - ``create_forecast_pipeline(pj, data)``
     - ``workflow.predict(dataset, forecast_start=...)``
   * - ``optimize_hyperparameters_pipeline(pj, data, n_trials)``
     - Configure via ``ForecastingWorkflowConfig`` or custom callbacks

Database Connector
------------------

``openstef-dbc`` provided opinionated database access (InfluxDB, MySQL) for fetching
input data and storing forecasts. V4 does not include a database connector because
storage requirements vary widely across deployments.

If your v3 code relied on ``openstef-dbc``:

- Extract your data-fetching logic into a function that returns a
  :class:`~openstef_core.datasets.TimeSeriesDataset`.
- Extract your result-storage logic into a
  :class:`~openstef_models.workflows.custom_forecasting_workflow.ForecastingCallback`
  that runs after ``fit`` or ``predict``.
- See :doc:`/user_guide/guides/deployment` for integration patterns.

This separation gives you full control over connection pooling, retry logic, and schema
evolution without waiting for upstream library changes.

.. _model-type-renames:

Model Type Renames
------------------

V4 uses unambiguous, full model names:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - V3 Name
     - V4 Name
     - Notes
   * - ``xgb``
     - ``xgboost``
     - Full library name
   * - ``lgb``
     - ``lgbm``
     - Matches ``lightgbm`` abbreviation convention

Quantile configuration is no longer embedded in the model name or type. Instead,
quantiles are specified in the workflow configuration and apply uniformly to whichever
forecaster you choose.

Migration Checklist
-------------------

Use this checklist when converting a v3 codebase:

- Replace ``from openstef.*`` imports with the appropriate v4 package
  (``openstef_core``, ``openstef_models``, ``openstef_beam``, ``openstef_meta``).
- Replace ``PredictionJobDataClass`` with ``ForecastingWorkflowConfig``.
- Convert quantile values from percentages to fractions (e.g., 10 becomes 0.1).
- Rename model types: ``xgb`` to ``xgboost``, ``lgb`` to ``lgbm``.
- Wrap input DataFrames in ``TimeSeriesDataset`` (or ``ForecastInputDataset``).
- Replace ``train_model_pipeline`` / ``create_forecast_pipeline`` calls with
  ``workflow.fit()`` / ``workflow.predict()``.
- Replace ``openstef-dbc`` data fetching with your own I/O code that produces a
  ``TimeSeriesDataset``.
- Replace ``openstef-dbc`` result storage with a ``ForecastingCallback``.
- Update MLflow tracking: v4 workflows use callbacks rather than direct URI arguments.
- Run your test suite; Pydantic validation will surface most remaining type mismatches.

.. warning::

   V3 and v4 cannot coexist in the same Python environment because they share the
   ``openstef`` namespace prefix. Migrate one project at a time or use separate
   virtual environments.