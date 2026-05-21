Deployment
==========

OpenSTEF is a **pure ML library** — it has no built-in scheduling, orchestration,
or data storage. You bring your own infrastructure and wrap OpenSTEF's training
and prediction steps in whatever execution environment fits your team.

This page explains how OpenSTEF fits into different deployment contexts, from
scheduled notebooks to enterprise-scale queued systems. It does not prescribe a
single architecture; instead, it shows the integration points where your
infrastructure connects to OpenSTEF's API.

.. mermaid:: /diagrams/user_guide/guides/deployment_diagram_1.mmd

What OpenSTEF Does and Does Not Provide
----------------------------------------

OpenSTEF provides:

- Feature engineering, model training, and prediction via :class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`
- Model versioning and artifact storage via :class:`~openstef_models.integrations.mlflow.MLFlowStorageCallback`
- Configurable callbacks for lifecycle events

OpenSTEF does **not** provide:

- Job scheduling or cron-like triggers
- Data ingestion or API connectors
- Database or message queue integrations
- Deployment infrastructure (containers, serverless functions)

Your deployment wraps two core operations:

.. code-block:: python

   # Training
   result = workflow.fit(train_dataset)

   # Prediction
   forecast = workflow.predict(predict_dataset, forecast_start=start)

Everything else — how you fetch data, when you trigger these calls, and where you
store results — is your responsibility.


Deployment Tiers
----------------

The table below summarizes three common patterns observed in production deployments:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Tier
     - Pattern
     - Typical Tools
     - When to Use
   * - Small teams / startups
     - Scheduled notebooks
     - SageMaker, Databricks, Jupyter + cron
     - < 50 locations, rapid iteration
   * - Mid-size with own infra
     - DAG-based pipelines
     - Airflow, Dagster, Prefect
     - 50–500 locations, separate concerns
   * - Enterprise at scale
     - Queued execution
     - Custom task queues, Kubernetes
     - 500+ locations, strict SLAs

Tier 1: Scheduled Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest deployment is a notebook (or script) that runs on a schedule. Each
notebook handles the full lifecycle: fetch data, train or predict, store results.

A single notebook might:

1. Query your data warehouse for historical load and weather data
2. Construct a :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`
3. Call ``workflow.fit()`` or ``workflow.predict()``
4. Write forecasts back to your database or dashboard

This works well for small teams because there's no infrastructure to maintain
beyond the compute environment. The tradeoff is that data integration, ML logic,
and storage are coupled in one artifact.

Tier 2: DAG-Based Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you need separation of concerns — different teams own data ingestion vs. ML
vs. delivery — a directed acyclic graph (DAG) orchestrator is natural:

- **Stage 1: Data integration** — fetch weather forecasts, meter readings, and
  assemble datasets
- **Stage 2: Training** — periodic retraining (e.g., weekly) using ``workflow.fit()``
- **Stage 3: Prediction** — frequent forecasting (e.g., every 15 minutes) using
  ``workflow.predict()``
- **Stage 4: Delivery** — push forecasts to downstream systems

Each stage is an independent task with clear inputs and outputs. MLflow (see below)
bridges stages 2 and 3 by persisting trained models.

Tier 3: Queued Execution at Scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large-scale deployments (thousands of locations), the pattern shifts to
lightweight per-location tasks dispatched from a queue:

- A scheduler enqueues tasks: "forecast location X" or "train location Y"
- Workers pull tasks, execute the OpenSTEF workflow, and store results
- Data integration is optimized: batch-fetch weather data for all locations, then
  fan out to individual forecast tasks

.. note::

   Sigholm (an OpenSTEF contributor) processes approximately 20,000 forecasts per
   week with 4-second cycle times per forecast using this queued pattern.

The key insight is that OpenSTEF's ``workflow.predict()`` is stateless once a model
is loaded — it scales horizontally without coordination between workers.


Model Storage with MLflow
-------------------------

Across all deployment tiers, you need a way to persist trained models between
training and prediction runs. OpenSTEF's **one opinionated choice** about storage
is the :class:`~openstef_models.integrations.mlflow.MLFlowStorageCallback`, which
integrates with MLflow for:

- **Model versioning** — every training run produces a versioned artifact
- **Experiment tracking** — metrics, hyperparameters, and metadata are logged
- **Model selection** — automatically pick the best-performing model across runs
- **Model reuse** — skip retraining if the current model is still recent and performant

The callback hooks into the workflow lifecycle:

.. code-block:: python

   from openstef_models.integrations.mlflow import MLFlowStorageCallback, MLFlowStorage

   callbacks = [
       MLFlowStorageCallback(
           storage=MLFlowStorage(tracking_uri="http://your-mlflow-server"),
           model_reuse_enable=True,
           model_reuse_max_age=timedelta(days=7),
       )
   ]

   workflow = CustomForecastingWorkflow(model_id="location_42", model=model, callbacks=callbacks)

With this in place, ``workflow.fit()`` logs the trained model to MLflow, and
``workflow.predict()`` loads the latest model automatically. This decouples your
training schedule from your prediction schedule.

.. note::

   MLflow is an optional dependency. If your deployment uses a different model
   registry, implement the :class:`~openstef_models.workflows.custom_forecasting_workflow.ForecastingCallback`
   interface to integrate with your storage backend.


Data Sources
------------

OpenSTEF expects input data as a :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`
— it does not fetch data itself. Your data integration layer must supply:

- **Load/generation measurements** — historical actuals for training
- **Weather forecasts** — temperature, wind speed, radiation, humidity, pressure
- **Calendar features** — handled internally by OpenSTEF's feature engineering

Common weather data sources:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Source
     - Coverage
     - Notes
   * - `Open-Meteo <https://open-meteo.com/>`_
     - Global, free tier available
     - Good default for most deployments
   * - KNMI
     - Netherlands
     - High-quality observations and forecasts
   * - MFFBAS
     - Netherlands
     - Specialized Dutch meteorological data

Your data integration code maps source-specific column names to the columns
OpenSTEF expects (e.g., ``temperature_2m``, ``wind_speed_10m``,
``shortwave_radiation``). See :doc:`datasets` for details on dataset structure.


Writing Custom Integrations
---------------------------

OpenSTEF is designed to be modular. If your deployment produces artifacts or uses
services that OpenSTEF doesn't support yet, extend it:

- **Custom callbacks** — implement :class:`~openstef_models.workflows.custom_forecasting_workflow.ForecastingCallback`
  to hook into ``on_fit_start``, ``on_fit_end``, ``on_predict_start``, etc.
- **Custom storage backends** — replace MLflow with your own model registry
- **Custom transforms** — add domain-specific feature engineering to the
  :class:`~openstef_core.mixins.TransformPipeline`

Contributions to the OpenSTEF project are welcome — if your integration is
general-purpose, consider upstreaming it.


Operational Considerations
--------------------------

Regardless of deployment tier, consider:

- **Retraining frequency** — models degrade over time as load patterns shift.
  Weekly or bi-weekly retraining is common. MLflow's model reuse feature avoids
  unnecessary retraining when performance is still acceptable.
- **Fallback behavior** — when data feeds fail, forecasts must still be produced.
  See :doc:`reliability_fallback` for OpenSTEF's approach.
- **Monitoring** — use the :class:`~openstef_models.integrations.mlflow.MLFlowStorageCallback`
  metrics logging or implement a custom callback to track forecast quality over time.
- **Backtesting before deployment** — validate model performance on historical data
  before going live. See :doc:`backtesting`.

.. mermaid:: /diagrams/user_guide/guides/deployment_diagram_2.mmd

Next Steps
----------

- :doc:`forecasting` — understand the full forecasting lifecycle
- :doc:`datasets` — learn how to structure input data
- :doc:`reliability_fallback` — handle data feed failures in production
- :doc:`backtesting` — validate models before deployment