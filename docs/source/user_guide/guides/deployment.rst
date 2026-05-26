.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guide_deployment:

Deployment
==========

OpenSTEF is a pure machine learning library. It has no built-in scheduler, no orchestration engine, no database layer, and no web server. You bring your own infrastructure and wrap OpenSTEF's training and prediction workflows in whatever execution environment suits your organization.

This page explains the three most common deployment strategies, how to handle model storage, and where to integrate external data sources.

Why Deployment is Your Responsibility
-------------------------------------

OpenSTEF's core API is intentionally narrow: you call :class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow` with data, and it returns forecasts or a trained model. Everything else (scheduling, data retrieval, result storage, alerting) lives outside the library.

This design means OpenSTEF works in any environment that can run Python, from a single notebook to a distributed task queue. The trade-off is that you must build the integration layer yourself.

Deployment Strategies
---------------------

The table below summarizes three common approaches, ordered by operational complexity:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Strategy
     - Description
     - Best For
     - Considerations
   * - Scheduled notebooks
     - Self-contained notebooks that load data, train/predict, and store results. Run on a timer.
     - Small teams, proof-of-concept, fewer than ~50 forecast targets.
     - Simple but limited visibility into failures.
   * - DAG-based orchestration
     - Separate tasks for data integration, training, and prediction, connected in a directed acyclic graph.
     - Teams needing retry logic, dependency tracking, and audit trails.
     - Requires an orchestration platform and task design.
   * - Queued execution
     - Lightweight per-location tasks dispatched to a pool of workers with optimized batch data loading.
     - Large-scale operations with thousands of forecast targets.
     - Highest throughput but most complex to operate.

Scheduled Notebooks
^^^^^^^^^^^^^^^^^^^

The simplest deployment: a notebook (or script) that contains the full cycle of data loading, model training or prediction, and result storage. You schedule it to run periodically using any compute platform that supports timed execution (managed notebook services, cron, CI/CD pipelines).

A typical notebook structure:

.. code-block:: python

   # 1. Load data from your source
   data = load_my_timeseries(location_id="substation_A")
   # 2. Run the OpenSTEF workflow
   forecast = workflow.predict(data)
   # 3. Store results
   save_forecast_to_database(forecast)

This approach works well when you have a small number of forecast targets and can tolerate coarse error handling (the whole notebook succeeds or fails).

DAG-based Orchestration
^^^^^^^^^^^^^^^^^^^^^^^

When you need visibility into individual pipeline stages, retry logic for transient failures, and clear dependency management, a DAG tool (Airflow, Dagster, Prefect, or similar) is a natural fit.

A typical DAG separates concerns into discrete tasks:

- **Data integration task**: fetches weather forecasts and meter data, validates completeness.
- **Training task**: calls ``workflow.fit()`` on prepared data, runs on a slower schedule (e.g., daily or weekly).
- **Prediction task**: calls ``workflow.predict()`` with fresh input data, runs on a fast schedule (e.g., every 15 minutes).

Each task is independently retriable. The DAG tool handles scheduling, logging, and alerting. OpenSTEF remains a library call inside each task.

Queued Execution
^^^^^^^^^^^^^^^^

For organizations managing thousands of forecast targets (e.g., one per grid connection or transformer), dispatching individual tasks to a worker pool provides the best throughput. A message queue or task broker distributes work; workers pull tasks, run the OpenSTEF workflow, and report results.

Key optimizations at this scale:

- **Batch data loading**: fetch weather data once and distribute it to all workers that need the same region.
- **Model caching**: reuse loaded model artifacts across predictions for the same location within a worker.
- **Graceful degradation**: if a single location fails, other locations continue unaffected. See :doc:`/user_guide/guides/reliability_fallback` for fallback strategies.

Data Sources
------------

OpenSTEF requires time series input data (load measurements) and, for most models, weather forecast features. The library does not fetch data; you provide it as a :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`.

Common weather data sources:

- **Open-Meteo** (https://open-meteo.com): free, global coverage, multiple weather models. A good default for any deployment.
- **KNMI** (https://www.knmi.nl): high-quality observations and forecasts specific to the Netherlands.

Your data integration layer is responsible for fetching from these (or other) sources, aligning timestamps, and assembling the dataset that OpenSTEF expects.

Model Storage with MLflow
-------------------------

OpenSTEF supports MLflow as a model registry backend through :class:`~openstef_models.integrations.mlflow.mlflow_storage_callback.MLFlowStorageCallback`. This callback hooks into the workflow lifecycle to automatically store trained models, log metrics, and retrieve previously trained models for reuse.

.. code-block:: python

   from openstef_models.integrations.mlflow.mlflow_storage_callback import MLFlowStorageCallback

   callback = MLFlowStorageCallback(
       storage=my_mlflow_storage,
       model_reuse_enable=True,
       model_reuse_max_age=timedelta(days=7),
   )
   workflow = CustomForecastingWorkflow(
       model=model, model_id="substation_A", callbacks=[callback]
   )

The callback handles:

- **Model versioning**: each training run produces a new model version in the MLflow registry.
- **Model reuse**: if a recently trained model exists and ``model_reuse_enable=True``, the workflow skips retraining.
- **Model selection**: optionally compares new models against previous versions using a configurable metric.

This mechanism works identically across all three deployment strategies. Whether you run in a notebook or a distributed queue, the same callback persists and retrieves models.

Custom Storage Backends
^^^^^^^^^^^^^^^^^^^^^^^

The callback system is pluggable. If MLflow does not fit your infrastructure, you can implement your own callback by subclassing :class:`~openstef_models.workflows.custom_forecasting_workflow.ForecastingCallback`:

.. code-block:: python

   class MyStorageCallback(ForecastingCallback):
       def on_fit_end(self, context, result):
           # Save model artifact to your backend
           ...

Register your callback in the workflow's ``callbacks`` list. The workflow calls your hooks at each lifecycle stage (``on_fit_start``, ``on_fit_end``, ``on_predict_start``, ``on_predict_end``).

Modularity and Extensibility
-----------------------------

OpenSTEF is designed to be composed, not configured. If your deployment needs something the library does not provide:

- Write a custom :class:`~openstef_models.workflows.custom_forecasting_workflow.ForecastingCallback` for monitoring, storage, or validation logic.
- Implement your own data loading layer that produces the expected dataset format.
- Add custom preprocessing transforms to the model pipeline (see :doc:`/user_guide/concepts/models`).

Contributions to the project are welcome. If you build an integration that others would benefit from, consider opening a pull request.

.. note::

   For details on how to structure the forecasting workflow itself (model creation, training, prediction), see :doc:`/user_guide/guides/forecasting`. This page focuses on the infrastructure surrounding those calls.