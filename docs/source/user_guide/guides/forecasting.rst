Forecasting
===========
Forecasting with OpenSTEF
============================

This page explains how forecasting works in OpenSTEF: what you provide, what you get back, and how the library's layered architecture lets you choose the right level of abstraction for your use case.

For runnable code, see :doc:`/tutorials/forecasting_quickstart` and :doc:`/tutorials/custom_pipeline`.

Why a Structured Forecasting Pipeline?
---------------------------------------

Energy forecasting is not just "call model.predict()." Real-world forecasting requires consistent preprocessing (alignment, feature engineering, standardization), proper handling of time zones and sample intervals, quantile estimation for uncertainty, and lifecycle management (versioning, logging, fallback). OpenSTEF encodes these concerns into a layered architecture so that each layer solves one problem well, and you only interact with the layer that matches your needs.

What You Provide
----------------

OpenSTEF expects a :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset` — a thin wrapper around a pandas DataFrame with specific invariants:

.. list-table:: Input Data Requirements
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Details
   * - **DatetimeIndex**
     - The DataFrame index must be a timezone-aware ``pd.DatetimeIndex``. All internal operations rely on this for alignment and horizon calculation.
   * - **Target column**
     - A column containing the value to forecast (e.g., ``load``, ``generation``). The name is configurable per model.
   * - **Feature columns**
     - Weather forecasts, calendar features, or any exogenous variables. Columns starting with ``__`` are treated as internal/metadata.
   * - **sample_interval**
     - A ``timedelta`` declaring the expected spacing between rows (e.g., ``timedelta(minutes=15)``). The dataset validates that the index frequency matches this interval.
   * - **available_at or horizon**
     - Optional versioning columns that indicate *when* a forecast was made or *how far ahead* it looks. Required for versioned/operational datasets.

.. warning::

   If your data frequency does not match the declared ``sample_interval``, the dataset will raise a ``ValueError`` at construction time. Resample or fill gaps before creating the dataset.

Time Zones
^^^^^^^^^^

All timestamps must be timezone-aware. OpenSTEF does not assume a default timezone — you must be explicit. This prevents subtle bugs when crossing DST boundaries, which is critical for energy systems operating on clock time.

What You Get Back
-----------------

Calling ``predict()`` returns a :class:`~openstef_core.datasets.forecast_dataset.ForecastDataset` containing:

- **Point forecasts** — the expected value at each timestamp
- **Quantile forecasts** (optional) — probabilistic bounds at configured quantiles (e.g., Q10, Q50, Q90)
- **Metadata** — horizon information, available_at timestamps

For details on quantile forecasts and their calibration, see :doc:`probabilistic_forecasting`.

The Forecasting Lifecycle
-------------------------

.. mermaid:: /diagrams/user_guide/guides/forecasting_diagram_1.mmd

The lifecycle has two phases:

1. **Fit** — Historical data flows through preprocessing transforms, producing a ``ForecastInputDataset``. The forecaster trains on this prepared data.
2. **Predict** — New data flows through the *same* preprocessing pipeline (ensuring consistency), the forecaster generates raw predictions, and postprocessing transforms produce the final ``ForecastDataset``.

The key insight: preprocessing is defined once and applied identically in both phases. This eliminates train/serve skew.

API Levels
----------

OpenSTEF provides four levels of abstraction. Choose based on how much control you need:

.. list-table:: API Levels
   :header-rows: 1
   :widths: 20 35 45

   * - Level
     - Entry Point
     - When to Use
   * - **Forecaster**
     - Individual forecaster classes (e.g., LightGBM, XGBoost)
     - Building custom pipelines from scratch; research on new algorithms
   * - **ForecastingModel**
     - :class:`~openstef_models.models.forecasting_model.ForecastingModel`
     - Single-forecaster pipeline with preprocessing/postprocessing; most flexibility
   * - **Workflow**
     - ``CustomForecastingWorkflow`` with callbacks
     - Adding lifecycle hooks (logging, validation, monitoring) around fit/predict
   * - **Preset**
     - :func:`~openstef_models.presets.create_forecasting_workflow`
     - Production deployments; configuration-driven setup via ``ForecastingWorkflowConfig``

Recommended Starting Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **For production:** Start with Presets. Use :func:`~openstef_models.presets.create_forecasting_workflow` with a ``ForecastingWorkflowConfig`` to get a fully configured workflow with sensible defaults. You configure via a dataclass — no pipeline assembly required.

- **For research:** Start with the Workflow level. Subclass callbacks to log experiments, then drop down to ``ForecastingModel`` when you need to swap transforms or forecasters.

The ForecastingModel Layer
--------------------------

:class:`~openstef_models.models.forecasting_model.ForecastingModel` is the core orchestrator. It manages:

- A **preprocessing** ``TransformPipeline`` (feature engineering, standardization, lag computation)
- A **Forecaster** (the ML model itself)
- A **postprocessing** ``TransformPipeline`` (inverse transforms, quantile assembly)

.. code-block:: python

   # Conceptual usage — see tutorials for complete examples
   model.fit(training_data)
   forecast = model.predict(new_data, forecast_start=target_time)

.. note::

   The ``cutoff_history`` parameter is crucial when using lag-based features. A lag-14 transform creates NaN values for the first 14 days. Set ``cutoff_history`` to exclude incomplete rows from training.

The Workflow Layer
------------------

Workflows wrap a ``ForecastingModel`` with lifecycle callbacks. The callback interface provides hooks at every stage:

- ``on_fit_start`` / ``on_fit_end`` — pre-training validation, post-training metrics
- ``on_predict_start`` / ``on_predict_end`` — input validation, forecast storage

All callbacks have no-op defaults, so you override only what you need. This is where you integrate MLflow tracking, model registries, or alerting systems.

For deployment patterns using workflows, see :doc:`deployment`.

The Preset Layer
----------------

Presets are configuration-driven workflow factories. You describe *what* you want (model type, quantiles, location metadata) and the preset assembles the full pipeline:

.. code-block:: python

   from openstef_models.presets import create_forecasting_workflow, ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(...)
   workflow = create_forecasting_workflow(config)

This is the recommended entry point for operational systems where reproducibility and configuration management matter more than flexibility.

Data Flow Summary
-----------------

.. list-table:: Key Types in the Pipeline
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Role
   * - ``TimeSeriesDataset``
     - Raw input: your datetime-indexed measurements and features
   * - ``ForecastInputDataset``
     - Preprocessed, model-ready features (output of the transform pipeline)
   * - ``ForecastDataset``
     - Model output: point forecasts and quantiles with temporal metadata

For a deeper understanding of these dataset types and why they exist, see :doc:`datasets`.

Handling Failures
-----------------

In production, data feeds fail and models degrade. OpenSTEF provides reliability mechanisms at the workflow level — see :doc:`reliability_fallback` for fallback strategies.

Next Steps
----------

- :doc:`/tutorials/forecasting_quickstart` — End-to-end example from data loading to forecast
- :doc:`/tutorials/custom_pipeline` — Building a custom preprocessing and forecasting pipeline
- :doc:`probabilistic_forecasting` — Understanding and configuring quantile forecasts
- :doc:`backtesting` — Evaluating forecast quality on historical data