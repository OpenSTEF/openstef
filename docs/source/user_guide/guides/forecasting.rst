.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guide_forecasting:

Forecasting
===========

This page explains how OpenSTEF's forecasting system is designed, what it expects from you, and what it produces. It covers the containment hierarchy, data requirements, and the fit/predict lifecycle. For runnable code examples, see :doc:`/tutorials/forecasting_quickstart` and :doc:`/tutorials/custom_pipeline`.

What OpenSTEF Needs From You
----------------------------

To produce a forecast, OpenSTEF requires three things:

- A datetime-indexed DataFrame with a ``load`` column. This is your target variable
  (energy demand or generation). The DataFrame index must be a ``DatetimeIndex``
  named ``timestamp``.
- Weather features (or other exogenous variables). These are additional columns in
  the same DataFrame, such as temperature, wind speed, or irradiance. OpenSTEF does
  not fetch weather data for you - your data integration layer provides it. See
  :doc:`/user_guide/guides/deployment` for common weather data sources.
- A declared ``sample_interval``. This tells OpenSTEF the expected time resolution
  of your data (e.g., 15 minutes, 1 hour). It defaults to 15 minutes if not specified.

You wrap your data in a :class:`~openstef_core.datasets.TimeSeriesDataset`:

.. code-block:: python

   from datetime import timedelta
   from openstef_core.datasets import TimeSeriesDataset

   dataset = TimeSeriesDataset(df, sample_interval=timedelta(minutes=15))

The ``target_column`` defaults to ``"load"`` but can be overridden. Any columns beyond the internal ones (``horizon``, ``available_at``, the target) are treated as features.

What OpenSTEF Gives Back
------------------------

OpenSTEF returns a :class:`~openstef_core.datasets.validated_datasets.ForecastDataset` containing:

- **Point forecasts** (the median or expected value).
- **Optional quantile forecasts** representing prediction intervals (e.g., the 10th and 90th percentiles).

The result is a validated, datetime-indexed dataset that you can directly use for operational decisions. For details on probabilistic output, see :doc:`/user_guide/guides/probabilistic_forecasting`.

The Containment Hierarchy
-------------------------

OpenSTEF organizes forecasting logic in layers of increasing abstraction:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Level
     - Responsibility
     - When to use
   * - **Forecaster**
     - A single ML model (e.g., XGBoost, linear). Implements ``fit()`` and ``predict()``.
     - When building custom model types.
   * - **ForecastingModel**
     - Wraps a Forecaster with preprocessing and postprocessing pipelines.
     - When you need full control over transforms.
   * - **CustomForecastingWorkflow**
     - Adds lifecycle management, callbacks, and model persistence around a ForecastingModel.
     - Research, experimentation, custom integrations.
   * - **Preset** (``create_forecasting_workflow``)
     - A factory that assembles a complete Workflow from a configuration object.
     - Production deployments.

**Recommendation:** Start with Presets (via :func:`~openstef_models.presets.create_forecasting_workflow`) for production use. They encode best-practice defaults for preprocessing, postprocessing, callbacks, and model storage. Use :class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow` directly when you need to experiment with non-standard pipelines or custom callback logic.

Data Requirements
-----------------

**Columns**

The only strictly required column is the target (``load`` by default). All other columns are treated as input features. Two special columns are recognized if present:

- ``horizon``: a ``timedelta`` indicating how far ahead each row's forecast applies. When present, the dataset is considered "versioned."
- ``available_at``: a ``datetime`` indicating when the data for that row became available.

**Time Zones**

The DataFrame index should be timezone-aware. OpenSTEF does not implicitly localize timestamps; you are responsible for ensuring consistency between your input data and any weather data sources.

**Sample Interval**

The ``sample_interval`` parameter declares the expected temporal resolution. It is used for validation, gap detection, and feature engineering (e.g., computing lag features). If your data has a 15-minute resolution, pass ``timedelta(minutes=15)``. Mismatches between declared and actual frequency can cause silent errors in lag-based features.

.. warning::

   When using lag-based preprocessing transforms, the first rows of your dataset will contain NaN values (e.g., a 14-day lag creates 14 days of incomplete data). Set the ``cutoff_history`` parameter on your :class:`~openstef_models.models.forecasting_model.ForecastingModel` to exclude these rows from training.

The Fit/Predict Lifecycle
-------------------------

At the Workflow level, both training and inference follow a structured lifecycle with callback hooks at each stage.

.. mermaid:: /diagrams/user_guide/guides/forecasting_diagram_1.mmd

**Fit (Training)**

1. **Preprocessing**: The input :class:`~openstef_core.datasets.TimeSeriesDataset` passes through a :class:`~openstef_core.mixins.transform.TransformPipeline` that performs validation, feature engineering, and standardization.
2. **Forecaster.fit()**: The preprocessed data is passed to the underlying forecaster for model training.
3. **Callbacks**: After training completes, lifecycle callbacks fire (e.g., persisting the trained model, logging metrics, and storing feature importance plots).

The ``fit()`` method returns a :class:`~openstef_models.models.forecasting_model.ModelFitResult` containing training metadata and metrics.

**Predict (Inference)**

1. **Preprocessing**: New data passes through the same preprocessing pipeline used during training, ensuring feature consistency.
2. **Forecaster.predict()**: The model generates raw predictions.
3. **Postprocessing**: A separate :class:`~openstef_core.mixins.transform.TransformPipeline` applies quantile sorting (via :class:`~openstef_models.transforms.postprocessing.quantile_sorter.QuantileSorter`) and confidence interval construction (via :class:`~openstef_models.transforms.postprocessing.confidence_interval_applicator.ConfidenceIntervalApplicator`).
4. **Callbacks**: The final :class:`~openstef_core.datasets.validated_datasets.ForecastDataset` is passed to callbacks for logging or downstream delivery.

Model Reuse
-----------

In production, you do not always need to retrain. The :class:`~openstef_models.integrations.mlflow.mlflow_storage_callback.MLFlowStorageCallback` supports model reuse: if a recently trained model exists in storage (within ``model_reuse_max_age``, defaulting to 7 days), the workflow can skip training entirely and load the existing model for prediction.

This behavior is controlled by two configuration parameters:

- ``model_reuse_enable``: Whether to attempt loading an existing model before training (default: ``True``).
- ``model_reuse_max_age``: Maximum age of a stored model that is still considered valid (default: 7 days).

When model reuse is active and a valid model is found, the workflow raises a :class:`~openstef_core.exceptions.SkipFitting` signal internally, and the ``fit()`` call returns ``None`` instead of a new fit result.

Model selection can also be enabled: when a new model is trained, its performance is compared against the existing model using a configurable metric. The old model receives a penalty factor (``model_selection_old_model_penalty``) to bias selection toward fresher models.

For details on how this integrates with scheduled retraining in production, see :doc:`/user_guide/guides/deployment`.

Connecting the Pieces
---------------------

A minimal production setup using a Preset looks like this:

.. code-block:: python

   from openstef_models.presets import create_forecasting_workflow, ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(model_id="my_substation_01", ...)
   workflow = create_forecasting_workflow(config=config)

The workflow object exposes ``fit(data)`` and ``predict(data)`` at the top level. The Preset handles wiring up preprocessing, postprocessing, callbacks, and storage based on your configuration.

For a complete walkthrough with real data, see :doc:`/tutorials/forecasting_quickstart`. For building custom pipelines with non-default transforms, see :doc:`/tutorials/custom_pipeline`.

.. seealso::

   - :ref:`concept_models` for details on available model types and the containment hierarchy.
   - :doc:`/user_guide/guides/probabilistic_forecasting` for quantile forecasts and calibration.
   - :doc:`/user_guide/guides/deployment` for integrating forecasting into production systems.
   - :doc:`/user_guide/guides/backtesting_tutorial` for a hands-on backtest walkthrough; see also :ref:`concept_beam` for the broader framework.