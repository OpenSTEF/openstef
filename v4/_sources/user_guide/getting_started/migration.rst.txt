.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

Migrating from OpenSTEF 3 to 4
===============================

This page guides users of the legacy ``openstef`` v3 package through the conceptual
and practical changes in OpenSTEF v4. If you are starting fresh, skip this page and
begin with :doc:`/user_guide/getting_started/installation`.

.. note::

   V4 is a ground-up redesign. While the forecasting *goals* are the same, the API
   surface is intentionally different. Plan for a rewrite of integration code, not a
   find-and-replace.

Package Structure
-----------------

V4 splits functionality into focused, independently installable packages. You can
use the models without any database dependency, and each package can be tested in
isolation.

In v3 these were all bundled in a single ``openstef`` package:

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

Configuration
-------------

V4's :class:`~openstef_models.presets.ForecastingWorkflowConfig` validates all fields
at construction time. Model types are constrained literals, durations use ``timedelta``,
and hyperparameters are typed per-model objects. This replaces v3's
``PredictionJobDataClass`` which used free-form strings and untyped dicts.

**Before (V3):**

.. code-block:: python

   from openstef.data_classes.prediction_job import PredictionJobDataClass

   pj = PredictionJobDataClass(
       id=287,
       model="xgb",
       resolution_minutes=15,
       forecast_type="demand",
       quantiles=[10, 30, 50, 70, 90],
   )

**After (V4):**

.. code-block:: python

   from openstef_models.presets import ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(
       model_id="loc_287",
       model="xgboost",
       sample_interval=timedelta(minutes=15),
       quantiles=[Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9)],
   )

Key improvements:

- :class:`~openstef_models.presets.ForecastingWorkflowConfig` uses Pydantic v2 with
  strict validation. Configuration errors surface immediately at construction time.
- Model types, quantiles, and durations use constrained types with IDE autocompletion.
- Location, horizon, and hyperparameter settings are structured sub-objects with their
  own validation and defaults.

.. list-table:: Field Mapping
   :header-rows: 1
   :widths: 35 35 30

   * - V3 (``PredictionJobDataClass``)
     - V4 (``ForecastingWorkflowConfig``)
     - Notes
   * - ``id``
     - ``model_id``
     - Now ``ModelIdentifier`` type
   * - ``model`` (free string)
     - ``model`` (Literal)
     - Validated; see Model Types below
   * - ``model_kwargs`` (dict)
     - ``xgboost_hyperparams`` / ``gblinear_hyperparams`` / etc.
     - Per-model typed config objects
   * - ``resolution_minutes`` (int)
     - ``sample_interval`` (timedelta)
     -
   * - ``horizon_minutes`` (int)
     - ``horizons`` (list[LeadTime])
     - Supports multiple horizons
   * - ``lat`` / ``lon``
     - ``location.coordinate``
     - Structured LocationConfig sub-object
   * - ``name``
     - ``location.name``
     -
   * - ``quantiles`` (list[float])
     - ``quantiles`` (list[Quantile])
     - Always set; default ``[0.5]``
   * - ``completeness_threshold``
     - ``completeness_threshold``
     - Same name and semantics
   * - ``flatliner_threshold_minutes`` (int)
     - ``flatliner_threshold`` (timedelta)
     -
   * - ``detect_non_zero_flatliner``
     - ``detect_non_zero_flatliner``
     - Unchanged
   * - ``predict_non_zero_flatliner``
     - ``predict_nonzero_flatliner``
     - Slight rename (no underscore in "nonzero")
   * - ``rolling_aggregate_features``
     - ``rolling_aggregate_features``
     - Unchanged
   * - ``forecast_type``
     - Removed
     - Handled by model choice + transforms
   * - ``electricity_bidding_zone``
     - ``location.country_code``
     - Simplified
   * - ``train_split_func``
     - ``data_splitter``
     - Structured config
   * - ``depends_on``
     - Removed
     - User manages orchestration
   * - ``pipelines_to_run``
     - Removed
     - Call workflow methods directly
   * - ``default_modelspecs``
     - ``selected_features``
     - Feature selection via enum

See the :class:`~openstef_models.presets.ForecastingWorkflowConfig` API reference for
the full list of fields and defaults.

Data Handling
-------------

V4 introduces :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`,
which carries the data **and** its metadata together. In v3, metadata like sample
interval and column roles lived separately in the prediction job.

Benefits of ``TimeSeriesDataset``:

- **Sample interval** -- validated on construction, ensuring consistent resampling.
- **Availability windows** -- tracks when each observation became available (critical
  for correct backtesting without lookahead).
- **Versioning** -- supports horizon-aware or ``available_at``-aware slicing.

**Before (V3):**

.. code-block:: python

   import pandas as pd

   input_data = pd.read_csv("data.csv", index_col="index", parse_dates=True)

   train_model_pipeline(pj, input_data)

**After (V4):**

.. code-block:: python

   import pandas as pd
   from openstef_core.datasets import TimeSeriesDataset

   df = pd.read_csv("data.csv", index_col="index", parse_dates=True)

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
   )

The dataset is then passed to workflows which can introspect its properties without
external configuration.

Workflows
---------

V4 unifies the v3 concepts of "tasks" (database-coupled orchestration) and
"pipelines" (ML logic) into a single concept: **Workflows**. A workflow encapsulates
the full train/predict cycle without assuming any particular storage backend.

.. list-table:: API Mapping
   :header-rows: 1
   :widths: 40 40

   * - V3
     - V4
   * - ``train_model_pipeline(pj, data)``
     - ``workflow.fit(dataset)``
   * - ``create_forecast(pj, data)``
     - ``workflow.predict(dataset)``
   * - Task (fetch -> pipeline -> store)
     - User code + workflow (you own I/O)

A **Preset** is a factory that builds a fully configured workflow from a config object:

.. code-block:: python

   from openstef_models.presets import create_forecasting_workflow, ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(
       model_id="loc_287",
       model="xgboost",
       sample_interval=timedelta(minutes=15),
       quantiles=[Q(0.1), Q(0.5), Q(0.9)],
   )

   workflow = create_forecasting_workflow(config)

   # Training
   workflow.fit(dataset)

   # Prediction
   forecast = workflow.predict(dataset)

:func:`~openstef_models.presets.create_forecasting_workflow` assembles preprocessing,
the forecaster, postprocessing, and callbacks into a
:class:`~openstef_models.presets.forecasting_workflow.CustomForecastingWorkflow`.

For ensemble approaches, ``openstef-meta`` provides
:func:`~openstef_meta.presets.forecasting_workflow.create_ensemble_forecasting_workflow`.

See :doc:`/user_guide/guides/forecasting` for a complete walkthrough.

Model Types
-----------

Model identifiers changed for clarity. Quantile variants are no longer separate model
types; configure quantiles via the ``quantiles`` field instead.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - V3
     - V4
     - Notes
   * - ``"xgb"``
     - ``"xgboost"``
     -
   * - ``"xgb_quantile"``
     - ``"xgboost"``
     - Quantiles now in config, not model name
   * - ``"xgb_multioutput_quantile"``
     - ``"xgboost"``
     - Same
   * - ``"lgb"``
     - ``"lgbm"``
     -
   * - ``"linear"``
     - Removed
     - Use ``"gblinear"`` instead
   * - ``"linear_quantile"``
     - ``"gblinear"``
     -
   * - ``"gblinear_quantile"``
     - ``"gblinear"``
     - Quantiles now in config
   * - ``"flatliner"``
     - ``"flatliner"``
     - Unchanged
   * - ``"median"``
     - ``"median"``
     - Unchanged
   * - (new)
     - ``"constant_quantile"``
     - Fallback model for low-data situations
   * - (new)
     - ``"lgbmlinear"``
     - LightGBM with linear learner

.. _migration_dbc:

Reference Implementation
------------------------

In v3, ``openstef-dbc`` provided scheduling, database integration, and orchestration.
V4 focuses on the core ML libraries and leaves integration to the user.

The `openstef-reference <https://github.com/OpenSTEF/openstef-reference>`_ repository
demonstrates how a complete v3 system was deployed (scheduling, data
integration, and storage). For v4 deployment patterns, see
:doc:`/user_guide/guides/deployment` instead.

If your v3 code relied on ``openstef-dbc``:

1. **Data ingestion** -- write an adapter that loads data into
   :class:`~openstef_core.datasets.timeseries_dataset.TimeSeriesDataset`.
2. **Result storage** -- extract forecasts from the workflow output and write them to
   your database of choice.
3. **Configuration storage** -- serialize
   :class:`~openstef_models.presets.ForecastingWorkflowConfig` to/from your config
   store (JSON, YAML, database row).