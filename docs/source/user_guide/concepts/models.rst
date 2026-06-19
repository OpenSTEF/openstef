.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concept_models:

Models
======

OpenSTEF's forecasting system is built from composable components. At the lowest level,
a Forecaster wraps a single ML algorithm. Transforms handle feature engineering and
postprocessing. A Model binds these together into a trainable unit. Higher-level
Workflows add lifecycle management, and Presets provide opinionated defaults for
production. You choose the level of abstraction that matches your use case.

.. note::

   For how models are *selected automatically* via metalearning, see
   :doc:`metalearning`. For how models are *evaluated* in backtesting, see
   :doc:`beam`.


Component Overview
------------------

OpenSTEF's model system is composed of five components, each building on the one
below. You can enter at any level depending on how much control you need.

.. mermaid:: /diagrams/user_guide/concepts/models_diagram_1.mmd

Forecasters
^^^^^^^^^^^

Forecasters are pure ML predictors. They wrap a specific algorithm (XGBoost,
LightGBM, GBLinear, etc.), receive a preprocessed :class:`~openstef_core.datasets.ForecastInputDataset`, and
return a :class:`~openstef_core.datasets.ForecastDataset`. No feature engineering or postprocessing happens inside
a Forecaster - it is solely responsible for the mathematical prediction step.

All forecasters implement the :class:`~openstef_models.models.forecasting.forecaster.Forecaster` interface with ``fit()`` and ``predict()``
methods.

.. note::

   **Output horizon guarantee**: a forecaster's ``predict()`` only returns timestamps
   in the interval ``[forecast_start, forecast_start + max_horizon]``. Input rows
   beyond ``max_horizon`` are silently discarded before the model runs. This is
   enforced by passing ``horizon=self.max_horizon`` to
   :meth:`~openstef_core.datasets.ForecastInputDataset.input_data`
   inside every concrete ``predict()`` implementation. You can therefore safely pass
   a :class:`~openstef_core.datasets.ForecastInputDataset` containing more data than
   the forecaster needs without worrying about spurious predictions far into the future.

Transforms
^^^^^^^^^^

Transforms are standalone pre- and postprocessing steps - lag features, holiday
indicators, datetime features, quantile sorting, and more. They compose into a
:class:`~openstef_core.mixins.TransformPipeline` which applies them sequentially. Transforms are stateless or
carry minimal fitted state (e.g., scalers).

Model
^^^^^

The Model (:class:`~openstef_models.models.ForecastingModel`) binds a preprocessing :class:`~openstef_core.mixins.TransformPipeline`, a
:class:`~openstef_models.models.forecasting.forecaster.Forecaster`, and a postprocessing :class:`~openstef_core.mixins.TransformPipeline` into a single saveable unit.
This is the core trainable object:

.. code-block:: python

   model = ForecastingModel(
       preprocessing=preprocess_pipeline,
       forecaster=forecaster,
       postprocessing=postprocess_pipeline,
       target_column="load",
   )

The model's ``fit()`` and ``predict()`` methods accept raw :class:`~openstef_core.datasets.TimeSeriesDataset`
objects and handle the full pipeline internally.

Workflow
^^^^^^^^

The Workflow (:class:`~openstef_models.workflows.CustomForecastingWorkflow`) wraps a Model and adds lifecycle
management: callbacks for MLflow storage, model reuse logic, model selection,
performance monitoring, experiment tagging, and run naming. This is where operational
concerns live - the Model stays focused on prediction.

Presets
^^^^^^^

For production use, :func:`~openstef_models.presets.create_forecasting_workflow` is an opinionated factory that
constructs a fully-wired :class:`~openstef_models.workflows.CustomForecastingWorkflow` from a
:class:`~openstef_models.presets.ForecastingWorkflowConfig`. Presets cover the majority of production use cases with
sensible defaults for preprocessing, feature engineering, and callbacks.

.. warning::

   OpenSTEF is **not** opinionated by default - the full configuration surface is
   exposed at every level. Presets *add* opinions for convenience. For research or
   experimentation, use the raw Workflow API for full configurability.


Model Selection Guide
---------------------

All forecasters in OpenSTEF support **quantile forecasting**, producing probabilistic
predictions at configurable quantiles. The exceptions are the Median and
Base Case forecasters, which produce only a single quantile.

.. seealso::

   For measured accuracy of these models on a public benchmark, see
   :ref:`Benchmark Results <benchmark_results>`.

.. list-table:: Forecaster Comparison
   :header-rows: 1
   :widths: 15 33 32 10 10

   * - Model
     - Strengths
     - Best For
     - Quantiles
     - Extrapolation
   * - :class:`XGBoost <openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster>`
     - Non-linear pattern capture; robust
     - General-purpose
     - Multi
     - No
   * - :class:`LightGBM <openstef_models.models.forecasting.lgbm_forecaster.LGBMForecaster>`
     - Fast training; low memory
     - General-purpose; large datasets
     - Multi
     - No
   * - :class:`GBLinear <openstef_models.models.forecasting.gblinear_forecaster.GBLinearForecaster>`
     - Extrapolates beyond training range
     - Congestion management
     - Multi
     - Yes
   * - :class:`LGBM Linear <openstef_models.models.forecasting.lgbmlinear_forecaster.LGBMLinearForecaster>`
     - Non-linear splits + linear leaves
     - Partial extrapolation
     - Multi
     - Partial
   * - Ensemble (via openstef-meta)
     - Complementary model combination
     - Best accuracy
     - Multi
     - Partial
   * - :class:`Constant Quantile <openstef_models.models.forecasting.constant_quantile_forecaster.ConstantQuantileForecaster>`
     - No features needed at prediction
     - Fallback
     - Multi
     - N/A
   * - :class:`Median <openstef_models.models.forecasting.median_forecaster.MedianForecaster>`
     - Robust; minimal assumptions
     - Stable loads; baseline
     - Single
     - N/A
   * - :class:`Base Case <openstef_models.models.forecasting.base_case_forecaster.BaseCaseForecaster>`
     - Zero cost; persistence
     - Baseline reference
     - Single
     - N/A

When to Use What
^^^^^^^^^^^^^^^^

**General-purpose forecasting**: Start with XGBoost or LightGBM. Both excel at
capturing non-linear patterns in load data (weather interactions, time-of-day effects,
calendar patterns). LightGBM trains faster on large datasets. For highest accuracy,
use the Ensemble approach (openstef-meta) which combines multiple forecasters.

**Congestion management**: Use GBLinear. Tree-based models cannot predict
values outside their training range - a critical limitation when forecasting peak loads
that may exceed historical maxima. GBLinear's linear structure allows natural
extrapolation.

**Best accuracy (production)**: Use the **Ensemble** approach (available via
openstef-meta). Combining tree-based and linear forecasters exploits their
complementary strengths - trees capture non-linear interactions while linear models
provide extrapolation capability and stability.

**Stable/predictable loads**: The Median forecaster provides a robust baseline with
minimal complexity. Useful for loads with very low variance or as a sanity-check
reference.

**Fallback/degraded mode**: The Constant Quantile forecaster learns fixed quantile values
per hour of day. It requires no input features at prediction time, making it suitable
as a last-resort fallback when data pipelines fail.


Choosing Your Abstraction Level
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Use Case
     - Recommended Component
     - Why
   * - Production deployment
     - Presets (:func:`~openstef_models.presets.create_forecasting_workflow`)
     - Sensible defaults, MLflow integration, model reuse
   * - Custom preprocessing research
     - :class:`~openstef_models.models.ForecastingModel`
     - Full control over transforms without lifecycle overhead
   * - Novel algorithm development
     - :class:`~openstef_models.models.forecasting.forecaster.Forecaster`
     - Implement the interface, plug into any higher level
   * - Operational monitoring
     - :class:`~openstef_models.workflows.CustomForecastingWorkflow`
     - Add callbacks without changing model logic

Most users start with Presets and only drop down to lower levels when they need
custom behavior. The component boundaries are designed so you can replace one piece
(e.g., swap a Forecaster or add a Transform) without rewriting the rest of the
pipeline.
