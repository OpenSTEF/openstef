Models
======

This page explains the model architecture in OpenSTEF: how individual components compose into production-ready forecasting systems, what models are available, and how to choose between them.

Understanding the architecture helps you decide whether to use a preset (the common case) or assemble custom pipelines when your use case demands it.

Containment Hierarchy
---------------------

OpenSTEF organizes forecasting logic into nested layers of responsibility. Each outer layer adds capabilities around the inner one.

.. mermaid:: /diagrams/user_guide/concepts/models_diagram_1.mmd

Forecaster
^^^^^^^^^^

A :class:`~openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster` (or any other forecaster) is a pure ML predictor. It receives preprocessed features and returns quantile predictions. It has no transforms inside; it only knows how to ``fit()`` and ``predict()``.

All forecasters share a common interface: they accept quantiles, horizons, and hyperparameters at construction time.

ForecastingModel
^^^^^^^^^^^^^^^^

:class:`~openstef_models.models.forecasting_model.ForecastingModel` binds three components into a single saveable unit:

- A **preprocessing** :class:`~openstef_core.mixins.TransformPipeline` (feature engineering, imputation, scaling)
- A **Forecaster** (the ML predictor)
- A **postprocessing** :class:`~openstef_core.mixins.TransformPipeline` (quantile sorting, confidence intervals)

When you call ``predict()``, the model applies preprocessing, passes the result to the forecaster, then applies postprocessing. When you call ``fit()``, preprocessing is fitted first, then the forecaster trains on the transformed data.

.. warning::

   When using lag-based transforms in preprocessing, you must set ``cutoff_history`` to exclude rows with NaN values created by the lag window. This cannot be inferred automatically.

CustomForecastingWorkflow
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow` wraps a ``ForecastingModel`` with production concerns:

- **Callbacks** (MLflow storage, model performance monitoring, data saving)
- **model_id** for tracking and persistence
- **Run lifecycle** (naming, experiment tags, deep-copy for parallel runs)

This is the main entry point for production systems where models need to be trained, saved, loaded, and monitored.

Presets
^^^^^^^

The ``ForecastingWorkflowConfig`` combined with ``create_forecasting_workflow`` is an opinionated factory that constructs a fully-wired ``CustomForecastingWorkflow`` from declarative configuration. It selects appropriate transforms, wires callbacks (MLflow, model reuse, performance thresholds), and applies sensible defaults.

For most users, presets are the recommended starting point.

Building Blocks: Forecasters and Transforms
--------------------------------------------

Forecasters and transforms sit at the same abstraction level; both are building blocks that ``ForecastingModel`` assembles. You can mix and match them freely.

Available Transforms
^^^^^^^^^^^^^^^^^^^^

Transforms are organized by domain:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Domain
     - Purpose
   * - **General**
     - Imputation, scaling, feature selection
   * - **Time domain**
     - Lags, cyclic features (hour-of-day, day-of-week), holidays
   * - **Weather domain**
     - Derived weather features (wind chill, radiation components)
   * - **Energy domain**
     - Wind power curves, solar position
   * - **Validation**
     - Completeness checks, flatliner detection
   * - **Postprocessing**
     - Quantile sorting, confidence interval construction

Model Selection
---------------

All forecasters support quantile (probabilistic) forecasting. The choice depends on your use case, data characteristics, and runtime constraints.

.. list-table::
   :header-rows: 1
   :widths: 20 35 25 20

   * - Model
     - Best for
     - Key property
     - Trade-off
   * - :class:`~openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster`
     - General-purpose forecasting
     - Best non-linear pattern capture
     - Cannot extrapolate beyond training range
   * - :class:`~openstef_models.models.forecasting.lgbm_forecaster.LGBMForecaster`
     - General-purpose forecasting
     - Fast training, good non-linear capture
     - Cannot extrapolate beyond training range
   * - :class:`~openstef_models.models.forecasting.gblinear_forecaster.GBLinearForecaster`
     - Congestion management
     - Can extrapolate beyond training range
     - Less expressive for non-linear patterns
   * - :class:`~openstef_models.models.forecasting.lgbm_linear_forecaster.LGBMLinearForecaster`
     - Balanced expressiveness
     - Tree structure with linear leaves
     - Balances extrapolation and non-linearity
   * - Ensemble (openstef-meta)
     - Production forecasting
     - Tree + linear models complement each other
     - Higher runtime cost
   * - :class:`~openstef_models.models.forecasting.median_forecaster.MedianForecaster`
     - Stable, predictable loads
     - Simple baseline
     - No feature responsiveness
   * - :class:`~openstef_models.models.forecasting.flatliner_forecaster.FlatlinerForecaster`
     - Fallback scenarios
     - Returns zero forecast
     - Utility model only
   * - :class:`~openstef_models.models.forecasting.constant_quantile_forecaster.ConstantQuantileForecaster`
     - Fallback scenarios
     - Returns constant quantile values
     - Utility model only

Choosing a Model
^^^^^^^^^^^^^^^^

**Ensemble is preferred when runtime allows.** Tree-based models (XGBoost, LGBM) excel at capturing non-linear load patterns, while linear models (GBLinear) handle extrapolation. Combining them in an ensemble yields robust forecasts across diverse conditions. See :doc:`/tutorials/ensemble_forecasting` for a worked example.

**Use GBLinear for congestion management.** When grid operators need to predict peak loads that may exceed historical maxima, tree-based models saturate at the training range. GBLinear's linear structure allows it to extrapolate, making it the right choice for capacity planning and congestion alerts.

**Use LGBMLinear as a middle ground.** It combines tree structure with linear leaves, offering more expressiveness than pure linear models while retaining some extrapolation capability.

**Use MedianForecaster for stable baselines.** For loads with minimal variability (e.g., industrial baseload), a simple median predictor may suffice and provides a useful benchmark.

**Use FlatlinerForecaster and ConstantQuantileForecaster for fallback.** These utility models are used by the :doc:`/user_guide/guides/reliability_fallback` system when primary models fail or data quality is insufficient.

Relationship to Other Concepts
------------------------------

- For how ensemble models are selected and weighted automatically, see :doc:`/user_guide/concepts/metalearning`.
- For how model performance is tracked and evaluated, see :doc:`/user_guide/concepts/beam`.
- For probabilistic output interpretation, see :doc:`/user_guide/guides/probabilistic_forecasting`.