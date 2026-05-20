Models
======

This page explains OpenSTEF's model architecture — the layered design that separates
pure prediction from data transformation, lifecycle management, and production
configuration. Understanding these layers helps you choose the right level of
abstraction for your use case: from research experimentation with raw components to
production deployment with opinionated presets.

.. note::

   For how models are *selected automatically* via metalearning, see
   :doc:`metalearning`. For how models are *evaluated* in backtesting, see
   :doc:`beam`.


Architecture Layers
-------------------

OpenSTEF's model system is composed of five distinct layers, each adding
orchestration on top of the one below it. You can enter at any layer depending on
how much control you need.

.. mermaid:: /diagrams/user_guide/concepts/models_diagram_1.mmd

Layer 1: Forecasters
^^^^^^^^^^^^^^^^^^^^^

Forecasters are **pure ML predictors**. They wrap a specific algorithm (XGBoost,
LightGBM, GBLinear, etc.), receive a preprocessed ``ForecastInputDataset``, and
return a ``ForecastDataset``. No feature engineering or postprocessing happens inside
a Forecaster — it is solely responsible for the mathematical prediction step.

All forecasters implement the :class:`~openstef_models.models.forecasting.Forecaster`
interface with ``fit()`` and ``predict()`` methods.

Layer 2: Transforms
^^^^^^^^^^^^^^^^^^^^

Transforms are standalone pre- and postprocessing steps — lag features, holiday
indicators, datetime features, quantile sorting, and more. They are composed into a
:class:`~openstef_core.mixins.TransformPipeline` which applies them sequentially.
Transforms are stateless or carry minimal fitted state (e.g., scalers).

Layer 3: ForecastingModel
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~openstef_models.models.forecasting_model.ForecastingModel` binds a
preprocessing ``TransformPipeline``, a ``Forecaster``, and a postprocessing
``TransformPipeline`` into a single saveable unit. This is the core trainable object:

.. code-block:: python

   model = ForecastingModel(
       preprocessing=TransformPipeline(transforms=[...]),
       forecaster=forecaster,
       postprocessing=TransformPipeline(transforms=[...]),
       target_column="load",
   )

The model's ``fit()`` and ``predict()`` methods accept raw ``TimeSeriesDataset``
objects and handle the full pipeline internally.

Layer 4: CustomForecastingWorkflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~openstef_models.models.CustomForecastingWorkflow` wraps a
``ForecastingModel`` and adds **lifecycle management**: callbacks for MLflow storage,
model reuse logic, model selection, performance monitoring, experiment tagging, and
run naming. This is where operational concerns live — the Model stays focused on
prediction.

Layer 5: Presets
^^^^^^^^^^^^^^^^^

For production use, :func:`~openstef_models.create_forecasting_workflow` is an
opinionated factory that constructs a fully-wired ``CustomForecastingWorkflow`` from a
:class:`~openstef_models.ForecastingWorkflowConfig`. Presets cover ~99% of production
use cases with sensible defaults for preprocessing, feature engineering, and callbacks.

.. warning::

   OpenSTEF is **not** opinionated by default — the full configuration surface is
   exposed at every layer. Presets *add* opinions for convenience. For research or
   experimentation, use the raw Workflow API (Layers 1–4) for full configurability.


Model Selection Guide
---------------------

All forecasters in OpenSTEF support **quantile forecasting** — producing probabilistic
predictions at configurable quantiles. The exceptions are ``MedianForecaster`` and
``BaseCaseForecaster``, which produce only a single quantile.

.. list-table:: Forecaster Comparison
   :header-rows: 1
   :widths: 18 30 30 12 10

   * - Model
     - Strengths
     - Best For
     - Quantiles
     - Extrapolation
   * - :class:`~openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster`
     - Excellent non-linear pattern capture; robust to noise; well-understood
     - General-purpose load forecasting; complex non-linear relationships
     - Multi
     - No
   * - :class:`~openstef_models.models.forecasting.lgbm_forecaster.LGBMForecaster`
     - Fast training; memory efficient; strong non-linear performance
     - General-purpose; large datasets; rapid iteration
     - Multi
     - No
   * - :class:`~openstef_models.models.forecasting.gblinear_forecaster.GBLinearForecaster`
     - Linear model with gradient boosting; can extrapolate beyond training range
     - Congestion management; scenarios exceeding historical peaks
     - Multi
     - Yes
   * - :class:`~openstef_models.models.forecasting.lgbm_linear_forecaster.LGBMLinearForecaster`
     - Trees with linear leaves; balances expressiveness and extrapolation
     - Mixed linear/non-linear patterns; moderate extrapolation needs
     - Multi
     - Partial
   * - Ensemble (via openstef-meta)
     - Combines tree + linear models; complementary strengths reduce error
     - Production when runtime allows; highest accuracy
     - Multi
     - Partial
   * - MedianForecaster
     - Simple; robust; minimal data requirements
     - Very stable/predictable loads; baseline comparison
     - Single
     - N/A
   * - BaseCaseForecaster
     - Persistence-based; zero training cost
     - Benchmarking; fallback
     - Single
     - N/A

When to Use What
^^^^^^^^^^^^^^^^

**General-purpose forecasting**: Start with XGBoost or LightGBM. Both excel at
capturing non-linear patterns in load data (weather interactions, time-of-day effects,
calendar patterns). LightGBM trains faster on large datasets.

**Congestion management**: Use :class:`~openstef_models.models.forecasting.gblinear_forecaster.GBLinearForecaster`.
Tree-based models cannot predict values outside their training range — a critical
limitation when forecasting peak loads that may exceed historical maxima. GBLinear's
linear structure allows natural extrapolation.

**Best accuracy (production)**: Use the **Ensemble** approach (available via
openstef-meta). Combining tree-based and linear forecasters exploits their
complementary strengths — trees capture non-linear interactions while linear models
provide extrapolation capability and stability.

**Stable/predictable loads**: ``MedianForecaster`` provides a robust baseline with
minimal complexity. Useful for loads with very low variance or as a sanity-check
reference.

.. note::

   OpenSTEF also scales beyond electricity — Sigholm uses it for 40% of Sweden's
   district heating production, demonstrating applicability to thermal energy domains.


Choosing Your Abstraction Level
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Use Case
     - Recommended Layer
     - Why
   * - Production deployment
     - Presets (``create_forecasting_workflow``)
     - Sensible defaults, MLflow integration, model reuse
   * - Custom preprocessing research
     - ``ForecastingModel`` (Layer 3)
     - Full control over transforms without lifecycle overhead
   * - Novel algorithm development
     - ``Forecaster`` (Layer 1)
     - Implement the interface, plug into any higher layer
   * - Operational monitoring
     - ``CustomForecastingWorkflow`` (Layer 4)
     - Add callbacks without changing model logic


Further Reading
---------------

- :doc:`/tutorials/ensemble_forecasting` — worked example of ensemble model setup
- :doc:`metalearning` — automatic model selection across prediction jobs
- :doc:`beam` — backtesting and benchmarking framework for model evaluation
- :doc:`component_splitting` — decomposing forecasts into interpretable components