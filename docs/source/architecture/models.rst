The openstef_models Package
===========================

This page provides a deep dive into the ``openstef_models`` package — the layer responsible for feature transforms, forecasting models, component splitting, explainability, and integrations with external systems like MLflow. It sits between the foundational ``openstef_core`` abstractions and the pipeline orchestration in ``openstef_beam``.

.. mermaid:: /diagrams/architecture/models_diagram_1.mmd

Overview
--------

The ``openstef_models`` package contains:

- **Transforms** — domain-organized feature engineering (weather, temporal, general)
- **Forecasting models** — concrete forecaster implementations
- **Component splitting** — decomposing energy signals into constituent parts
- **Explainability** — feature contributions and interpretable outputs
- **Mixins** — cross-cutting concerns (serialization, callbacks)
- **Integrations** — MLflow for model tracking and storage

All models build on the ``BaseModel`` and ``TransformPipeline`` abstractions from ``openstef_core`` (see :doc:`core`) and are orchestrated by pipelines in ``openstef_beam`` (see :doc:`beam`).

Transforms by Domain
---------------------

Transforms implement the ``TimeSeriesTransform`` interface from ``openstef_core`` and are organized into domain-specific subpackages. Each transform exposes ``fit()``, ``transform()``, and ``features_added()`` methods.

Weather Domain
^^^^^^^^^^^^^^

The ``weather_domain`` subpackage engineers features from raw meteorological data:

- ``AtmosphereDerivedFeaturesAdder`` — computes derived atmospheric variables (e.g., humidity indices, wind chill)
- ``DaylightFeatureAdder`` — extracts daylight duration and solar position features
- ``RadiationDerivedFeaturesAdder`` — derives radiation-based features (e.g., clear-sky index, diffuse fraction)

.. code-block:: python

   from openstef_models.transforms.weather_domain import (
       AtmosphereDerivedFeaturesAdder,
       DaylightFeatureAdder,
       RadiationDerivedFeaturesAdder,
   )

   # Each transform operates on a TimeSeriesDataset
   daylight = DaylightFeatureAdder()
   enriched_data = daylight.transform(data)
   print(daylight.features_added())  # e.g., ["daylight_hours", "solar_elevation"]

General Domain
^^^^^^^^^^^^^^

The ``general`` subpackage contains domain-agnostic transforms such as outlier handling:

.. code-block:: python

   from openstef_models.transforms.general.outlier_handler import OUTLIER_NAN_MASK_PREFIX

The outlier handler marks detected outliers with a NaN mask column (prefixed with ``OUTLIER_NAN_MASK_PREFIX``) so downstream models can handle them appropriately.

Transform Pipelines
^^^^^^^^^^^^^^^^^^^

Transforms are composed into pipelines using the ``TransformPipeline`` mixin from ``openstef_core``. This allows models to declare their preprocessing chain declaratively and ensures transforms are fitted and applied consistently during training and prediction.

Forecasting Models
------------------

Forecasting models live under ``openstef_models.models.forecasting`` and implement the ``Forecaster`` base class. All forecasters share a common interface:

- ``fit(data, data_val=None)`` — train on a ``TimeSeriesDataset``
- ``predict(data)`` — produce a ``ForecastDataset``
- ``hparams`` — typed hyperparameter configuration via ``HyperParams``

Base Case Forecaster
^^^^^^^^^^^^^^^^^^^^

The ``BaseCaseForecaster`` provides a simple but effective baseline by repeating weekly lag patterns:

.. code-block:: python

   from openstef_models.models.forecasting.base_case_forecaster import (
       BaseCaseForecaster,
       BaseCaseForecasterHyperParams,
   )

   forecaster = BaseCaseForecaster(hparams=BaseCaseForecasterHyperParams())
   forecaster.fit(train_data)
   forecast = forecaster.predict(input_data)

This model inherits from both ``Forecaster`` and ``ExplainableForecaster``, making it useful as a reference implementation.

Model Categories
^^^^^^^^^^^^^^^^

Forecasting models generally fall into these categories:

- **Baseline models** — simple heuristics (e.g., ``BaseCaseForecaster``)
- **ML-based models** — gradient boosting, neural networks, etc.
- **Ensemble models** — combine multiple forecasters (see :doc:`meta` for ``EnsembleForecastingModel``)

Each model declares its own ``HyperParams`` subclass for typed configuration and validation.

Component Splitting
-------------------

The ``ComponentSplittingModel`` decomposes aggregate energy signals into components (e.g., solar, wind, base load). It combines ``BaseModel`` with the ``ComponentSplitter`` interface:

.. code-block:: python

   from openstef_models.models.component_splitting_model import ComponentSplittingModel
   from openstef_core.datasets import TimeSeriesDataset, EnergyComponentDataset

   model = ComponentSplittingModel(config=splitting_config)
   model.fit(train_data)
   components: EnergyComponentDataset = model.predict(input_data)

The model orchestrates preprocessing, the splitting algorithm, and postprocessing into a unified pipeline. It validates required columns on input data and produces an ``EnergyComponentDataset`` with named component columns.

Explainability
--------------

Explainability is provided through two mixins in ``openstef_models.explainability.mixins``:

- ``ExplainableForecaster`` — interface for models that can explain their predictions
- ``ContributionsMixin`` — adds feature contribution computation to forecasters

Models that inherit from these mixins (like ``BaseCaseForecaster``) can produce per-feature contribution scores alongside their predictions, enabling transparency in forecast outputs.

.. code-block:: python

   # Any ExplainableForecaster can provide contributions
   from openstef_models.explainability.mixins import ExplainableForecaster

   # Check if a model supports explainability
   if isinstance(model, ExplainableForecaster):
       # Contributions are available after prediction
       pass

Mixins
------

The ``openstef_models.mixins`` package provides cross-cutting functionality:

ModelSerializer
^^^^^^^^^^^^^^^

Abstract interface for persisting and loading models:

.. code-block:: python

   from openstef_models.mixins import ModelSerializer, ModelIdentifier

``ModelSerializer`` defines ``serialize(model, file)`` and ``deserialize(file)`` methods, allowing different storage backends to be plugged in.

PredictorCallback
^^^^^^^^^^^^^^^^^

Lifecycle hooks for monitoring model workflows:

.. code-block:: python

   from openstef_models.mixins import PredictorCallback

   class LoggingCallback(PredictorCallback):
       def on_fit_start(self, context, data):
           print(f"Starting fit for {context}")

       def on_fit_end(self, context, result):
           print(f"Fit complete: {result}")

       def on_predict_start(self, context, data):
           print("Generating predictions...")

       def on_predict_end(self, context, data, result):
           print("Predictions ready")

Callbacks receive a ``WorkflowContext`` object containing execution state, making them suitable for logging, metrics collection, and triggering side effects.

Integrations
------------

MLflow
^^^^^^

The ``openstef_models.integrations.mlflow`` package provides model lifecycle management through MLflow:

.. code-block:: python

   from openstef_models.integrations.mlflow import MLFlowStorage, MLFlowStorageCallback

- ``MLFlowStorage`` — stores and retrieves models from the MLflow model registry
- ``MLFlowStorageCallback`` — a ``PredictorCallback`` that automatically logs models, metrics, and artifacts to MLflow during training

.. note::

   MLflow is an optional dependency. Install it separately when production model versioning and experiment tracking are required.

The callback integrates seamlessly with the ``PredictorCallback`` interface, so enabling MLflow tracking requires only adding the callback to your workflow — no changes to model code.

Relationship to Other Packages
------------------------------

``openstef_models`` depends on:

- **openstef_core** — base classes (``BaseModel``, ``TimeSeriesTransform``, ``TransformPipeline``, ``HyperParams``), dataset types, and validation. See :doc:`core`.
- **openstef_beam** — pipeline orchestration that composes models into training, prediction, and evaluation workflows. See :doc:`beam`.

The ``openstef_meta`` package builds *on top of* ``openstef_models``, combining multiple forecasters into ensemble models. See :doc:`meta`.