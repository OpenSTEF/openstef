The openstef_meta Package
=========================

The ``openstef_meta`` package is the orchestration layer of OpenSTEF, combining multiple base forecasters into ensemble predictions. It depends on all three sibling packages—``openstef_core`` for data structures and transforms, ``openstef_models`` for individual forecasters, and ``openstef_beam`` for pipeline execution—to deliver ensemble forecasting workflows that outperform any single model.

This page covers the ensemble architecture: how base forecasters feed into forecast combiners, and how the workflow configuration presets wire everything together.

.. mermaid:: /diagrams/architecture/meta_diagram_1.mmd

EnsembleForecastingModel
------------------------

``EnsembleForecastingModel`` is the central class that orchestrates the ensemble pipeline. It inherits from ``BaseForecastingModel`` and implements a two-phase training strategy:

1. **Phase 1** — Fit each base forecaster independently and collect their in-sample predictions into an ``EnsembleForecastDataset``.
2. **Phase 2** — Fit the combiner on those collected predictions, learning how to optimally aggregate them.

At prediction time, the model runs all base forecasters in parallel, then passes their outputs through the fitted combiner to produce the final forecast.

.. code-block:: python

   from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel

   # After construction (typically via a workflow preset), the model exposes:
   model: EnsembleForecastingModel

   # Inspect configured base forecasters
   print(model.forecaster_names())       # e.g., ['lgbm_short', 'xgb_long', 'linear']
   print(model.forecaster_configs())     # dict[str, Forecaster]

   # Two-phase fit
   fit_result = model.fit(data=train_data, data_val=val_data)

   # Access per-component results
   for name, result in fit_result.component_fit_results().items():
       print(f"{name}: {result.metrics_to_flat_dict()}")

   # Predict
   forecast = model.predict(data=input_data)

Key properties of ``EnsembleForecastingModel``:

- ``quantiles`` — The quantiles produced by the ensemble (union of combiner capabilities).
- ``max_horizon`` — The maximum lead time supported across all base forecasters.
- ``component_hyperparams()`` — Hyperparameters for each base forecaster, keyed by name.
- ``get_explainable_components()`` — Returns forecasters that support contribution explanations.

ForecastCombiner ABC
--------------------

The ``ForecastCombiner`` abstract base class defines the contract for combining base forecaster predictions. It inherits from ``BaseConfig``, ``Predictor[EnsembleForecastDataset, ForecastDataset]``, and ``ExplainableForecaster``.

.. code-block:: python

   from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner

Every combiner must implement:

- ``fit(data, data_val, additional_features)`` — Learn combination weights or a meta-model from base forecaster predictions.
- ``predict(data, additional_features)`` — Produce final predictions from an ``EnsembleForecastDataset``.
- ``is_fitted`` — Whether the combiner is ready for prediction.
- ``hparams`` — The combiner's hyperparameters.
- ``predict_contributions(data, additional_features)`` — Explain each base forecaster's contribution to the final output.

OpenSTEF provides two concrete implementations.

WeightsCombiner
^^^^^^^^^^^^^^^

``WeightsCombiner`` uses a classification approach to learn per-quantile weights for each base forecaster. At each timestep, a classifier determines which forecaster is most likely to be closest to the true value, producing soft weights that blend the base predictions.

.. code-block:: python

   from openstef_meta.models.forecast_combiners.learned_weights_combiner import (
       WeightsCombiner,
       LGBMCombinerHyperParams,
       XGBCombinerHyperParams,
       RFCombinerHyperParams,
       LogisticCombinerHyperParams,
   )

   # WeightsCombiner supports multiple classifier backends:
   combiner = WeightsCombiner(
       hyperparams=LGBMCombinerHyperParams(),
       horizons=horizons,
       quantiles=quantiles,
   )

The classifier backends available are:

- ``LGBMCombinerHyperParams`` — LightGBM classifier (default, fast and accurate).
- ``XGBCombinerHyperParams`` — XGBoost classifier.
- ``RFCombinerHyperParams`` — Random Forest classifier.
- ``LogisticCombinerHyperParams`` — Logistic regression (lightweight, interpretable).

Each backend exposes a ``get_classifier()`` method that returns a scikit-learn compatible ``ClassifierMixin``. The ``feature_importances`` property provides per-quantile feature importance from the internal classifiers.

StackingCombiner
^^^^^^^^^^^^^^^^

``StackingCombiner`` trains a meta-forecaster on top of the base forecaster predictions. Rather than learning weights, it learns a full model that maps base predictions to final outputs—capturing non-linear interactions between forecasters.

.. code-block:: python

   from openstef_meta.models.forecast_combiners.stacking_combiner import StackingCombiner
   from openstef_models.forecasters.lgbm_forecaster import LGBMForecaster

   # Define a template meta-forecaster (cloned per-quantile internally)
   template = LGBMForecaster(
       hyperparams=lgbm_hyperparams,
       horizons=[max(horizons)],
       quantiles=[quantiles[0]],  # Single quantile — StackingCombiner clones per quantile
   )

   combiner = StackingCombiner(
       meta_forecaster=template,
       horizons=horizons,
       quantiles=quantiles,
   )

The ``StackingCombiner`` accepts any forecaster from ``openstef_models`` as its meta-learner, including ``LGBMForecaster`` and ``GBLinearForecaster``. The template is cloned per-quantile so each quantile gets its own trained meta-model.

EnsembleForecastingWorkflowConfig
---------------------------------

Rather than manually wiring together forecasters, combiners, and preprocessing, the ``EnsembleForecastingWorkflowConfig`` provides a declarative configuration that the ``create_ensemble_forecasting_workflow`` factory function translates into a complete ``CustomForecastingWorkflow``.

.. code-block:: python

   from openstef_meta.presets.forecasting_workflow import (
       EnsembleForecastingWorkflowConfig,
       create_ensemble_forecasting_workflow,
   )

   config = EnsembleForecastingWorkflowConfig(
       ensemble_type="weights",          # or "stacking"
       combiner_model="lgbm",            # classifier/meta-model backend
       horizons=horizons,
       quantiles=quantiles,
       target_column="load",
       location=location_config,
       # ... forecaster-specific and combiner-specific hyperparams
   )

   workflow = create_ensemble_forecasting_workflow(config)

The factory function handles the full assembly:

1. **Common preprocessing** — Validation checks, feature shifters, holiday features, datetime features, and standardizers via ``TransformPipeline``.
2. **Base forecasters** — Built from the config with model-specific preprocessing (e.g., different feature selections per forecaster).
3. **Combiner** — Instantiated based on ``ensemble_type`` and ``combiner_model``, with its own preprocessing (sample weighting and column selection).
4. **Postprocessing** — Quantile sorting, confidence interval application, and other output transforms.

Supported ``(ensemble_type, combiner_model)`` combinations:

- ``("weights", "lgbm")`` — WeightsCombiner with LightGBM classifier.
- ``("weights", "xgb")`` — WeightsCombiner with XGBoost classifier.
- ``("weights", "rf")`` — WeightsCombiner with Random Forest classifier.
- ``("weights", "logistic")`` — WeightsCombiner with logistic regression.
- ``("stacking", "lgbm")`` — StackingCombiner with LGBMForecaster meta-learner.
- ``("stacking", "gblinear")`` — StackingCombiner with GBLinearForecaster meta-learner.

Cross-Package Dependencies
--------------------------

The ``openstef_meta`` package sits at the top of the dependency hierarchy:

- **From openstef_core** — ``BaseConfig``, ``TimeSeriesDataset``, ``EnsembleForecastDataset``, ``ForecastDataset``, ``TransformPipeline``, ``LeadTime``, ``Quantile``, and the ``Predictor`` mixin. See :doc:`core`.
- **From openstef_models** — Base forecasters (``LGBMForecaster``, ``GBLinearForecaster``, etc.) and the ``ExplainableForecaster`` mixin for contribution explanations. See :doc:`models`.
- **From openstef_beam** — ``CustomForecastingWorkflow`` for pipeline execution, and ``MetricProvider`` implementations (``R2Provider``, ``ObservedProbabilityProvider``) for evaluation. See :doc:`beam`.

.. warning::

   Because ``openstef_meta`` depends on all three other packages, it should be imported only when ensemble functionality is needed. For single-model forecasting, use ``openstef_models`` and ``openstef_beam`` directly.

When to Use Ensembles
---------------------

Ensemble forecasting adds complexity. Use it when:

- Multiple model architectures capture different aspects of the signal (e.g., tree-based models for non-linear patterns, linear models for trend stability).
- You need robust uncertainty quantification across diverse model families.
- Forecast accuracy improvements justify the additional training and inference cost.

For simpler use cases, a single forecaster configured through ``openstef_models`` with a standard ``openstef_beam`` workflow is sufficient. See :doc:`models` and :doc:`beam` for those approaches.