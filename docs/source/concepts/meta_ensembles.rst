Ensemble Forecasting
====================

This page explains how OpenSTEF combines multiple base forecasters into a single,
more accurate prediction through its meta-ensemble framework. You will learn the
motivation behind ensembling, how the two combiner strategies—``WeightsCombiner``
and ``StackingCombiner``—work internally, and when to choose one approach over
another for energy forecasting tasks.

For background on what short-term forecasting is and how quantile predictions work,
see :doc:`forecasting_basics` and :doc:`quantiles_and_confidence`.

Why Combine Multiple Forecasters?
---------------------------------

Individual forecasting models each have strengths and weaknesses. A gradient-boosted
tree may capture non-linear weather effects well but struggle with long-horizon
trends, while a linear model may generalize better during unusual conditions. By
combining several base forecasters, the ensemble can:

- **Reduce variance** — averaging over diverse models smooths out individual errors.
- **Improve robustness** — if one model degrades (e.g., due to a data quality issue),
  others compensate.
- **Cover more quantiles accurately** — different models may excel at different parts
  of the predictive distribution.

In energy forecasting specifically, load patterns shift with seasons, holidays, and
grid topology changes. No single model dominates across all conditions, making
ensembles a natural fit.

.. mermaid:: /diagrams/concepts/meta_ensembles_diagram_1.mmd

The EnsembleForecastingModel
----------------------------

The top-level class that orchestrates the ensemble is
``EnsembleForecastingModel``. It manages:

1. **Base forecasters** — a collection of independently trained models (e.g., LGBM,
   XGBoost, linear).
2. **A combiner** — a meta-model that learns how to merge base predictions into a
   final forecast.
3. **Preprocessing/postprocessing** — shared and model-specific data transformations.

During prediction, the ensemble first generates forecasts from each base model, then
passes those predictions to the combiner:

.. code-block:: python

   from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel

   # After fitting, prediction follows a two-stage process:
   # 1. Each base forecaster produces its own ForecastDataset
   # 2. The combiner merges them into the final output
   forecast = ensemble_model.predict(data)

   # You can also inspect per-model contributions:
   contributions = ensemble_model.predict_contributions(data)

The ``EnsembleForecastDataset`` is the intermediate representation that holds
predictions from all base models, organized by model name and quantile (columns
like ``lgbm|p50``, ``xgboost|p50``, etc.).

WeightsCombiner: Learned Classifier Weights
-------------------------------------------

The ``WeightsCombiner`` treats ensemble combination as a **classification problem**:
for each timestep, which base forecaster should be trusted most?

How It Works
^^^^^^^^^^^^

For each quantile, the ``WeightsCombiner`` trains a classifier (e.g., logistic
regression, random forest, XGBoost, or LGBM) that learns to assign weights to base
forecasters based on the input features and the base predictions themselves.

- **Hard selection** — the classifier picks the single best forecaster for each
  timestep (argmax of predicted class probabilities).
- **Soft selection** — the classifier outputs probability-like weights, and the final
  prediction is a weighted average of all base forecasters.

The soft approach is generally preferred because it produces smoother forecasts and
is more robust to classifier uncertainty.

.. code-block:: python

   from openstef_meta.models.forecast_combiners.weights_combiner import WeightsCombiner

   combiner = WeightsCombiner(
       hyperparams=combiner_lgbm_hyperparams,
       horizons=[0.25, 1.0, 4.0, 24.0, 47.0],
       quantiles=quantiles,
   )

   # Fit on an EnsembleForecastDataset (base model predictions + actuals)
   combiner.fit(data=ensemble_train_data, data_val=ensemble_val_data)

   # Predict: returns a ForecastDataset with combined predictions
   combined_forecast = combiner.predict(data=ensemble_test_data)

   # Inspect which models matter most
   importances = combiner.feature_importances

Supported classifier backends:

- ``"lgbm"`` — LightGBM classifier (default, good balance of speed and accuracy)
- ``"rf"`` — Random forest classifier
- ``"xgboost"`` — XGBoost classifier
- ``"logistic"`` — Logistic regression (fastest, most interpretable)

When to Use WeightsCombiner
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- You have **3+ diverse base forecasters** and want the combiner to learn which one
  to trust in different conditions.
- You value **interpretability** — feature importances reveal which model dominates
  and under what circumstances.
- You want a **lightweight** combiner that adds minimal inference latency.

StackingCombiner: Per-Quantile Meta-Regressors
----------------------------------------------

The ``StackingCombiner`` takes a different approach: it trains a **regression model
per quantile** that uses base forecaster outputs as input features. This is classical
stacking (also called "blending").

How It Works
^^^^^^^^^^^^

A template ``Forecaster`` instance is provided (e.g., an LGBM regressor). The
``StackingCombiner`` clones this template for each quantile and trains each clone to
predict the target at that quantile, using the base model predictions as features.

.. code-block:: python

   from openstef_meta.models.forecast_combiners.stacking_combiner import StackingCombiner
   from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster

   # Create a template meta-forecaster (cloned per quantile internally)
   template = LGBMForecaster(
       hyperparams=stacking_lgbm_hyperparams,
       horizons=[max(horizons)],
       quantiles=[quantiles[0]],  # placeholder; StackingCombiner overrides
   )

   combiner = StackingCombiner(
       meta_forecaster=template,
       horizons=horizons,
       quantiles=quantiles,
   )

   combiner.fit(data=ensemble_train_data, data_val=ensemble_val_data)
   combined_forecast = combiner.predict(data=ensemble_test_data)

Because each quantile has its own regressor, the stacking combiner can learn
**non-linear transformations** of base predictions—for example, it might learn that
for the 90th percentile, one model's p90 should be scaled up while another's should
be dampened.

When to Use StackingCombiner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- You need **maximum accuracy** and are willing to accept higher computational cost.
- Your base forecasters produce predictions that benefit from **non-linear
  recombination** (e.g., one model is biased high, another low).
- You have **sufficient training data** to avoid overfitting the meta-regressors.

.. mermaid:: /diagrams/concepts/meta_ensembles_diagram_2.mmd

Configuring Ensembles via Workflow Config
-----------------------------------------

In practice, you configure the ensemble type and combiner model through the workflow
configuration. The factory logic selects the appropriate combiner:

.. code-block:: python

   # Configuration determines the combiner type
   # ensemble_type: "learned_weights" or "stacking"
   # combiner_model: "lgbm", "rf", "xgboost", or "logistic"

   # Example: learned weights with LGBM classifier
   config.ensemble_type = "learned_weights"
   config.combiner_model = "lgbm"

   # Example: stacking with LGBM meta-regressor
   config.ensemble_type = "stacking"
   config.combiner_model = "lgbm"

Ensembles vs. Single Models: Trade-offs
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - Single Model
     - Ensemble
   * - Accuracy
     - Good for well-behaved loads
     - Better on diverse/volatile loads
   * - Training time
     - Fast (one model)
     - Slower (N models + combiner)
   * - Inference latency
     - Minimal
     - N× base + combiner overhead
   * - Interpretability
     - Direct feature importances
     - Two-level: base + combiner importances
   * - Maintenance
     - Simple retraining
     - Must retrain base models and combiner
   * - Robustness
     - Single point of failure
     - Graceful degradation

**Rules of thumb for energy forecasting:**

- For a **single, stable load** with good weather data, a well-tuned single model
  (e.g., LGBM) often suffices.
- For **aggregated or volatile loads**, or when you need to cover many prediction
  jobs with one configuration, ensembles consistently outperform.
- Start with ``WeightsCombiner`` (lower complexity), then try ``StackingCombiner``
  if you observe systematic biases in the combined output.

Practical Considerations
------------------------

**Data splitting for ensembles:** The combiner must be trained on predictions from
base models that did *not* see the combiner's training data during their own
training. OpenSTEF handles this through its ``data_splitter`` configuration, which
ensures proper temporal separation.

**Overfitting risk:** Stacking combiners with too many parameters relative to
training samples can overfit. Use regularized meta-regressors (e.g., GBLinear) or
limit tree depth in LGBM templates.

**Fallback behavior:** If a base forecaster fails during inference, the ensemble
framework can still produce predictions from the remaining models. See
:doc:`reliability_and_fallback` for details on production resilience strategies.

**Feature contributions:** Both combiners support ``predict_contributions()``,
which returns per-base-model contribution scores. This is valuable for monitoring
which models are driving the ensemble output over time.

.. code-block:: python

   # Inspect contributions to understand ensemble behavior
   contributions = ensemble_model.predict_contributions(data)
   # Returns a TimeSeriesDataset where each column is a base model's contribution

Summary
-------

OpenSTEF's meta-ensemble framework provides two complementary strategies for
combining base forecasters:

- **WeightsCombiner** — a classifier-based approach that learns to select or weight
  models, offering speed and interpretability.
- **StackingCombiner** — a regression-based approach that learns non-linear
  transformations of base predictions per quantile, offering maximum flexibility.

Both approaches improve forecast accuracy and robustness compared to single models,
at the cost of increased computational requirements. The choice between them depends
on your accuracy needs, available training data, and operational constraints.