.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concept_metalearning:

Metalearning
============

This page explains the metalearning approach implemented in the ``openstef-meta`` package: why combining forecasters with complementary strengths produces better energy forecasts than any single model, and how :class:`~openstef_meta.models.ensemble_forecasting_model.EnsembleForecastingModel` orchestrates this combination.

For a hands-on walkthrough, see :doc:`/tutorials/ensemble_forecasting`.

Why Combine Models?
-------------------

Short-term energy forecasting presents a fundamental tension between two desirable properties:

- **Non-linear pattern capture.** Tree-based models (LightGBM, XGBoost) excel at learning complex interactions between weather, time-of-day, and load. However, decision trees partition the feature space into regions and predict constants within each region. They *cannot extrapolate* beyond the range of values seen during training.

- **Extrapolation.** Linear models (GBLinear) fit a global linear function. They cannot learn non-linear interactions, but they *can* extrapolate to values outside the training range because a linear function extends naturally.

This distinction matters most for:

- **Congestion management**, where peak values (often near or beyond historical maxima) drive operational decisions.
- **Seasonal transitions**, where load patterns shift to levels not recently observed in training data.

Neither model family alone covers both needs. A tree-based model underestimates unprecedented peaks; a linear model misses the nuanced patterns during stable periods. Metalearning combines them so that a learned combiner can trust the linear model when extrapolation is needed and the tree model when complex patterns dominate.

.. note::

   The term "metalearning" here refers to learning *how to combine* base model predictions, not meta-learning in the few-shot sense. The combiner is itself a learned model that operates on the outputs of other models.

Architecture
------------

:class:`~openstef_meta.models.ensemble_forecasting_model.EnsembleForecastingModel` is a **sibling** of :class:`~openstef_models.models.forecasting_model.ForecastingModel`, not a subclass. Both inherit from :class:`~openstef_models.models.forecasting_model.BaseForecastingModel` and share the same ``preprocessing → predict → postprocessing`` contract, but the ensemble model fans out to multiple forecasters internally.

.. mermaid:: /diagrams/user_guide/concepts/metalearning_diagram_1.mmd

The key stages are:

- **Common preprocessing.** Shared feature engineering (lag creation, holiday features, datetime encoding, standardization) runs once and produces a rich feature set for all forecasters.
- **Per-forecaster preprocessing.** Each base model receives a tailored view of the features. For GBLinear, this aggressively filters the feature set: it keeps only ``load_lag_P7D`` (removing all other lags to avoid collinearity) and drops holiday and datetime one-hot columns (which create near-singular design matrices for a linear model). Tree-based models receive the full feature set.
- **Parallel prediction.** Each forecaster independently produces quantile predictions over the forecast horizon.
- **Combination.** An :class:`~openstef_meta.models.ensemble_forecasting_model.EnsembleForecastingModel` merges all base predictions into an :class:`~openstef_core.datasets.validated_datasets.EnsembleForecastDataset`, then a :class:`~openstef_meta.models.forecast_combiners.forecast_combiner.ForecastCombiner` produces the final :class:`~openstef_core.datasets.validated_datasets.ForecastDataset`.

Per-Forecaster Feature Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-forecaster preprocessing for GBLinear illustrates a core design principle: the same raw data can require very different representations for different algorithms.

Each forecaster's preprocessing pipeline can include :class:`~openstef_models.transforms.general.selector.Selector` transforms, which are preprocessing steps that filter which features reach a given model. For GBLinear, these selectors aggressively reduce the feature set:

- Remove all lag features except ``load_lag_P7D`` (the 7-day lag capturing weekly seasonality). Multiple correlated lags would destabilize the linear model's coefficients.
- Remove holiday and datetime features, which create near-singular design matrices for a global linear fit.

The result is a small, well-conditioned feature set that lets GBLinear focus on the linear trend. Tree-based forecasters skip this filtering because they are inherently robust to correlated and redundant features.

Combiner Strategies
-------------------

The combiner is the "meta" in metalearning: it learns *when* to trust each base forecaster. OpenSTEF provides two strategies, configured via the ``ensemble_type`` parameter.

.. list-table:: Combiner Strategy Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Property
     - WeightsCombiner (``"learned_weights"``)
     - StackingCombiner (``"stacking"``)
   * - Mechanism
     - Classifier per quantile predicts which forecaster is best; uses ``predict_proba`` as soft weights
     - Meta-regressor per quantile takes base predictions as input features and directly predicts the target
   * - Output
     - Weighted average of base predictions
     - New prediction derived from base predictions
   * - Default model
     - LGBM classifier (also supports random forest, XGBoost, logistic regression)
     - LGBM regressor (a template Forecaster cloned per quantile)
   * - Interpretability
     - High (weights sum to 1; you can inspect which model is trusted at each timestep)
     - Lower (meta-regressor coefficients are less directly interpretable)
   * - When to prefer
     - When base models are individually strong and you want adaptive selection
     - When base model predictions contain complementary signal that a regressor can exploit beyond simple weighting

WeightsCombiner
^^^^^^^^^^^^^^^

.. mermaid:: /diagrams/user_guide/concepts/metalearning_diagram_2.mmd

The ``WeightsCombiner`` frames combination as a classification problem: "which forecaster will be closest to the true value at this timestep?" A classifier (default: LightGBM) is trained per quantile. At prediction time, ``predict_proba`` returns soft probabilities that serve as weights:

.. math::

   \hat{y} = \sum_{i=1}^{N} w_i \, \hat{y}_i

where :math:`\hat{y}` is the final combined forecast, :math:`\hat{y}_i` is the prediction from base forecaster *i*, :math:`w_i = P(\text{forecaster } i \text{ is best})` is the classifier's predicted probability that forecaster *i* will be closest to the true value, and *N* is the number of base forecasters.

This approach is adaptive: during stable periods the combiner may assign 80% weight to the tree-based model, while near historical peaks it shifts weight toward the linear model. The tutorial at :doc:`/tutorials/ensemble_forecasting` demonstrates how to visualize these weight dynamics.

StackingCombiner
^^^^^^^^^^^^^^^^

.. mermaid:: /diagrams/user_guide/concepts/metalearning_diagram_3.mmd

The :class:`~openstef_meta.models.forecast_combiners.stacking_combiner.StackingCombiner` treats base forecaster outputs as features for a second-stage regression model. A template :class:`~openstef_models.models.forecasting.forecaster.Forecaster` is cloned for each quantile, and each clone learns to map the vector of base predictions to the final target value. This is more expressive than weighted averaging (the meta-regressor can learn non-linear transformations of base predictions) but requires more training data to avoid overfitting.

Configuration
-------------

Ensemble models are configured through :class:`~openstef_meta.presets.EnsembleForecastingWorkflowConfig`:

.. code-block:: python

   from openstef_meta.presets import EnsembleForecastingWorkflowConfig

   config = EnsembleForecastingWorkflowConfig(
       base_models=["lgbm", "gblinear"],
       ensemble_type="learned_weights",  # or "stacking"
       combiner_model="lgbm",
   )

The ``base_models`` list determines which forecasters participate, ``ensemble_type`` selects the combiner strategy, and ``combiner_model`` specifies the algorithm used within the combiner. See :doc:`/tutorials/ensemble_forecasting` for a complete working example.

.. seealso::

   - :doc:`/user_guide/concepts/models` for details on individual model types (LightGBM, XGBoost, GBLinear).
   - :doc:`/user_guide/concepts/beam` for the metrics framework used to evaluate ensemble models.
   - :doc:`/user_guide/guides/forecasting` for the single-model workflow using the same :class:`~openstef_core.mixins.transform.TransformPipeline` pattern.