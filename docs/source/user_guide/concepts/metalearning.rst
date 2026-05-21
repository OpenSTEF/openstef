Metalearning
============

This page explains how OpenSTEF combines multiple forecasting models into an ensemble that outperforms any individual model. It covers the motivation for metalearning, the architecture of :class:`EnsembleForecastingModel`, and the two combiner strategies available.

For a hands-on walkthrough, see :doc:`/tutorials/ensemble_forecasting`.

.. contents:: On this page
   :local:
   :depth: 2

Why Combine Models?
-------------------

Short-term energy forecasting faces a fundamental tension between two desirable properties:

- **Non-linear pattern learning** — Tree-based models (LightGBM, XGBoost) excel at capturing complex interactions between weather, time-of-day, and load patterns. However, they *cannot extrapolate* beyond the range of values seen during training.
- **Extrapolation capability** — Linear models (GBLinear) can predict values outside the training range because their output is an unbounded linear combination of inputs. However, they *cannot learn non-linear patterns*.

This tension matters most in two scenarios:

1. **Congestion management** — Grid operators care about peak loads. If a new peak exceeds all historical values, tree-based models will systematically underpredict it. A linear model can extrapolate to the new peak.
2. **Seasonal transitions** — When load patterns shift (e.g., first heat wave of summer), tree-based models lag behind because they haven't seen the new regime. Linear models adapt faster through their extrapolation ability.

OpenSTEF's metalearning layer addresses this by combining tree-based and linear forecasters, letting a *combiner* learn when each model's strengths apply.

Architecture Overview
---------------------

:class:`EnsembleForecastingModel` sits alongside :class:`ForecastingModel` as a sibling under :class:`BaseForecastingModel`. Both share the same API contract (``fit``, ``predict``, ``predict_contributions``), but the ensemble model orchestrates multiple base forecasters and a combiner.

.. mermaid:: /diagrams/user_guide/concepts/metalearning_diagram_1.mmd

The key phases during training are:

1. **Phase 1 — Fit base forecasters**: Each forecaster is trained independently on the same data (after its own preprocessing). In-sample predictions are collected.
2. **Phase 2 — Fit combiner**: The combiner learns from the base forecasters' in-sample predictions how to optimally combine them.

Per-Forecaster Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all models receive the same features. GBLinear requires aggressive feature filtering because global linear models suffer from collinearity when given many correlated lag features or one-hot encoded columns:

- **Most lag features are removed** — only ``load_lag_P7D`` (the 7-day lag) is retained
- **Holiday and datetime features are dropped** — one-hot and cyclic encodings create near-singular design matrices for linear models

This filtering is applied via :class:`Selector` steps in the per-forecaster preprocessing pipeline, configured automatically by :func:`create_ensemble_forecasting_workflow`.

Combiner Strategies
-------------------

OpenSTEF provides two combiner strategies, each training one model *per quantile*.

.. list-table:: Combiner Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - WeightsCombiner
     - StackingCombiner
   * - Approach
     - Classification: "which forecaster is best here?"
     - Regression: "what is the best prediction here?"
   * - Output
     - Weighted blend of base predictions
     - New prediction (may differ from all base predictions)
   * - Default model
     - LGBM classifier
     - LGBM regressor (template Forecaster)
   * - Interpretability
     - High — weights show model preference per timestep
     - Moderate — feature importances show which base model matters
   * - Configuration
     - ``ensemble_type="learned_weights"``
     - ``ensemble_type="stacking"``

.. mermaid:: /diagrams/user_guide/concepts/metalearning_diagram_2.mmd

WeightsCombiner
^^^^^^^^^^^^^^^

The :class:`WeightsCombiner` frames combination as a classification problem: "for this timestep and quantile, which base forecaster would have produced the lowest error?"

During training, it labels each historical timestep with the best-performing forecaster, then trains a classifier to predict those labels. At inference time, it uses ``predict_proba`` to obtain soft probabilities and computes a weighted mean of the base predictions:

.. code-block:: python

   # Conceptual logic (simplified)
   weights = classifier.predict_proba(base_predictions)  # shape: (n_samples, n_forecasters)
   final = (weights * base_predictions).sum(axis=1)

Available classifier backends include LGBM (default), Random Forest, XGBoost, and Logistic Regression — configured via ``combiner_model`` in :class:`EnsembleForecastingWorkflowConfig`.

StackingCombiner
^^^^^^^^^^^^^^^^

The :class:`StackingCombiner` takes a more flexible approach: it treats the base forecaster predictions as *input features* to a meta-regressor. This allows the combiner to learn non-linear relationships between base predictions and the optimal output — for example, it might learn to trust one model more when its prediction diverges significantly from another.

A template :class:`Forecaster` (e.g., :class:`LGBMForecaster`) is cloned once per quantile. Each clone is trained to predict the target directly from the base forecasters' outputs.

Configuration
-------------

Ensemble workflows are configured through :class:`EnsembleForecastingWorkflowConfig`, which extends the standard forecasting configuration with ensemble-specific settings:

.. code-block:: python

   from openstef_meta.presets import create_ensemble_forecasting_workflow

   workflow = create_ensemble_forecasting_workflow(config)

The ``config.base_models`` list specifies which forecasters to include (e.g., ``["lgbm", "gblinear"]``), while ``config.ensemble_type`` and ``config.combiner_model`` control the combination strategy.

See :doc:`/tutorials/ensemble_forecasting` for a complete worked example.

When to Use Ensemble Forecasting
---------------------------------

Ensemble forecasting adds complexity. Consider it when:

- **Peak accuracy matters** — congestion management use cases where underestimating peaks has operational consequences
- **The prediction horizon spans regime changes** — seasonal transitions, new connection points ramping up
- **Individual model performance is inconsistent** — one model wins in summer, another in winter

For simpler use cases where a single LightGBM model performs well, the standard :class:`ForecastingModel` (see :ref:`models`) is sufficient and faster to train.

Relationship to Other Concepts
------------------------------

- **Model architecture** — Ensemble models share the same :class:`BaseForecastingModel` interface as single models. See :doc:`models` for the full model taxonomy.
- **BEAM** — The backtest evaluation framework works with ensemble models the same way it works with single models. See :doc:`beam`.
- **Component splitting** — Ensemble forecasting operates on the total load signal. Component splitting (see :doc:`component_splitting`) is a separate concern applied downstream.