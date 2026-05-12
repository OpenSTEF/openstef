Core Concepts
=============

Understand the foundations of short-term energy forecasting and the design decisions behind OpenSTEF's machine learning framework.

**Forecasting Basics** (:doc:`forecasting_basics`)
   What short-term energy forecasting is, why it matters for grid operations, and how it differs from other prediction tasks.

**Quantiles and Confidence** (:doc:`quantiles_and_confidence`)
   How OpenSTEF produces probabilistic forecasts with uncertainty bands, and how to interpret quantile predictions.

**Feature Engineering** (:doc:`feature_engineering`)
   The domain-specific predictors OpenSTEF uses—weather data, time features, lag variables—and how they improve forecast accuracy.

**Reliability and Fallback** (:doc:`reliability_and_fallback`)
   How OpenSTEF handles production failures gracefully: fallback strategies for missing data, failed models, and degraded inputs.

**Meta-Ensembles** (:doc:`meta_ensembles`)
   Why combining multiple models outperforms any single approach, and how the ensemble architecture works.

**Component Splitting** (:doc:`component_splitting`)
   Decomposing aggregate load measurements into constituent energy components like solar, wind, and base load.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   forecasting_basics
   quantiles_and_confidence
   feature_engineering
   reliability_and_fallback
   meta_ensembles
   component_splitting
