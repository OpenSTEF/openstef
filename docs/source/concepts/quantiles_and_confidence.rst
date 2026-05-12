Probabilistic Forecasts and Quantiles
=====================================

Probabilistic forecasts go beyond a single "best guess" by expressing *uncertainty*
about future values. In energy forecasting, this uncertainty is critical: grid
operators need to know not just what load or generation is *expected*, but how much
it might deviate. OpenSTEF produces probabilistic forecasts using **quantiles**,
giving operators actionable information about forecast uncertainty.

This page explains what quantiles are, how OpenSTEF generates them, and how to
interpret prediction intervals for operational decision-making.

What Are Quantiles?
-------------------

A quantile divides a probability distribution at a specific point. The quantile
value ``q`` (between 0 and 1) represents the probability that the actual outcome
will fall *below* that value.

- **Quantile 0.1 (P10)** — there is a 10% chance the actual value will be below this
- **Quantile 0.5 (P50)** — the median; equally likely to be above or below
- **Quantile 0.9 (P90)** — there is a 90% chance the actual value will be below this

In OpenSTEF, quantiles are represented by the ``Quantile`` type:

.. code-block:: python

   from openstef_core.types import Quantile

   # Create quantiles directly
   q10 = Quantile(0.1)
   q50 = Quantile(0.5)
   q90 = Quantile(0.9)

   # Convert from percentile notation
   q95 = Quantile.from_percentile(95)

   # Get the complementary quantile (1 - q)
   q10.complementary()  # Returns Quantile(0.9)

.. mermaid:: /diagrams/concepts/quantiles_and_confidence_diagram_1.mmd

Prediction Intervals vs. Confidence Intervals
----------------------------------------------

These terms are often confused but have distinct meanings:

**Prediction interval**
   A range that is expected to contain a *future observation* with a given
   probability. In energy forecasting, the interval between P10 and P90 is an
   80% prediction interval — we expect 80% of actual load values to fall within
   this band.

**Confidence interval**
   A range that quantifies uncertainty about an *estimated parameter* (like a
   model coefficient). This is a property of the model fitting process, not of
   future observations.

OpenSTEF produces **prediction intervals**. When you see a forecast with P10 and
P90 bounds, this means: "We expect the actual load to fall between these values
approximately 80% of the time."

.. warning::

   The term "confidence interval" appears in some OpenSTEF class names (e.g.,
   ``ConfidenceIntervalApplicator``) for historical reasons, but the output is
   technically a prediction interval — it bounds future observations, not model
   parameters.

How OpenSTEF Generates Quantile Forecasts
-----------------------------------------

OpenSTEF supports two approaches to generating probabilistic forecasts:

Direct Quantile Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``MultiQuantileRegressor`` trains separate models for each quantile. Each model
optimizes a quantile-specific loss function (pinball loss), producing direct
estimates of each quantile level:

.. code-block:: python

   from openstef_models.utils.multi_quantile_regressor import MultiQuantileRegressor
   from lightgbm import LGBMRegressor

   # Train models for three quantiles simultaneously
   regressor = MultiQuantileRegressor(
       base_learner=LGBMRegressor,
       quantile_param="alpha",
       quantiles=[0.1, 0.5, 0.9],
       hyperparams={"objective": "quantile", "n_estimators": 200},
   )

   regressor.fit(X_train, y_train)

   # Predictions: array with shape (n_samples, 3) — one column per quantile
   predictions = regressor.predict(X_test)

This approach works with any scikit-learn compatible regressor that supports quantile
regression, including LightGBM and XGBoost.

Post-hoc Confidence Interval Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ConfidenceIntervalApplicator`` learns hour-specific uncertainty patterns from
validation errors and applies them to point forecasts. This is useful when your
model only produces a single point prediction:

.. code-block:: python

   from openstef_core.types import Quantile
   from openstef_models.transforms.postprocessing.confidence_interval_applicator import (
       ConfidenceIntervalApplicator,
   )

   applicator = ConfidenceIntervalApplicator(
       quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)]
   )

   # Learn uncertainty patterns from validation data
   applicator.fit((validation_data, validation_predictions))

   # Apply to new forecasts — adds quantile columns
   result = applicator.transform((new_input_data, new_predictions))
   # Result columns: ['quantile_P10', 'quantile_P50', 'quantile_P90']

This approach assumes forecast errors follow a normal distribution and learns
hour-specific standard deviations. It captures the fact that uncertainty varies
by time of day (e.g., solar forecasts are more uncertain at midday than at night).

.. mermaid:: /diagrams/concepts/quantiles_and_confidence_diagram_2.mmd

Quantile Ordering Invariant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenSTEF enforces that quantile predictions maintain proper ordering:

   P10 ≤ P50 ≤ P90

This invariant ensures that lower quantiles never exceed higher ones — a physical
impossibility that can occur with independently trained quantile models. The
``ConfidenceIntervalApplicator`` guarantees this by construction since it derives
all quantiles from a single standard deviation estimate.

Interpreting Quantile Forecasts
-------------------------------

When working with OpenSTEF forecast output, the quantile columns have a direct
operational interpretation:

.. code-block:: python

   from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

   fig = (
       ForecastTimeSeriesPlotter()
       .add_measurements(measurements=forecast_dataset.data["load"])
       .add_model(
           model_name="GBLinear",
           forecast=forecast.median_series,       # P50 prediction
           quantiles=forecast.quantiles_data,     # P10-P90 band
       )
       .plot()
   )

.. note:: [VISUALIZATION: An interactive time series plot showing actual load as a solid line, the P50 forecast as a dashed line, and the P10-P90 prediction interval as a shaded band around the forecast]

Reading the forecast:

- **Narrow band** — the model is confident; conditions are predictable (e.g., a
  clear winter weekday with stable weather)
- **Wide band** — high uncertainty; conditions are volatile (e.g., variable cloud
  cover affecting solar generation, or unusual demand patterns)
- **Actual outside the band** — this should happen roughly 20% of the time for an
  80% prediction interval. If it happens more often, the model is overconfident.

Why Quantiles Matter for Operations
-----------------------------------

Grid operators and energy traders use quantile forecasts differently depending on
their risk tolerance:

**Conservative operations (use P90 for demand, P10 for generation)**
   Ensures sufficient capacity is available with high probability. Useful for
   reliability-critical decisions like reserve scheduling.

**Cost-optimal operations (use P50)**
   Minimizes expected cost by planning around the median outcome. Appropriate when
   over- and under-estimation have symmetric costs.

**Risk-aware trading (use asymmetric quantiles)**
   Energy traders may use P25 or P75 depending on market position and imbalance
   penalty structures.

The choice of which quantiles to generate is configured at the model level:

.. code-block:: python

   from openstef_core.types import Quantile

   # Configure quantiles based on operational needs
   quantiles = [
       Quantile(0.05),   # Extreme low — for worst-case planning
       Quantile(0.10),   # P10
       Quantile(0.25),   # P25
       Quantile(0.50),   # Median
       Quantile(0.75),   # P75
       Quantile(0.90),   # P90
       Quantile(0.95),   # Extreme high — for worst-case planning
   ]

Evaluating Probabilistic Forecasts
-----------------------------------

Standard error metrics (MAE, RMSE) only evaluate point forecasts. For quantile
forecasts, OpenSTEF uses specialized metrics:

**Pinball loss (quantile score)**
   Measures how well each quantile is calibrated. A well-calibrated P10 should
   have roughly 10% of observations falling below it.

**Coverage**
   The fraction of actual values that fall within the prediction interval. An 80%
   interval (P10–P90) should achieve approximately 80% coverage.

**Interval width**
   Narrower intervals are more informative — but only if coverage is maintained.
   The goal is the narrowest interval that achieves the target coverage.

OpenSTEF provides metric providers that compute these across all quantiles:

.. code-block:: python

   from openstef_core.types import Quantile

   # Evaluate specific quantiles
   from openstef_beam.evaluation.metric_providers import MetricProvider

   # Custom metric providers can target specific quantiles
   provider = MetricProvider(quantiles=[Quantile(0.1), Quantile(0.9)])

Related Topics
--------------

- For an introduction to how forecasting works in OpenSTEF, see :doc:`forecasting_basics`
- For how features drive forecast accuracy, see :doc:`feature_engineering`
- For what happens when probabilistic models fail in production, see :doc:`reliability_and_fallback`