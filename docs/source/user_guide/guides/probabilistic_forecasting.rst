Probabilistic Forecasting
=========================

.. _guide_probabilistic_forecasting:

A point forecast tells you what is *most likely* to happen — "expect 100 MW at 14:00 tomorrow." But grid operators cannot act on a single number alone. If reality could be anywhere between 80 MW and 120 MW, that uncertainty range drives critical decisions:

- **Reserve planning** — how much backup capacity to procure
- **Congestion management** — whether a line might overload under plausible scenarios
- **Market bidding** — balancing risk of over- vs. under-procurement

Probabilistic forecasts express this uncertainty as a set of *quantiles*: the P10 value should be exceeded roughly 90% of the time, the P90 value only about 10% of the time. Together, these quantiles form prediction intervals that communicate the full range of plausible outcomes.

OpenSTEF treats probabilistic forecasting as a first-class capability. Every supported model produces quantile predictions natively — this is not bolted on as an afterthought.

.. mermaid:: /diagrams/user_guide/guides/probabilistic_forecasting_diagram_1.mmd

How Quantile Forecasting Works
------------------------------

Traditional regression minimises squared error, producing a conditional mean. Quantile regression instead minimises the *pinball loss* (also called quantile loss), which asymmetrically penalises over- and under-predictions depending on the target quantile level. For quantile τ:

- Under-predictions are penalised by weight τ
- Over-predictions are penalised by weight (1 − τ)

This causes the model to learn the τ-th conditional quantile of the target distribution rather than the mean.

OpenSTEF implements this differently depending on the model backend:

.. list-table:: Quantile Regression by Model Type
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Approach
   * - XGBoost
     - Custom multi-quantile pinball loss objective (``pinball_loss_multi_objective``) or arctan-smoothed variant (``arctan_loss_multi_objective``). All quantiles are predicted simultaneously in a single multi-output model.
   * - GBLinear
     - XGBoost's built-in ``reg:quantileerror`` objective, which natively supports quantile regression.
   * - LightGBM
     - ``MultiQuantileRegressor`` wrapper that trains quantile predictions using LightGBM's quantile objective.

All approaches produce a single model that outputs all requested quantiles at once, rather than training separate models per quantile.


Configuring Quantiles
---------------------

Quantiles are specified when setting up a forecasting workflow via the ``quantiles`` field on your workflow configuration. The standard set used across OpenSTEF benchmarks and examples is:

.. code-block:: python

   from openstef_core.types import Quantile as Q

   quantiles = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

The P50 quantile corresponds to the median forecast. The outer quantiles (P05, P95) define a 90% prediction interval — you expect the actual value to fall within this range 90% of the time.

Output columns in the forecast dataset are named ``quantile_P05``, ``quantile_P10``, etc., making them easy to identify and process downstream.

.. note::

   The choice of quantiles depends on your operational needs. For reserve planning you may care most about extreme tails (P01, P99). For day-ahead market bidding, P10–P90 may suffice. OpenSTEF lets you specify any set of quantiles between 0 and 1.


The Calibration Problem
-----------------------

Raw quantile models often *miscalibrate*: the predicted P10 value might actually be exceeded 15% of the time rather than the expected 10%. This happens because:

- The model's loss landscape has local minima
- Feature distributions shift between training and deployment
- The pinball loss converges slowly in the tails where data is sparse

Miscalibration means your prediction intervals are unreliable — a "90% interval" might only cover 80% of outcomes, leading to under-estimated risk.

.. note:: [VISUALIZATION: Reliability diagram (calibration plot) showing expected quantile level on x-axis vs. observed frequency on y-axis. A perfectly calibrated model follows the diagonal. The "before calibration" line deviates, the "after calibration" line hugs the diagonal.]


Isotonic Quantile Calibration
-----------------------------

OpenSTEF addresses miscalibration with :class:`~openstef_models.transforms.postprocessing.IsotonicQuantileCalibrator`, a postprocessing transform that restores proper quantile coverage.

**How it works:**

1. During training, the calibrator observes predicted quantile values on a validation split and computes the *empirical* quantile level each prediction actually corresponds to.
2. It fits a ``sklearn.isotonic.IsotonicRegression`` per quantile, learning a monotonic mapping from predicted values to calibrated values.
3. During prediction, each quantile column is remapped through its learned isotonic function.

This approach has two key properties:

- **Proper coverage** — after calibration, the predicted P10 is exceeded approximately 10% of the time
- **Monotonicity preservation** — isotonic regression guarantees that higher quantiles always produce values ≥ lower quantiles (no "crossing" quantiles)

Adding Calibration to a Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibrator is appended to the model's postprocessing pipeline:

.. code-block:: python

   from openstef_models.transforms.postprocessing import IsotonicQuantileCalibrator

   workflow.model.postprocessing.transforms.append(
       IsotonicQuantileCalibrator(
           quantiles=quantiles,
           use_local_quantile_estimation=True,
       )
   )

When ``use_local_quantile_estimation=True``, the calibrator estimates empirical quantile levels using a local rolling window (minimum 20 samples), which adapts better to non-stationary data than a global estimate.

After calling ``workflow.fit()``, the calibrator is fitted on the validation portion of the training data. Subsequent calls to ``workflow.predict()`` automatically apply the calibration correction.

.. warning::

   Calibration requires sufficient validation data to estimate empirical quantile levels reliably. With fewer than ~200 validation samples, the isotonic mapping may overfit. Ensure your training dataset includes an adequate validation split.


Evaluating Probabilistic Forecasts
-----------------------------------

Point forecast metrics (MAE, RMSE) do not capture probabilistic forecast quality. Key metrics for quantile forecasts include:

.. list-table:: Probabilistic Forecast Metrics
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - What it measures
   * - Pinball loss (quantile score)
     - Average asymmetric error per quantile — lower is better
   * - Coverage
     - Fraction of actuals falling within a prediction interval — should match the nominal level
   * - Winkler score
     - Interval width penalised for non-coverage — rewards tight intervals that still cover
   * - Calibration error
     - Deviation between expected and observed quantile levels — should be near zero

To assess calibration, compare the expected quantile level against the observed frequency:

.. code-block:: python

   observed_p10 = (actuals <= forecast["quantile_P10"]).mean()
   # Should be approximately 0.10

For systematic evaluation across multiple prediction jobs, see :doc:`/user_guide/guides/backtesting`.


When to Use Probabilistic Forecasts
------------------------------------

Not every application needs full quantile predictions. Use this decision framework:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Use Case
     - Point Forecast Sufficient?
     - Quantiles Needed?
   * - Dashboard display
     - Often yes
     - Nice-to-have
   * - Reserve procurement
     - No
     - Yes (tails: P01–P10, P90–P99)
   * - Congestion risk assessment
     - No
     - Yes (upper tail: P90, P95)
   * - Market bidding
     - Sometimes
     - Yes for optimal bidding
   * - Model comparison / benchmarking
     - Partial
     - Yes for full evaluation


Next Steps
----------

- :doc:`/tutorials/quantile_calibration` — worked example showing calibration before and after isotonic correction
- :doc:`/user_guide/guides/forecasting` — the full forecasting lifecycle including model selection
- :doc:`/user_guide/guides/backtesting` — evaluating forecast quality on historical data