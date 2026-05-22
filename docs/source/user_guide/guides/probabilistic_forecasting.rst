Probabilistic Forecasting
=========================

A point forecast tells you what is most likely to happen, but it does not tell you how wrong it might be. If your model predicts 100 MW of load, reality could be anywhere from 80 to 120 MW. Grid operators need that range of possible outcomes to make sound decisions about reserve planning, congestion management, and market bidding. Probabilistic forecasting quantifies this uncertainty by producing multiple quantile predictions that describe the full distribution of possible outcomes.

This page explains how OpenSTEF produces calibrated probabilistic forecasts and why calibration matters for operational use.

Why Quantiles, Not Confidence Intervals
----------------------------------------

OpenSTEF expresses uncertainty through quantile forecasts. A quantile forecast at level 0.1 (P10) means "we expect the actual value to fall below this prediction about 10% of the time." A set of quantiles (e.g., P5, P10, P30, P50, P70, P90, P95) describes the shape of the predictive distribution without assuming normality or symmetry.

This is important for energy forecasting because:

- Solar generation uncertainty is highly asymmetric (bounded at zero, skewed by cloud cover)
- Wind power distributions are non-Gaussian due to the cubic relationship between wind speed and power
- Load uncertainty varies by time of day and season

Configuring Quantile Forecasts
------------------------------

All forecaster models in OpenSTEF support quantile prediction. You configure which quantiles to produce through the ``quantiles`` field on :class:`~openstef_models.presets.ForecastingWorkflowConfig`:

.. code-block:: python

   from openstef_core.types import Quantile as Q
   from openstef_models.presets import ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(
       model_id="my_forecast",
       model="gblinear",
       quantiles=[Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)],
   )

The median quantile (P50) serves as the point forecast. The outer quantiles define prediction intervals of varying width.

How Each Model Produces Quantiles
---------------------------------

Different model backends use different strategies to estimate quantiles:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Quantile Method
     - Key Detail
   * - XGBoost
     - Custom pinball loss objective
     - Multi-output: trains all quantiles jointly using gradient/hessian of pinball loss (or smoothed arctan variant)
   * - GBLinear
     - ``reg:quantileerror`` objective
     - XGBoost's built-in quantile regression for linear boosters
   * - LightGBM
     - ``MultiQuantileRegressor``
     - Native multi-quantile support in LightGBM

All approaches train a single model that produces all requested quantiles simultaneously, rather than fitting separate models per quantile. This is more efficient and helps maintain consistency between quantile levels.

.. note::

   The XGBoost pinball loss implementation includes a non-degenerate hessian approximation required for proper tree splitting. The ``max_delta_step`` hyperparameter must satisfy ``0.5 * max_delta_step <= min(quantile, 1 - quantile)`` for stable convergence.

The Calibration Problem
-----------------------

Raw quantile models often produce predictions where the stated coverage does not match observed coverage. For example, a predicted P10 might actually be exceeded 15% of the time rather than the expected 10%. This miscalibration can arise from:

- Distribution shift between training and deployment data
- Model misspecification (the model cannot perfectly capture the true conditional distribution)
- Feature interactions that affect uncertainty differently than the mean

For operational use, calibration matters enormously. If a grid operator uses P95 to set reserves, they need that bound to actually contain 95% of outcomes. Systematic miscalibration leads to either over-procurement (wasting money) or under-procurement (risking reliability).

Isotonic Quantile Calibration
-----------------------------

OpenSTEF addresses miscalibration with :class:`~openstef_models.transforms.postprocessing.IsotonicQuantileCalibrator`, a postprocessing transform that corrects quantile predictions after the model produces them.

The calibrator works as follows:

1. During training, it observes the model's predicted quantile values on a validation split alongside actual outcomes
2. For each quantile level, it fits a scikit-learn :class:`~sklearn.isotonic.IsotonicRegression` that maps predicted values to calibrated values
3. During prediction, each quantile column is passed through its learned isotonic mapping

Isotonic regression is ideal for this task because it is monotone by construction: if the raw model predicts a higher value, the calibrated output will also be higher (or equal). This preserves the ordering of quantiles, ensuring that P90 is always greater than or equal to P50, which is always greater than or equal to P10.

Adding Calibration to a Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibrator is appended to the postprocessing pipeline of a forecasting workflow:

.. code-block:: python

   from openstef_models.transforms.postprocessing import IsotonicQuantileCalibrator

   workflow.model.postprocessing.transforms.append(
       IsotonicQuantileCalibrator(
           quantiles=quantiles,
           use_local_quantile_estimation=True,
       )
   )

When ``use_local_quantile_estimation=True``, the calibrator estimates observed quantile levels using a local window approach rather than global statistics, which better captures conditional calibration.

.. warning::

   The calibrator must be fitted (via ``workflow.fit()``) before it can be used for prediction. Calling ``predict()`` on a workflow with an unfitted calibrator will raise a :class:`~openstef_core.exceptions.NotFittedError`.

For a complete worked example showing calibration before and after, including diagnostic plots, see :doc:`/tutorials/quantile_calibration`.

Evaluating Probabilistic Forecasts
-----------------------------------

Calibration quality can be assessed by comparing expected vs. observed quantile levels. For each quantile Q, compute the fraction of actual observations that fall below the predicted quantile value. A well-calibrated model produces a diagonal on a calibration plot (expected = observed).

Key metrics for probabilistic forecast quality include:

- **Calibration error**: the difference between expected and observed coverage per quantile
- **Sharpness**: the width of prediction intervals (narrower is better, given proper calibration)
- **Pinball loss**: the proper scoring rule for quantile forecasts, penalizing both miscalibration and lack of sharpness

See :doc:`/user_guide/guides/backtesting` for how to evaluate forecast quality on historical data.

.. seealso::

   - :doc:`/user_guide/guides/forecasting` for the overall forecasting workflow (fitting, predicting, model selection).
   - :doc:`/user_guide/concepts/models` for understanding how different model types compare.
   - :doc:`/user_guide/guides/backtesting` for evaluating forecast performance systematically.
   - :doc:`/user_guide/guides/reliability_fallback` for operational concerns like fallback behavior when data is missing.