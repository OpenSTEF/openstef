.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guide_probabilistic_forecasting:

Probabilistic Forecasting
=========================

A point forecast tells you what is most likely to happen, but it does not tell
you how wrong it might be. If your model predicts 100 MW of load, reality could
be anywhere from 80 to 120 MW. Grid operators need that range of possible
outcomes to make sound decisions about reserve planning, congestion management,
and market bidding. Probabilistic forecasting quantifies this uncertainty by
producing multiple quantile predictions that describe the full distribution of
possible outcomes.

This page explains how OpenSTEF produces calibrated probabilistic forecasts and
why calibration matters for operational use. For a runnable end-to-end example
with diagnostic plots, see :doc:`/tutorials/quantile_calibration`.

What You Get Back
-----------------

When you configure quantiles on a workflow, ``predict()`` returns a
:class:`~openstef_core.datasets.ForecastDataset` whose ``data`` DataFrame
contains:

- One ``quantile_PXX`` column per requested quantile level (e.g.,
  ``quantile_P10``, ``quantile_P50``, ``quantile_P90``).
- The median (P50) doubles as the point forecast.
- All quantiles share the same datetime index and horizon metadata as the
  input.

Quantiles are sorted at postprocessing time
(:class:`~openstef_models.transforms.postprocessing.quantile_sorter.QuantileSorter`),
so you can rely on ``quantile_P10 <= quantile_P50 <= quantile_P90`` row-by-row.

.. figure:: /images/guides/probabilistic_fan_chart.svg
   :alt: A probabilistic forecast showing the median load prediction with
         nested confidence bands at P5-P95, P10-P90, and P30-P70.
   :align: center

   A real GBLinear forecast with seven quantiles. The dark band is the
   inter-quartile range (P30-P70); the lighter bands widen out to the
   90% and 95% prediction intervals. Width varies with the time of day:
   uncertainty is higher around the morning ramp than during the
   overnight trough.

Why Quantiles, Not Confidence Intervals
----------------------------------------

OpenSTEF expresses uncertainty through quantile forecasts. A quantile forecast at level 0.1 (P10) means "we expect the actual value to fall below this prediction about 10% of the time." A set of quantiles (e.g., P5, P10, P30, P50, P70, P90, P95) describes the shape of the predictive distribution without assuming normality or symmetry.

This is important for energy forecasting because:

- Solar generation uncertainty is highly asymmetric (bounded at zero, skewed by cloud cover)
- Wind power distributions are non-Gaussian due to the cubic relationship between wind speed and power
- Load uncertainty varies by time of day and season

Why Not Full Probability Distributions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A natural question is why OpenSTEF outputs a discrete set of quantiles rather
than a parametric distribution (e.g., a Gaussian with predicted mean and
variance). The choice is deliberate, for three reasons:

1. **Most production-grade gradient-boosting models do not output
   distributions natively.** XGBoost and LightGBM are tree ensembles; they
   produce point estimates. They can be adapted to predict at a configured set
   of quantile levels (via pinball loss or built-in quantile objectives), but
   they cannot emit a continuous density. Coercing them into a parametric
   distribution would require either an inappropriate normality assumption or
   a separate two-stage model that estimates mean and spread.
2. **There is no standard distribution shape for energy load or generation.**
   Solar generation is bounded at zero with a heavy tail on sunny days; wind
   power has a cubic relationship to wind speed; aggregated load looks
   roughly normal but mixes regimes. Picking a single parametric family would
   force a poor fit somewhere. Quantiles are non-parametric and adapt to
   whatever shape the data has.
3. **Operational decisions are usually quantile-based anyway.** Reserve
   procurement, congestion bidding, and risk-of-imbalance calculations
   consume specific percentiles directly. A grid operator asks "what is the
   95th percentile of expected load?", not "what is the variance of a fitted
   normal?". Quantiles cut out a lossy intermediate step.

If you do need a continuous distribution downstream, you can always
interpolate between the quantiles OpenSTEF returns (linear interpolation
between adjacent quantiles, extrapolating beyond P5/P95 if needed). The
reverse, recovering a non-parametric shape from a fitted distribution, is
much harder.

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

Different backends use different strategies to estimate quantiles. All of them
train a single model that produces all requested quantiles simultaneously,
rather than fitting one model per quantile. This is more efficient and helps
maintain consistency between levels.

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Model
     - Quantile method
     - Pick when
     - Key detail
   * - **XGBoost**
     - Custom pinball loss objective
     - Non-linear targets where you want full control over the loss (e.g.,
       smoothed arctan variant for sharper gradients).
     - Multi-output: trains all quantiles jointly using gradient/hessian of
       pinball loss.
   * - **GBLinear**
     - ``reg:quantileerror`` objective
     - Fast iteration on linear-tractable problems, or as a robust baseline.
     - Uses XGBoost's built-in quantile regression for linear boosters.
   * - **LightGBM**
     - ``MultiQuantileRegressor``
     - Large feature sets where LightGBM's training speed pays off.
     - Native multi-quantile support inside LightGBM.

.. warning::

   The XGBoost pinball loss includes a non-degenerate hessian approximation
   that gradient boosting needs for proper tree splitting. Set
   ``max_delta_step`` such that
   ``0.5 * max_delta_step <= min(quantile, 1 - quantile)`` across your
   configured quantiles, otherwise training can diverge.

The Calibration Problem
-----------------------

Raw quantile models often produce predictions where the stated coverage does not match observed coverage. For example, a predicted P10 might actually be exceeded 15% of the time rather than the expected 10%. This miscalibration can arise from:

- Distribution shift between training and deployment data
- Model misspecification (the model cannot perfectly capture the true conditional distribution)
- Feature interactions that affect uncertainty differently than the mean

For operational use, calibration matters enormously. If a grid operator uses P95 to set reserves, they need that bound to actually contain 95% of outcomes. Systematic miscalibration leads to either over-procurement (wasting money) or under-procurement (risking reliability).

.. figure:: /images/guides/calibration_plot.svg
   :alt: A calibration plot showing forecasted versus observed probability
         for seven quantile levels. Points off the diagonal indicate
         miscalibration.
   :align: center
   :width: 70%

   Calibration check: for each requested quantile (x-axis), the y-axis shows
   the fraction of actual observations that fell below it. A perfectly
   calibrated model lands on the dashed diagonal. Deviations above the
   diagonal mean the model is *over-predicting* (predicted P50 actually
   covers 60% of outcomes); below means *under-predicting*.

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

MAPIE Quantile Calibration
--------------------------

For conformal calibration of individual quantiles, OpenSTEF also provides
:class:`~openstef_models.transforms.postprocessing.MapieQuantileCalibrator`.
Unlike conformalized quantile regression for a prediction interval, this
transform calibrates each requested quantile independently. It can therefore be
used with arbitrary quantile sets, such as P10, P30, P50, P70, and P90, without
requiring complementary outer quantiles or a P50 median.

The transform uses a separate signed MAPIE residual calibration for every
quantile:

1. During fitting, it compares each forecast quantile with the observed target
   on a calibration dataset.
2. MAPIE computes a quantile-specific conformal residual correction.
3. During prediction, the learned correction is applied only to the matching
   ``quantile_PXX`` column.

Because the corrections are estimated independently, the transform sorts the
calibrated quantile columns once more before returning the forecast. This keeps
the required monotonic ordering when independent corrections would otherwise
produce quantile crossings.

For the underlying conformal prediction concepts and calibration details, see
the `MAPIE documentation <https://mapie.readthedocs.io/en/stable/>`_.

Usage: Fitting on a Held-Out Calibration Period
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   Appending :class:`~openstef_models.transforms.postprocessing.MapieQuantileCalibrator`
   to the postprocessing pipeline and then calling ``workflow.fit()`` **does not**
   provide an independent calibration set. ``ForecastingModel.fit`` fits
   postprocessing transforms on predictions for ``input_data_train``, the same
   rows used to train the forecaster. This results in in-sample residual
   calibration rather than split conformal calibration, which can collapse
   corrections for an overfit model.

To satisfy split-conformal calibration assumptions, reserve a separate
calibration period, predict it with the already-fitted forecaster, and fit the
calibrator on those held-out predictions:

.. code-block:: python

   from openstef_core.datasets import ForecastDataset
   from openstef_core.types import Quantile
   from openstef_models.transforms.postprocessing import MapieQuantileCalibrator

   # 1. Fit the forecaster on the training split (without the calibrator).
   workflow.fit(input_data_train)

   # 2. Create a held-out calibration split that the forecaster has never seen.
   calibrator = MapieQuantileCalibrator(
       quantiles=[
           Quantile(0.1),
           Quantile(0.3),
           Quantile(0.5),
           Quantile(0.7),
           Quantile(0.9),
       ],
   )

   # 3. Generate forecasts on the held-out calibration period.
   cal_forecasts = workflow.predict(input_data_cal)

   # 4. Attach the observed actuals to the forecast dataset so that the
   #    calibrator can compute residuals.  ``target_column`` must be the name
   #    of the column that holds the observed values (default: "load").
   cal_forecasts.data[cal_forecasts.target_column] = input_data_cal[target_col]
   cal_dataset = ForecastDataset(
       data=cal_forecasts.data,
       sample_interval=cal_forecasts.sample_interval,
       target_column=cal_forecasts.target_column,
   )

   # 5. Fit the calibrator on the held-out ForecastDataset (predictions + actuals).
   calibrator.fit(cal_dataset)

   # 6. Append the fitted calibrator to the postprocessing pipeline so that
   #    subsequent calls to workflow.predict() apply the correction.
   workflow.model.postprocessing.transforms.append(calibrator)

MAPIE calibration corrects marginal quantile coverage; it does not guarantee
conditional calibration for every time of day, season, or weather regime. The
postprocessing pipeline can still apply :class:`~openstef_models.transforms.postprocessing.quantile_sorter.QuantileSorter`
after calibration to enforce row-wise quantile ordering.

For a complete worked example showing calibration before and after, including diagnostic plots, see :doc:`/tutorials/quantile_calibration`.

Evaluating Probabilistic Forecasts
-----------------------------------

Calibration quality can be assessed by comparing expected vs. observed quantile levels. For each quantile Q, compute the fraction of actual observations that fall below the predicted quantile value. A well-calibrated model produces a diagonal on a calibration plot (expected = observed).

Key metrics for probabilistic forecast quality include:

- **Calibration error**: the difference between expected and observed coverage per quantile
- **Sharpness**: the width of prediction intervals (narrower is better, given proper calibration)
- **Pinball loss**: the proper scoring rule for quantile forecasts, penalizing both miscalibration and lack of sharpness

See :doc:`/user_guide/guides/backtesting_tutorial` for how to evaluate forecast quality on historical data.

.. seealso::

   - :doc:`/user_guide/guides/forecasting` for the overall forecasting workflow (fitting, predicting, model selection).
   - :doc:`/user_guide/concepts/models` for understanding how different model types compare.
   - :doc:`/user_guide/guides/backtesting_tutorial` for evaluating forecast performance systematically.
   - :doc:`/user_guide/guides/reliability_fallback` for operational concerns like fallback behavior when data is missing.