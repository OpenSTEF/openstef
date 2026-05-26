.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guide_reliability_fallback:

Reliability and Fallback
========================

Energy forecasting operates on live data feeds that inevitably degrade. Smart meters stop reporting, weather APIs experience outages, and network issues cause data to arrive late or not at all. A production forecasting system must detect these conditions and respond gracefully, rather than silently producing unreliable predictions.

OpenSTEF addresses this through two complementary mechanisms: **validation transforms** that detect data quality problems, and **fallback forecasters** that produce safe predictions when the primary model cannot be trusted.

The Problem: Silent Failures
----------------------------

Consider a substation meter that freezes at its last reported value. The data pipeline continues to receive records (the value simply does not change), so no obvious error occurs. If a forecasting model trains on this stale data, it learns a false pattern. Worse, if it predicts using stale inputs, the forecast may look plausible while being completely wrong.

Similarly, when 60% of input features are missing due to a weather API outage, a tree-based model will still produce a number. That number may be meaningless, but nothing in the model itself will flag the problem.

OpenSTEF's approach is to make these failures explicit through validation checks that run before forecasting, and to provide degraded-but-honest predictions through fallback models.

Detecting Data Quality Issues
-----------------------------

Flatline Detection
^^^^^^^^^^^^^^^^^^

The :class:`~openstef_models.transforms.validation.FlatlineChecker` detects when a signal stops changing, which typically indicates a meter or sensor failure rather than genuinely constant consumption.

Configuration parameters (set in ``ForecastingWorkflowConfig``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Purpose
   * - ``flatliner_threshold``
     - 24 hours
     - Duration the load must remain constant before flagging a flatline
   * - ``detect_non_zero_flatliner``
     - ``False``
     - When ``True``, detects flatlines at any constant value (not just zero)
   * - ``absolute_tolerance``
     - 0.0
     - Absolute tolerance for considering values as equal
   * - ``relative_tolerance``
     - 1e-5
     - Relative tolerance for considering values as equal

When a flatline is detected, the checker raises a ``FlatlinerDetectedError``. The workflow catches this and switches to a fallback forecaster automatically.

.. note::

   Setting ``detect_non_zero_flatliner=True`` is important for loads that genuinely idle at zero (e.g., solar generation at night). Without this flag, only zero-value flatlines are detected, which would miss a meter stuck at its last non-zero reading.

Completeness Checking
^^^^^^^^^^^^^^^^^^^^^

The :class:`~openstef_models.transforms.validation.CompletenessChecker` enforces minimum data availability by computing the ratio of non-missing values to total expected values.

.. code-block:: python

   from openstef_models.transforms.validation import CompletenessChecker

   checker = CompletenessChecker(
       columns=["load", "temperature_2m"],
       completeness_threshold=0.8,
   )

When completeness falls below the threshold, an ``InsufficientlyCompleteError`` is raised. The checker supports optional column weights to prioritize certain features (e.g., the load column may be more critical than a secondary weather variable).

The ``completeness_threshold`` in ``ForecastingWorkflowConfig`` controls the minimum fraction of data required for a regular forecast (default: 0.5).

Fallback Forecasters
--------------------

When validation checks fail, OpenSTEF does not simply skip the forecast. Instead, it switches to a fallback model that produces honest, if limited, predictions.

FlatlinerForecaster
^^^^^^^^^^^^^^^^^^^

The :class:`~openstef_models.models.forecasting.flatliner_forecaster.FlatlinerForecaster` activates when flatline detection triggers. It produces a constant prediction for all horizons and quantiles:

- **Default behavior**: predicts zero for all time steps
- **With** ``predict_nonzero_flatliner=True``: predicts the median of historical load measurements

This is appropriate because if the meter is genuinely flatlining (e.g., a decommissioned connection), zero is the correct forecast. If the meter is stuck at a non-zero value, the historical median provides a safer estimate than the frozen reading.

ConstantQuantileForecaster
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~openstef_models.models.forecasting.constant_quantile_forecaster.ConstantQuantileForecaster` serves as a minimal fallback when only a tiny fraction of data is available. Rather than attempting a time-series forecast with insufficient information, it returns constant quantile values derived from whatever training data was available.

This forecaster is configured through ``completeness_threshold_target_constant_quantile`` in the workflow configuration, which defines the completeness level below which this ultra-conservative fallback is used instead of the regular model.

How the Workflow Selects Forecasters
------------------------------------

The preset factory and workflow configuration handle fallback selection automatically. You do not need to implement switching logic yourself. The decision flow follows this sequence:

1. **Flatline check**: If the load signal is flatlining, use ``FlatlinerForecaster``
2. **Completeness check (severe)**: If data completeness is below ``completeness_threshold_target_constant_quantile``, use ``ConstantQuantileForecaster``
3. **Completeness check (moderate)**: If data completeness is below ``completeness_threshold``, raise an error (no forecast produced)
4. **Normal operation**: Use the configured primary forecaster

This tiered approach ensures that some prediction is always available for operational systems that cannot tolerate gaps, while clearly signaling degraded confidence.

Configuration in Practice
-------------------------

All reliability parameters live in ``ForecastingWorkflowConfig``:

.. code-block:: python

   from openstef_core.config import ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(
       flatliner_threshold=timedelta(hours=24),
       detect_non_zero_flatliner=True,
       predict_nonzero_flatliner=False,
       completeness_threshold=0.5,
   )

.. warning::

   Setting ``flatliner_threshold`` too low (e.g., 1 hour) may cause false positives for loads with naturally flat periods, such as industrial processes with steady-state operation or solar panels during overcast conditions. Tune this threshold based on the variability profile of your specific prediction job.

Interpreting Fallback Predictions
---------------------------------

Downstream systems consuming OpenSTEF forecasts should be aware that fallback predictions carry different semantics:

- **FlatlinerForecaster output**: All quantiles collapse to the same constant value. The forecast conveys "we believe this connection is inactive or the meter is broken."
- **ConstantQuantileForecaster output**: Quantiles reflect historical distribution but carry no temporal structure. The forecast conveys "we lack sufficient data for a time-aware prediction."

Both cases produce valid forecast objects with proper quantile structure, so consuming systems do not need special handling. However, monitoring systems should track how often fallbacks activate, as frequent activation indicates upstream data quality problems that need resolution.

For details on how quantile forecasts are structured in normal operation, see :doc:`/user_guide/guides/probabilistic_forecasting`. For the broader forecasting workflow that these fallbacks plug into, see :doc:`/user_guide/guides/forecasting`.