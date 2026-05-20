Reliability and Fallback
========================

In production energy forecasting, the question is not *whether* data feeds will fail, but *when*. Smart meters stop reporting. Weather APIs go down. Telemetry arrives late or corrupted. A forecasting system that silently produces garbage during these events is worse than one that raises an alarm — grid operators making decisions on bad forecasts face real consequences.

This page explains how OpenSTEF detects degraded data conditions and responds gracefully, ensuring that forecasts are either trustworthy or explicitly flagged as unreliable.

.. mermaid:: /diagrams/user_guide/guides/reliability_fallback_diagram_1.mmd

The Problem: Silent Failures
----------------------------

Energy data pipelines face several failure modes that are difficult to detect without explicit checks:

.. list-table:: Common Data Failure Modes
   :header-rows: 1
   :widths: 20 40 40

   * - Failure Mode
     - Symptom
     - Risk if Undetected
   * - Meter flatlining
     - Load value stays constant (often zero) for hours
     - Model trains on or predicts from fake "stable" load
   * - Stale data
     - Most recent measurement is hours or days old
     - Forecast anchored to outdated state
   * - Missing columns
     - Weather source drops a variable between train and predict
     - Model receives incomplete features, produces unreliable output
   * - Incomplete data
     - Large gaps in historical data used for training
     - Model learns from sparse, unrepresentative patterns

OpenSTEF addresses these through **validation transforms** — pipeline components that check data quality and either raise exceptions or log warnings before forecasting proceeds.

Flatliner Detection
-------------------

A "flatliner" is a time series where the measured value stops changing — typically indicating a sensor malfunction, communication failure, or data pipeline freeze rather than genuinely constant energy consumption.

Why this matters
^^^^^^^^^^^^^^^^

A meter reporting exactly 0.0 kW for 24 hours is almost certainly broken, not measuring a building that happens to use zero energy. If this data enters training, the model learns incorrect patterns. If it enters prediction, the forecast is anchored to a false baseline.

OpenSTEF's approach
^^^^^^^^^^^^^^^^^^^

The :class:`~openstef_models.transforms.validation.FlatlineChecker` transform detects ongoing flatliners by examining whether recent measurements remain constant (within tolerance) over a configurable duration:

.. code-block:: python

   from openstef_models.transforms.validation import FlatlineChecker

   checker = FlatlineChecker(
       flatliner_threshold=timedelta(hours=24),
       detect_non_zero_flatliner=True,
       error_on_flatliner=True,
   )

Key configuration options:

- **flatliner_threshold** — How long the signal must be constant before triggering detection (default: 24 hours). Shorter thresholds catch failures faster but risk false positives on legitimately stable loads.
- **detect_non_zero_flatliner** — When ``True``, detects flatliners at any constant value (e.g., a meter stuck at 42.5 kW), not just zero. Uses the median of recent measurements as the reference.
- **absolute_tolerance** / **relative_tolerance** — Numeric tolerances for "constant." Small fluctuations from sensor noise won't prevent detection.
- **error_on_flatliner** — When ``True``, raises :class:`~openstef_core.exceptions.FlatlinerDetectedError`. When ``False``, logs a warning and allows the pipeline to continue.

The detection algorithm compares recent measurements against a reference value using ``numpy.isclose``, ensuring that minor floating-point variations don't mask a genuine flatliner.

.. note:: [VISUALIZATION: Time series plot showing normal load pattern transitioning to a flat line at hour T, with the flatliner_threshold window highlighted and the detection point marked]

Data Completeness Validation
----------------------------

Even when data is arriving, it may contain too many gaps to produce a reliable forecast. The :class:`~openstef_models.transforms.validation.CompletenessChecker` validates that required columns meet a minimum completeness threshold:

.. code-block:: python

   from openstef_models.transforms.validation import CompletenessChecker

   checker = CompletenessChecker(
       columns=["load", "temperature_2m"],
       completeness_threshold=0.8,
   )

If more than 20% of values are missing in any specified column, the transform raises :class:`~openstef_core.exceptions.InsufficientlyCompleteError`. This prevents training on sparse data that would produce unreliable models.

Schema Drift Detection
----------------------

When a weather data source stops providing a column between training and prediction time, the model receives incomplete features. The :class:`~openstef_models.transforms.validation.InputConsistencyChecker` learns the expected column schema during ``fit()`` and raises :class:`~openstef_core.exceptions.MissingColumnsError` if columns disappear at prediction time.

This catches a subtle but dangerous failure: the model doesn't crash — it simply produces forecasts without important features, degrading quality silently.

Fallback Forecasting
--------------------

When validation detects degraded conditions, the system needs a response beyond simply refusing to forecast. Grid operators need *something* — even a rough estimate is better than nothing, as long as it's clearly marked as degraded.

The BaseCaseForecaster
^^^^^^^^^^^^^^^^^^^^^^

OpenSTEF provides :class:`~openstef_models.models.forecasting.BaseCaseForecaster` as a fallback strategy. This model requires no weather data or complex features — it simply repeats historical weekly load patterns:

- **Primary strategy**: Use load data from 7 days ago (weekly periodicity assumption)
- **Fallback strategy**: If primary lag data is unavailable, fall back to 14 days ago
- **Uncertainty**: Calculates confidence intervals from hourly standard deviations of the repeated pattern

.. code-block:: python

   from openstef_models.models.forecasting import BaseCaseForecaster

   fallback = BaseCaseForecaster(
       quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
       horizons=[LeadTime(timedelta(hours=1))],
   )

This is intentionally simple. It won't capture weather-driven variations, but it provides a reasonable baseline that degrades gracefully — a property more valuable than accuracy when primary data feeds are down.

Building a Resilient Pipeline
-----------------------------

The recommended pattern combines validation transforms with fallback logic:

.. mermaid:: /diagrams/user_guide/guides/reliability_fallback_diagram_2.mmd

The key principles are:

1. **Fail fast on training** — Use ``error_on_flatliner=True`` and strict completeness thresholds during model training. Bad training data corrupts the model permanently.

2. **Degrade gracefully on prediction** — During real-time forecasting, catch validation exceptions and switch to fallback strategies rather than producing no forecast at all.

3. **Signal degradation clearly** — Any fallback forecast should be flagged so downstream consumers (operators, trading systems) know to treat it with appropriate skepticism.

4. **Layer your checks** — Place validation transforms early in the pipeline to reject bad data before expensive computation occurs.

Exception Hierarchy
^^^^^^^^^^^^^^^^^^^

OpenSTEF defines specific exceptions for each failure mode, enabling targeted error handling:

.. list-table:: Validation Exceptions
   :header-rows: 1
   :widths: 35 65

   * - Exception
     - Meaning
   * - :class:`~openstef_core.exceptions.FlatlinerDetectedError`
     - Load signal is constant — likely meter failure
   * - :class:`~openstef_core.exceptions.InsufficientlyCompleteError`
     - Too many missing values for reliable forecasting
   * - :class:`~openstef_core.exceptions.MissingColumnsError`
     - Expected input columns are absent (schema drift)
   * - :class:`~openstef_core.exceptions.PredictError`
     - Forecasting operation failed at runtime

Tuning for Your Environment
----------------------------

The right validation thresholds depend on your operational context:

- **Industrial loads** with genuinely stable consumption may need longer ``flatliner_threshold`` values or higher tolerances to avoid false positives.
- **Solar generation** legitimately flatlines at zero overnight — use time-of-day awareness or disable zero-flatliner detection for generation meters.
- **Completeness thresholds** should be stricter for training (0.9+) than for prediction (0.7–0.8), since a single bad training run has lasting effects.

.. warning::

   Setting ``error_on_flatliner=False`` in production means flatliners will only generate log warnings. Ensure your monitoring system captures these warnings — otherwise failures become truly silent.

Related Topics
--------------

- :doc:`forecasting` — The primary forecasting workflow that these reliability checks protect
- :doc:`datasets` — How OpenSTEF's dataset types support validation and quality metadata
- :doc:`probabilistic_forecasting` — Uncertainty quantification, which becomes especially important during degraded conditions
- :doc:`deployment` — Operational patterns for integrating fallback logic into production systems