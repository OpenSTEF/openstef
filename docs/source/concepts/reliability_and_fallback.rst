Reliability and Fallback Strategies
===================================

Production energy forecasting systems must deliver predictions continuously, even when
models fail, data is missing, or inputs are degraded. This page covers the strategies
OpenSTEF provides for graceful degradation: fallback forecasters, data validation,
model staleness detection, and handling incomplete inputs.

.. mermaid:: /diagrams/concepts/reliability_and_fallback_diagram_1.mmd

Why Reliability Matters
-----------------------

In grid operations, a missing forecast is often worse than an imperfect one. Decisions
about energy dispatch, congestion management, and market bidding happen on fixed
schedules. OpenSTEF addresses this with multiple layers of defense:

- **Data validation** catches bad inputs before they reach the model
- **Model staleness detection** prevents outdated models from silently degrading
- **Fallback forecasters** provide reasonable predictions when the primary model cannot
- **Lag fallback** fills missing feature values from alternative time offsets

Data Validation
---------------

OpenSTEF provides validation transforms that run before prediction, catching problems
early and raising specific exceptions that downstream systems can handle.

Completeness Checking
^^^^^^^^^^^^^^^^^^^^^

The ``CompletenessChecker`` ensures input data has sufficient non-null values before
forecasting proceeds:

.. code-block:: python

   from openstef_models.transforms.validation.completeness_checker import CompletenessChecker

   checker = CompletenessChecker(
       completeness_threshold=0.5,  # At least 50% of data must be present
       columns=["load", "temperature", "wind_speed"],
       weights={"load": 2.0, "temperature": 1.0, "wind_speed": 1.0},
   )

   # Raises InsufficientlyCompleteError if data is too sparse
   validated_data = checker.transform(input_data)

The ``completeness_threshold`` parameter (default 0.5) controls how strict the check is.
For critical forecasting points, increase this value. For points with historically
unreliable telemetry, you may lower it to avoid unnecessary fallback activations.

Flatline Detection
^^^^^^^^^^^^^^^^^^

Sensors sometimes fail by reporting a constant value. The ``FlatlineChecker`` detects
this pattern and raises ``FlatlinerDetectedError``:

.. code-block:: python

   from datetime import timedelta
   from openstef_models.transforms.validation.flatline_checker import FlatlineChecker

   flatline_checker = FlatlineChecker(
       flatliner_threshold=timedelta(hours=24),
       detect_non_zero_flatliner=False,
   )

   # Detects if the latest measurements are suspiciously constant
   validated_data = flatline_checker.transform(input_data)

When ``detect_non_zero_flatliner`` is ``True``, the checker also flags constant non-zero
values (e.g., a sensor stuck at 42 MW), not just zero flatlines.

Input Consistency
^^^^^^^^^^^^^^^^^

The ``InputConsistencyChecker`` validates that prediction-time inputs match the schema
the model was trained on:

.. code-block:: python

   from openstef_models.transforms.validation.input_consistency_checker import InputConsistencyChecker

   consistency_checker = InputConsistencyChecker()
   consistency_checker.fit(training_data)  # Learn expected schema

   # At prediction time, validates columns and types match
   validated_data = consistency_checker.transform(new_data)

Model Staleness Detection
-------------------------

Models degrade over time as patterns in energy consumption shift. OpenSTEF's workflow
system tracks model age and enforces retraining through the ``model_reuse_max_age``
parameter:

.. code-block:: python

   from datetime import timedelta

   workflow_config = {
       "model_reuse_enable": True,
       "model_reuse_max_age": timedelta(days=7),
       "model_selection_enable": True,
       "model_selection_metric": ("Q50", "R2", "higher_is_better"),
       "model_selection_old_model_penalty": 1.2,
   }

Key parameters:

- ``model_reuse_max_age``: Maximum age before a model is considered stale and retraining
  is triggered. Default is 7 days.
- ``model_selection_old_model_penalty``: A multiplier (default 1.2) applied to the old
  model's metric score during comparison, biasing selection toward freshly trained models.
  This prevents a slightly-better old model from blocking adoption of a newer one that
  will age better.

When a model exceeds its maximum age and retraining has not yet succeeded, the system
raises ``ModelLoadingError`` or falls back to the base case forecaster.

Fallback Forecasters
--------------------

When the primary ML model cannot produce a forecast—due to missing features, model
loading failure, or validation errors—OpenSTEF provides the ``BaseCaseForecaster`` as a
reliable fallback.

The BaseCaseForecaster
^^^^^^^^^^^^^^^^^^^^^^

This forecaster uses a simple but effective strategy: repeat the most recent weekly
pattern from historical data.

.. code-block:: python

   from datetime import timedelta
   from openstef_models.models.forecasting.base_case_forecaster import (
       BaseCaseForecaster,
       BaseCaseForecasterHyperParams,
   )

   params = BaseCaseForecasterHyperParams(
       primary_lag=timedelta(days=7),    # Use last week's pattern
       fallback_lag=timedelta(days=14),  # If last week unavailable, use two weeks ago
   )

   fallback_model = BaseCaseForecaster(hyperparams=params)

The two-tier lag design means even if the most recent week of history is incomplete
(e.g., due to a data pipeline outage), the forecaster can still produce predictions
from older data.

.. mermaid:: /diagrams/concepts/reliability_and_fallback_diagram_2.mmd

Lag Feature Fallback
^^^^^^^^^^^^^^^^^^^^

For the primary ML model, lag-based features (e.g., "load 7 days ago") are critical
inputs. When these values are missing, OpenSTEF's lag transform applies a fallback
shift rather than leaving NaN values that would break prediction:

.. code-block:: python

   # Internal behavior of the lag transform:
   # For each lag L with missing values, fill from L + lag_fallback_offset
   # Example: if 7-day lag is missing, fill from 7 + 7 = 14-day lag

   # Window guard prevents using lags beyond available history:
   # if fallback_lag > history_available: skip

This happens automatically within the feature engineering pipeline. The
``lag_fallback_offset`` parameter controls how far back the fallback reaches.

Exception Hierarchy for Error Handling
--------------------------------------

OpenSTEF defines specific exceptions that enable fine-grained error handling in
production systems:

.. code-block:: python

   from openstef_core.mixins import (
       FlatlinerDetectedError,
       ModelUnderperformingError,
       InsufficientlyCompleteError,
       PredictError,
       ModelLoadingError,
       ModelNotFoundError,
   )

   def produce_forecast(prediction_job):
       try:
           return run_primary_forecast(prediction_job)
       except ModelLoadingError:
           # Model file corrupted or missing — use fallback
           return run_fallback_forecast(prediction_job)
       except InsufficientlyCompleteError:
           # Not enough input data — use fallback with degraded confidence
           return run_fallback_forecast(prediction_job)
       except FlatlinerDetectedError:
           # Sensor stuck — return zero or median depending on config
           return handle_flatliner(prediction_job)
       except ModelUnderperformingError:
           # Model metrics below threshold — trigger retraining, use last good forecast
           trigger_retraining(prediction_job)
           return get_last_valid_forecast(prediction_job)
       except PredictError:
           # Unexpected prediction failure — log and use fallback
           return run_fallback_forecast(prediction_job)

.. warning::

   Never silently swallow exceptions. Always log which fallback path was taken and why.
   Silent failures accumulate and mask systemic issues like data pipeline degradation.

Designing for Graceful Degradation
----------------------------------

A well-configured production system layers these mechanisms:

1. **Validate inputs** — Catch data problems before they reach the model
2. **Try primary model** — Use the trained ML model for best accuracy
3. **Fall back gracefully** — Use ``BaseCaseForecaster`` when the primary fails
4. **Signal degradation** — Mark forecasts produced by fallback so operators know
5. **Trigger recovery** — Automatically schedule retraining or alert on persistent failures

Configuration Example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from datetime import timedelta

   forecast_config = {
       # Validation thresholds
       "completeness_threshold": 0.5,
       "flatliner_threshold": timedelta(hours=24),
       "detect_non_zero_flatliner": False,

       # Model freshness
       "model_reuse_max_age": timedelta(days=7),
       "model_selection_old_model_penalty": 1.2,

       # Fallback behavior
       "predict_nonzero_flatliner": False,  # Predict zero when flatliner detected
   }

Monitoring Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Track these metrics to detect reliability issues before they become outages:

- **Fallback activation rate** — Percentage of forecasts using the fallback model.
  A sudden increase indicates upstream problems.
- **Data completeness trend** — Rolling average completeness score per prediction point.
  Gradual decline suggests sensor degradation.
- **Model age distribution** — How old are your active models? Clustering near
  ``model_reuse_max_age`` suggests retraining is failing.
- **Forecast confidence width** — Widening quantile intervals (see
  :doc:`quantiles_and_confidence`) may indicate model uncertainty is growing.

Related Topics
--------------

- :doc:`forecasting_basics` — Core concepts of short-term energy forecasting
- :doc:`quantiles_and_confidence` — How probabilistic forecasts communicate uncertainty
- :doc:`feature_engineering` — How input features are constructed, including lag features