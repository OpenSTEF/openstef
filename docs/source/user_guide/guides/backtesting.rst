Backtesting
===========

.. _guide_backtesting:

Before deploying a forecasting model to production, you need confidence that it
performs well across diverse conditions—peak demand, weather transitions, holidays,
and data gaps. Backtesting provides this confidence by replaying history as if it
were happening in real time, ensuring your model never sees future data during
evaluation.

This guide explains how to set up and interpret a backtest using OpenSTEF's BEAM
components. For the underlying architecture, see :ref:`concept_beam`. For a
runnable notebook, see :doc:`/tutorials/backtesting_quickstart`.

Why Backtesting Matters for Energy Forecasting
----------------------------------------------

A single train/test split is insufficient for energy forecasting because:

- **Seasonality**: a model trained on summer may fail in winter.
- **Regime changes**: grid topology changes, new solar installations, or demand shifts invalidate older patterns.
- **Operational realism**: in production, your model retrains periodically and predicts at fixed intervals. A backtest must replicate this cadence.

OpenSTEF's backtesting framework addresses these concerns by simulating the exact
operational loop—periodic retraining, scheduled predictions, and strict data
availability constraints—across an arbitrary historical window.


The Three Stages of a Backtest
------------------------------

A complete backtest in OpenSTEF follows three stages:

1. **Configure** — wrap your forecaster with :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin` and define timing via :class:`~openstef_beam.backtesting.backtest_pipeline.BacktestConfig`.
2. **Run** — execute :class:`~openstef_beam.backtesting.backtest_pipeline.BacktestPipeline` which steps through history generating train and predict events.
3. **Evaluate** — feed collected predictions into :class:`~openstef_beam.evaluation.EvaluationPipeline` for fair per-horizon assessment.

.. mermaid:: /diagrams/user_guide/guides/backtesting_diagram_1.mmd

Stage 1: Configure the Forecaster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your forecasting model must implement the :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin` interface. This mixin defines two key methods:

- ``fit(data)`` — train on a :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries`
- ``predict(data)`` — generate forecasts from the same restricted view

The :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterConfig` controls operational parameters:

.. code-block:: python

   from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig

   forecaster_config = BacktestForecasterConfig(
       requires_training=True,
       predict_length=timedelta(days=2),
       predict_sample_interval=timedelta(minutes=15),
       training_context_length=timedelta(days=365),
   )

The ``predict_length`` determines how far ahead each prediction event forecasts,
while ``training_context_length`` controls how much history the model sees during
training.


Stage 2: Run the BacktestPipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline orchestrates the simulation:

.. code-block:: python

   from openstef_beam.backtesting.backtest_pipeline import BacktestPipeline, BacktestConfig

   config = BacktestConfig(
       prediction_sample_interval=timedelta(minutes=15),
       prediction_interval=timedelta(hours=1),
       training_interval=timedelta(days=7),
   )

   pipeline = BacktestPipeline(config=config, forecaster=my_forecaster)
   predictions = pipeline.run(
       ground_truth=ground_truth_dataset,
       predictors=predictor_dataset,
       start=datetime(2023, 1, 1),
       end=datetime(2023, 12, 31),
   )

**How it works internally:**

1. A ``BacktestEventGenerator`` creates a chronological schedule of train and predict events based on ``training_interval`` and ``prediction_interval``.
2. At each **train event**, the pipeline wraps the full dataset in a :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries` with the event's timestamp as the horizon. This ensures ``fit()`` cannot access any data published after that moment.
3. At each **predict event**, the same restriction applies—``predict()`` receives only data that would have been available operationally.
4. All predictions are collected into a single versioned :class:`~openstef_beam.datasets.TimeSeriesDataset` with ``available_at`` metadata.

Preventing Data Leakage
""""""""""""""""""""""""

The :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries` is the critical guardrail. It wraps a :class:`~openstef_beam.datasets.VersionedTimeSeriesDataset` and exposes a ``get_window()`` method that enforces ``available_before`` filtering. Even if your forecaster accidentally requests future timestamps, the wrapper returns only data published before the current simulation time.

.. warning::

   If your ``prediction_sample_interval`` in the backtest config does not match
   your forecaster's ``predict_sample_interval``, the pipeline raises a
   ``ValueError`` at initialization. Always ensure these are consistent.


Stage 3: Evaluate with EvaluationPipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw predictions are not directly comparable—a 15-minute-ahead forecast is
inherently easier than a 48-hour-ahead forecast. The
:class:`~openstef_beam.evaluation.EvaluationPipeline` slices predictions along
multiple dimensions for fair comparison:

.. list-table:: Evaluation Dimensions
   :header-rows: 1
   :widths: 20 40 40

   * - Dimension
     - What it controls
     - Example
   * - ``available_at``
     - When the forecast was issued
     - ``D-1T06:00`` (day-ahead at 6 AM)
   * - ``lead_time``
     - Gap between issuance and target
     - ``PT36H`` (36 hours ahead)
   * - ``windows``
     - Rolling evaluation periods
     - 21-day rolling windows

.. code-block:: python

   from openstef_beam.evaluation import EvaluationPipeline, EvaluationConfig

   evaluation_config = EvaluationConfig(
       available_ats=[AvailableAt.from_string("D-1T06:00")],
       lead_times=[LeadTime.from_string("PT36H")],
       windows=[Window(lag=timedelta(hours=0), size=timedelta(days=21))],
   )

   eval_pipeline = EvaluationPipeline(
       config=evaluation_config,
       quantiles=my_forecaster.quantiles,
       window_metric_providers=[RMAEProvider(quantiles=[Q(0.5)]), RCRPSProvider()],
       global_metric_providers=[RMAEProvider(quantiles=[Q(0.5)]), RCRPSProvider()],
   )

   report = eval_pipeline.run(
       predictions=predictions,
       ground_truth=ground_truth,
       target_column="load",
   )

The pipeline automatically includes observed probability as a calibration metric,
ensuring you always assess whether your quantile forecasts are well-calibrated.


Interpreting Backtest Results
-----------------------------

Per-Window vs. Aggregated Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The evaluation report contains both:

- **Windowed metrics** — computed over rolling time windows (e.g., 21-day blocks). These reveal performance drift: degradation after regime changes, seasonal patterns in error, or recovery after retraining.
- **Global metrics** — aggregated across the entire backtest period. These give an overall performance summary but can mask temporal variation.

Always inspect windowed metrics first. A model with acceptable global RMAE but
large spikes in specific windows may be unreliable for operations.

What "Good" Looks Like
^^^^^^^^^^^^^^^^^^^^^^

Typical error ranges depend heavily on the aggregation level and forecast horizon:

.. list-table:: Indicative Performance Ranges
   :header-rows: 1
   :widths: 30 25 25 20

   * - Aggregation Level
     - Horizon
     - Typical RMAE
     - Notes
   * - National/regional (GW-scale)
     - Day-ahead
     - 2–5%
     - Smoothing from aggregation
   * - Substation (MW-scale)
     - Day-ahead
     - 5–15%
     - Weather-sensitive
   * - Individual connection (kW-scale)
     - Day-ahead
     - 20–50%+
     - High stochasticity
   * - Any level
     - Intraday (<4h)
     - 30–70% of day-ahead error
     - Recent data helps significantly

.. note::

   These ranges are indicative. Actual performance depends on load type (base load
   vs. weather-driven), data quality, and feature availability. Use your own
   historical baselines rather than absolute thresholds.

Key Metrics to Monitor
^^^^^^^^^^^^^^^^^^^^^^

- **RMAE** (Relative Mean Absolute Error) at the median quantile — primary accuracy measure.
- **RCRPS** (Relative Continuous Ranked Probability Score) — assesses the full probabilistic forecast quality.
- **Observed Probability** — calibration check: does the 90th percentile quantile actually contain 90% of observations?

See :doc:`probabilistic_forecasting` for details on interpreting probabilistic metrics.


Common Pitfalls
---------------

- **Too-short backtest window**: include at least one full seasonal cycle (ideally 12+ months) to capture all operational conditions.
- **Ignoring retraining cadence**: set ``training_interval`` to match your production schedule. A model retrained daily in backtesting but weekly in production gives optimistic results.
- **Evaluating only global metrics**: always check per-window results to detect performance instability.
- **Mismatched sample intervals**: ensure your data, forecaster config, and backtest config all use the same ``prediction_sample_interval``.


Next Steps
----------

- :doc:`/tutorials/backtesting_quickstart` — full worked example with real data
- :ref:`concept_beam` — architecture details on how pipelines compose
- :doc:`forecasting` — how to configure the forecasting model itself
- :doc:`datasets` — understanding versioned time series and data access patterns
- :doc:`probabilistic_forecasting` — interpreting quantile forecasts and calibration