.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guide_backtesting:

Backtesting
===========

Backtesting answers a critical question before deploying any forecasting model: *how would this model have performed if it had been running in production over the past weeks or months?* A naive approach (training on all history, then predicting the same period) produces optimistic results because the model has already "seen" the future. OpenSTEF's backtesting framework prevents this by simulating real-time constraints: at every point in the replay, the model can only access data that would have been available at that moment.

This page explains how to configure and run a backtest, then evaluate the results fairly across different forecast horizons.

.. mermaid:: /diagrams/user_guide/guides/backtesting_diagram_1.mmd

Why Backtesting Requires Special Infrastructure
------------------------------------------------

Energy forecasting models operate under strict temporal constraints. In production, a model predicting tomorrow's load at 06:00 today can only use data published before 06:00. Weather forecasts, meter readings, and market prices all arrive at different times with different latencies. A backtest must replicate these constraints faithfully, or the results will overestimate real-world performance.

OpenSTEF addresses this with three coordinated components:

- **Lookahead prevention** via :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries`, which wraps your data and blocks access to anything published after the simulated "current time."
- **Event scheduling** via :class:`~openstef_beam.backtesting.BacktestPipeline`, which steps through history generating train and predict events at configurable intervals.
- **Fair evaluation** via :class:`~openstef_beam.evaluation.EvaluationPipeline`, which segments predictions by when they were made (``available_at``) and how far ahead they look (``lead_time``).

For the architectural context of how these components fit into the broader BEAM framework, see :ref:`concept_beam`.


Configuring the Backtest
-------------------------

The backtest is configured through :class:`~openstef_beam.backtesting.BacktestConfig`, which controls how the pipeline steps through time:

.. code-block:: python

   from datetime import timedelta
   from openstef_beam.backtesting import BacktestConfig

   config = BacktestConfig(
       predict_interval=timedelta(hours=1),
       train_interval=timedelta(days=7),
       prediction_sample_interval=timedelta(minutes=15),
   )

The key fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Purpose
   * - ``predict_interval``
     - How often the pipeline generates a new forecast (e.g., every hour). Smaller intervals produce more predictions but take longer to run.
   * - ``train_interval``
     - How often the model is retrained on accumulated data (e.g., every 7 days). This simulates periodic retraining in production.
   * - ``prediction_sample_interval``
     - The resolution of the output forecast (e.g., 15-minute intervals). Must match your forecaster's configuration.


Implementing a Forecaster
--------------------------

Your model must implement :class:`~openstef_beam.backtesting.BacktestForecasterMixin`, which defines two methods:

- ``fit(data)`` — train the model on a :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries` (only data up to the current simulated time is accessible).
- ``predict(data)`` — generate quantile forecasts from the same restricted view.

The ``RestrictedHorizonVersionedTimeSeries`` wrapper is the key to preventing lookahead. When your forecaster calls ``data.get_window(start, end)``, the wrapper silently enforces ``available_before=current_horizon``, ensuring no future data leaks into training or prediction.

The forecaster also requires a :class:`~openstef_beam.backtesting.backtest_forecaster.BacktestForecasterConfig` specifying operational parameters like ``predict_length`` (how far ahead to forecast), ``predict_context_length`` (how much history the model needs), and ``requires_training`` (whether the model needs periodic retraining).


Running the Pipeline
--------------------

Once configured, running the backtest is straightforward:

.. code-block:: python

   pipeline = BacktestPipeline(config=config, forecaster=my_forecaster)
   predictions = pipeline.run(
       ground_truth=ground_truth_dataset,
       predictors=predictor_dataset,
       start=start_datetime,
       end=end_datetime,
   )

The pipeline internally:

1. Creates a schedule of train and predict events using ``BacktestEventGenerator`` based on your configured intervals.
2. At each train event, wraps the dataset with a restricted horizon and calls ``forecaster.fit()``.
3. At each predict event, wraps the dataset again (horizon set to the event timestamp) and calls ``forecaster.predict()``.
4. Collects all predictions into a single ``TimeSeriesDataset`` with ``available_at`` metadata preserved.

The result is a time series of forecasts, each tagged with when it was generated, ready for evaluation.


Evaluating Results
------------------

Raw predictions are not directly comparable: a 1-hour-ahead forecast should be more accurate than a 36-hour-ahead forecast. The :class:`~openstef_beam.evaluation.EvaluationPipeline` segments predictions along multiple dimensions to produce fair comparisons.

.. code-block:: python

   from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline

   evaluation_config = EvaluationConfig(
       available_ats=[AvailableAt.from_string("D-1T06:00")],
       lead_times=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT36H")],
       windows=[Window(lag=timedelta(hours=0), size=timedelta(days=21))],
   )

The three evaluation dimensions are:

- **available_at** — filters predictions to those generated at a specific time of day (e.g., the day-ahead forecast made at 06:00).
- **lead_time** — groups predictions by how far ahead they look (e.g., 1 hour vs. 36 hours).
- **windows** — defines rolling time windows for computing metrics, allowing you to see how performance evolves over the backtest period.

The pipeline produces an ``EvaluationReport`` containing subset reports for each combination of these dimensions, with metrics computed per window and globally.


Interpreting Metrics
--------------------

OpenSTEF's evaluation supports both deterministic metrics (applied to the median quantile) and probabilistic metrics (applied across all quantiles):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - What it tells you
   * - RMAE (Relative MAE)
     - Point forecast accuracy relative to a naive baseline. Values below 1.0 indicate the model outperforms persistence.
   * - RCRPS (Relative CRPS)
     - Probabilistic forecast skill relative to a baseline. Evaluates the full predictive distribution, not just the median.
   * - Observed Probability
     - Calibration check: do 90% prediction intervals actually contain 90% of observations? Always included automatically.

What constitutes "good" performance depends heavily on context:

- **Aggregation level** — a single household is inherently noisier than a substation serving thousands. Expect higher relative errors at lower aggregation.
- **Forecast horizon** — accuracy degrades with lead time. A 15-minute-ahead forecast might achieve 2-3% MAPE at substation level, while a 36-hour-ahead forecast might show 8-12%.
- **Season and weather** — performance typically degrades during extreme weather events and seasonal transitions.

Use the windowed metrics to identify periods where performance drops, which may indicate missing features or concept drift.

.. warning::

   A model that looks excellent in backtesting may still underperform in production if the backtest period does not include challenging conditions (extreme weather, holidays, grid topology changes). Always backtest across at least one full year if possible.


Probabilistic Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^

For probabilistic forecasts (see :doc:`/tutorials/backtesting_quickstart` for a full worked example), pay special attention to the observed probability metric. If your 90% prediction interval only captures 75% of observations, the model is overconfident. This matters for grid operations where under-estimated uncertainty can lead to insufficient reserve capacity.

The ``EvaluationPipeline`` automatically includes :class:`~openstef_beam.evaluation.ObservedProbabilityProvider` in global metrics to ensure calibration is always assessed.

Backtesting validates your model *before* deployment, but the forecaster you backtest should be as close as possible to what runs in production. The :class:`~openstef_beam.backtesting.BacktestForecasterMixin` interface is intentionally similar to the production forecasting interface described in :doc:`/user_guide/guides/forecasting`. The key difference is that backtesting wraps data access in ``RestrictedHorizonVersionedTimeSeries`` to enforce temporal constraints that production naturally provides (you simply cannot access tomorrow's meter readings today).

.. seealso::

   - :doc:`/user_guide/guides/forecasting` for the production forecasting workflow.
   - :doc:`/user_guide/guides/probabilistic_forecasting` for probabilistic output formats and quantile configuration.