Backtesting Your Models
=======================

This page covers how to evaluate forecasting models on historical data using
BEAM's ``BacktestPipeline``. Backtesting replays history as if it were happening
in real-time, ensuring your model is never exposed to future information during
training or prediction—giving you realistic performance estimates that match
what you'd see in production.

.. mermaid:: /diagrams/getting_started/backtesting_diagram_1.mmd

Why Backtest?
-------------

Standard train/test splits can overestimate model performance because they don't
account for operational constraints:

- **Data leakage prevention**: The model only sees data that would have been
  available at each point in time.
- **Realistic retraining**: Models are retrained on a schedule (e.g., weekly),
  just like in deployment.
- **Fair comparison**: Multiple models can be evaluated under identical
  conditions.

If you haven't built your first forecast yet, see :doc:`first_forecast` before
proceeding.

Backtest Configuration
----------------------

The ``BacktestConfig`` controls how the simulation replays history. Its three
core parameters define the temporal structure of the backtest:

.. code-block:: python

   from datetime import timedelta, time
   from openstef_beam.backtesting import BacktestConfig

   config = BacktestConfig(
       prediction_sample_interval=timedelta(minutes=15),
       predict_interval=timedelta(hours=6),
       train_interval=timedelta(days=7),
       align_time=time.fromisoformat("00:00+00"),
   )

**Parameters explained:**

- ``prediction_sample_interval`` — Resolution of the output forecast (default:
  15 minutes). Must match your forecaster's ``predict_sample_interval``.
- ``predict_interval`` — How often new predictions are generated during the
  backtest (default: every 6 hours). Smaller values produce more prediction
  windows but take longer to run.
- ``train_interval`` — How often the model is retrained (default: every 7 days).
  This simulates operational retraining schedules.
- ``align_time`` — Reference time for aligning prediction schedules to regular
  clock intervals (default: midnight UTC).

Running a Backtest
------------------

The ``BacktestPipeline`` orchestrates the full simulation. You provide it with
ground truth measurements, predictor features, and a time range:

.. code-block:: python

   from datetime import datetime
   from openstef_beam.backtesting import BacktestPipeline, BacktestConfig

   # Configure the backtest
   config = BacktestConfig(
       predict_interval=timedelta(hours=6),
       train_interval=timedelta(days=7),
   )

   # Initialize the pipeline with your forecaster
   pipeline = BacktestPipeline(
       config=config,
       forecaster=my_forecaster,  # implements BacktestForecasterMixin
   )

   # Run the backtest over a historical period
   predictions = pipeline.run(
       ground_truth=measurements,    # VersionedTimeSeriesDataset
       predictors=feature_data,      # VersionedTimeSeriesDataset
       start=datetime(2024, 1, 1),
       end=datetime(2024, 6, 1),
       show_progress=True,
   )

The ``run()`` method returns a ``TimeSeriesDataset`` containing all predictions
with their timestamps and availability information. Each prediction is tagged
with when it was generated, allowing you to analyze forecast accuracy as a
function of lead time.

.. note::

   The ``start`` and ``end`` parameters are optional. If omitted, the pipeline
   uses the minimum and maximum timestamps from your data.

The Forecaster Interface
^^^^^^^^^^^^^^^^^^^^^^^^

Your forecaster must implement the ``BacktestForecasterMixin`` interface. This
ensures the pipeline can call ``train()`` and ``predict()`` methods with the
correct temporal constraints. The pipeline validates that your forecaster's
``predict_sample_interval`` matches the config at initialization time.

Evaluation and Metrics
----------------------

After running a backtest, evaluate predictions against ground truth using the
``EvaluationPipeline``:

.. code-block:: python

   from openstef_beam.backtesting import BacktestPipeline

   # After obtaining predictions from the backtest...
   from openstef_beam.evaluation import EvaluationPipeline, EvaluationConfig

   eval_pipeline = EvaluationPipeline(
       config=EvaluationConfig(),
       quantiles=quantiles,
       window_metric_providers=metrics,
       global_metric_providers=metrics,
   )

   report = eval_pipeline.run(
       ground_truth=measurements,
       predictions=predictions,
   )

The evaluation pipeline computes metrics over configurable windows, giving you
both global performance summaries and time-resolved accuracy profiles.

Typical metrics for energy forecasting include:

- **MAE** (Mean Absolute Error) — Average magnitude of errors in the same units
  as your target variable.
- **RMSE** (Root Mean Square Error) — Penalizes large errors more heavily.
- **Quantile scores** — Evaluate probabilistic forecasts across confidence
  intervals.

Performance Visualization
-------------------------

BEAM provides visualization outputs that can be generated from evaluation
reports. These are organized by different grouping dimensions:

.. code-block:: python

   # Visualizations can be created by target, by run, or by group
   viz_output = visualizer.create_by_target(reports=report_tuples)

.. note:: [VISUALIZATION: Line plot comparing predicted vs actual load over the backtest period, with shaded confidence intervals showing quantile predictions and error bands]

Common visualization patterns include:

- **Predicted vs. actual** time series overlays for specific windows
- **Error distribution** histograms across the full backtest period
- **Performance by lead time** showing how accuracy degrades with forecast
  horizon
- **Retraining impact** showing metric changes after each model update

Full Example: End-to-End Backtest
---------------------------------

Here's a complete workflow tying configuration, execution, and evaluation
together:

.. code-block:: python

   from datetime import datetime, timedelta, time
   from openstef_beam.backtesting import BacktestConfig, BacktestPipeline

   # 1. Configure: predict every 6 hours, retrain weekly
   config = BacktestConfig(
       prediction_sample_interval=timedelta(minutes=15),
       predict_interval=timedelta(hours=6),
       train_interval=timedelta(days=7),
       align_time=time.fromisoformat("00:00+00"),
   )

   # 2. Build pipeline
   pipeline = BacktestPipeline(config=config, forecaster=my_forecaster)

   # 3. Run over 6 months of history
   predictions = pipeline.run(
       ground_truth=measurements,
       predictors=features,
       start=datetime(2024, 1, 1),
       end=datetime(2024, 7, 1),
   )

   # 4. Evaluate results
   # (see evaluation documentation for detailed metric configuration)

.. warning::

   Ensure your historical data covers enough time *before* the backtest start
   date for the initial training window. If the model requires 30 days of
   history to train, your data should begin at least 30 days before ``start``.

Tips and Best Practices
-----------------------

- **Start small**: Use a short backtest period (e.g., 2 weeks) to validate your
  setup before running multi-month simulations.
- **Match production settings**: Set ``predict_interval`` and
  ``train_interval`` to match your intended operational schedule for the most
  realistic results.
- **Compare models fairly**: Run multiple forecasters with the same
  ``BacktestConfig`` and time range to ensure apples-to-apples comparison.
- **Watch for seasonality**: Ensure your backtest period covers relevant
  seasonal patterns (weekdays/weekends, summer/winter) for representative
  results.

Next Steps
----------

- :doc:`advanced_customization` — Customize forecasters, callbacks, and pipeline
  behavior for complex scenarios.
- :doc:`first_forecast` — If you need to build a forecaster before backtesting.