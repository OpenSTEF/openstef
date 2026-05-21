BEAM
====

This page explains the layered architecture of **BEAM** (Backtesting, Evaluation, Analysis, and Model-comparison), OpenSTEF's framework for systematic, reproducible forecasting model comparison. BEAM separates concerns into distinct pipeline stages so that each can be configured, tested, and extended independently.

.. mermaid:: /diagrams/user_guide/concepts/beam_diagram_1.mmd

Why BEAM Exists
---------------

Comparing forecasting models is deceptively hard. Naive approaches introduce subtle biases:

- **Future data leakage** — using information that wouldn't have been available at prediction time.
- **Inconsistent evaluation windows** — comparing models on different time periods or lead times.
- **Non-reproducibility** — ad-hoc scripts that can't be re-run or audited.

BEAM solves these problems by enforcing a strict separation between *generating predictions*, *evaluating predictions*, and *visualizing results*. Each stage has a single responsibility, clear inputs and outputs, and no hidden coupling to the others.

Layer 1: BacktestForecasterMixin
---------------------------------

The :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin` defines the interface that any forecasting model must implement to participate in a backtest. It is **not** the same as a Forecaster in ``openstef-models`` — it is glue code that simulates production data access patterns.

In production, a forecaster would request data from live APIs. In BEAM, the mixin's ``fit`` and ``predict`` methods receive a :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries` — a wrapper that provides only the data that would have been *published* before the simulated prediction moment. This prevents future data leakage by construction, not by convention.

.. code-block:: python

   class MyForecaster(BacktestForecasterMixin):
       def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None: ...
       def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None: ...

For batch-capable models, :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestBatchForecasterMixin` adds a ``predict_batch`` method for efficient multi-horizon prediction.

.. warning::

   ``RestrictedHorizonVersionedTimeSeries`` enforces temporal constraints at the data-access layer. Even if your model code accidentally requests future timestamps, the wrapper will not return them. This is the key mechanism that makes BEAM backtests trustworthy.

Layer 2: BacktestPipeline
--------------------------

:class:`~openstef_beam.backtesting.backtest_pipeline.BacktestPipeline` takes a ``BacktestForecasterMixin`` and replays history as if it were happening in real time.

**How it works:**

- A ``BacktestEventGenerator`` creates a schedule of **train** and **predict** events based on configured intervals (e.g., retrain daily, predict every 15 minutes).
- The pipeline iterates through events chronologically. At each train event, it calls ``forecaster.fit()`` with a horizon-restricted view. At each predict event, it calls ``forecaster.predict()``.
- Output: a single :class:`~openstef_beam.backtesting.backtest_pipeline.TimeSeriesDataset` containing all predictions, annotated with ``available_at`` timestamps.

**What it does NOT do:** BacktestPipeline does not evaluate predictions. It has no concept of metrics, accuracy, or ground truth comparison. This separation is intentional — it means you can re-evaluate the same predictions with different metrics without re-running the expensive simulation.

Layer 3: EvaluationPipeline
----------------------------

:class:`~openstef_beam.evaluation.EvaluationPipeline` is **completely separate** from BacktestPipeline. It takes predictions and ground truth as inputs and produces an :class:`~openstef_beam.evaluation.EvaluationReport`.

The key insight is *segmentation*: not all predictions are equally interesting. EvaluationPipeline slices results along multiple dimensions:

.. list-table:: Evaluation Segmentation Dimensions
   :header-rows: 1
   :widths: 25 75

   * - Dimension
     - Purpose
   * - ``available_at``
     - When the prediction was made (time-of-day, day-of-week effects)
   * - ``lead_time``
     - How far ahead the prediction looks (1h vs 24h accuracy)
   * - ``time_windows``
     - Which calendar period the target falls in (seasonal patterns)

Metrics (e.g., :class:`~openstef_beam.evaluation.metric_providers.RMAEProvider`, :class:`~openstef_beam.evaluation.metric_providers.RCRPSProvider`) are configurable via :class:`~openstef_beam.evaluation.EvaluationConfig`. Each metric is computed per subset, giving a multi-dimensional view of model performance.

Layer 4: AnalysisPipeline
--------------------------

:class:`~openstef_beam.analysis.analysis_pipeline.AnalysisPipeline` transforms ``EvaluationReport`` objects into human-readable HTML visualizations. It operates at three aggregation levels:

.. list-table:: Analysis Aggregation Levels
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Description
   * - ``NONE``
     - Individual target detail — one report, full diagnostic depth
   * - ``TARGET``
     - Cross-target comparison within a single model run
   * - ``GROUP``
     - Grouped comparison (e.g., by region, asset type, or model variant)

Visualizations are provided by pluggable :class:`~openstef_beam.analysis.visualizations.VisualizationProvider` implementations (e.g., ``SummaryTableVisualization``, ``GroupedTargetMetricVisualization``, ``TimeSeriesVisualization``). You configure which providers to use in :class:`~openstef_beam.analysis.analysis_pipeline.AnalysisConfig`.

Layer 5: BenchmarkPipeline
---------------------------

:class:`~openstef_beam.benchmarking.benchmark_pipeline.BenchmarkPipeline` is the top-level orchestrator that ties everything together. For each target it:

1. Obtains data from a **TargetProvider** (measurements, predictors, metadata).
2. Creates a ``BacktestForecasterMixin`` via a configured factory.
3. Runs **BacktestPipeline** → predictions.
4. Runs **EvaluationPipeline** → evaluation report.
5. Runs **AnalysisPipeline** → visualizations.
6. Persists all outputs to **BenchmarkStorage**.

.. code-block:: python

   pipeline = BenchmarkPipeline(
       backtest_config=backtest_config,
       evaluation_config=evaluation_config,
       analysis_config=analysis_config,
       target_provider=my_target_provider,
       storage=LocalBenchmarkStorage(base_path=Path("./results")),
   )

BenchmarkPipeline supports **parallel target processing** — multiple targets can be backtested concurrently since they are independent. Storage backends are pluggable: local filesystem, cloud storage, or in-memory for testing.

Design Principles
-----------------

BEAM's architecture embodies several principles that make model comparison trustworthy:

- **Temporal integrity by construction** — ``RestrictedHorizonVersionedTimeSeries`` makes data leakage structurally impossible, not just discouraged.
- **Separation of generation and evaluation** — predictions are a reusable artifact. Change your metrics without re-running backtests.
- **Reproducibility** — all configuration is declarative (``BacktestConfig``, ``EvaluationConfig``, ``AnalysisConfig``). Store configs alongside results for full audit trails.
- **Fair comparison** — all models see exactly the same data at exactly the same timestamps. The event schedule is model-agnostic.

Next Steps
----------

- :ref:`guide_backtesting` — practical guide to setting up and running backtests
- :doc:`models` — understand which models to compare using BEAM
- :doc:`intro_to_energy_forecasting` — context on the forecasting problem BEAM helps solve