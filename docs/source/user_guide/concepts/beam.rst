BEAM
====

Energy forecasting teams need to compare models fairly. A model that appears superior on
one target or time horizon may underperform on another. Without a systematic framework,
comparisons become ad-hoc: different evaluation windows, inconsistent data splits, or
accidental future data leakage can all invalidate results. BEAM (Backtesting, Evaluation,
Analysis, and Metrics) solves this by providing a reproducible, sequential pipeline for
model comparison.

This page explains the architecture and design principles of BEAM. For a hands-on
walkthrough, see :ref:`guide_backtesting`.

Why Systematic Benchmarking Matters
-----------------------------------

In production energy forecasting, you must answer questions like:

- Does a new model outperform the current one across all targets?
- How does performance vary by lead time (1-hour ahead vs. day-ahead)?
- Are improvements consistent across seasonal windows?

Answering these requires three guarantees:

1. **No future data leakage** during backtesting (the model must never see data it would not have in production).
2. **Consistent segmentation** of results by horizon, availability time, and evaluation window.
3. **Reproducible analysis** that can be re-run or extended without re-executing expensive training.

BEAM encodes these guarantees into its pipeline structure.

Pipeline Overview
-----------------

BEAM is organized as three sequential stages, orchestrated by :class:`~openstef_beam.benchmarking.benchmark_pipeline.BenchmarkPipeline`:

.. mermaid:: /diagrams/user_guide/concepts/beam_diagram_1.mmd

.. list-table:: BEAM Pipeline Stages
   :header-rows: 1
   :widths: 20 30 25 25

   * - Stage
     - Responsibility
     - Input
     - Output
   * - BacktestPipeline
     - Simulate production forecasting over historical data
     - Ground truth, predictors, BacktestForecasterMixin
     - ``TimeSeriesDataset`` of predictions
   * - EvaluationPipeline
     - Segment predictions and compute metrics
     - Predictions, ground truth, metric providers
     - ``EvaluationReport``
   * - AnalysisPipeline
     - Generate visualizations at configurable aggregation levels
     - EvaluationReports, VisualizationProviders
     - ``AnalysisOutput`` (HTML)

These stages are completely decoupled. You can re-run evaluation with different metrics
without re-running the backtest, or generate new visualizations without recomputing metrics.

Backtesting Stage
-----------------

The :class:`~openstef_beam.backtesting.backtest_pipeline.BacktestPipeline` simulates how a
model would have performed in production by stepping through historical data sequentially.

Preventing Future Data Leakage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central design challenge in backtesting is ensuring the model never accesses data from
the future. BEAM solves this with
:class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries`,
a wrapper that restricts the model's view of the underlying dataset to only data available
before the current simulation timestamp.

.. warning::

   A :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin`
   is **not** the same as a Forecaster in openstef-models. It is glue code that adapts any
   model to the backtesting interface, receiving only a restricted data view and never
   having direct access to the full dataset.

Event-Driven Simulation
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~openstef_beam.backtesting.backtest_event_generator.BacktestEventGenerator`
schedules train and predict events according to the configured horizon and window step.
For each event, the pipeline either calls ``fit()`` or ``predict()`` on the forecaster,
always passing a :class:`~openstef_beam.backtesting.restricted_horizon_timeseries.RestrictedHorizonVersionedTimeSeries`
that enforces the temporal boundary.

The output is a :class:`~openstef_beam.datasets.TimeSeriesDataset` containing all
predictions, each tagged with an ``available_at`` timestamp indicating when that
prediction would have been generated in production.

Evaluation Stage
----------------

The :class:`~openstef_beam.evaluation.EvaluationPipeline` is entirely separate from
backtesting. It takes predictions and ground truth, then segments and scores them along
three dimensions:

- **available_at**: When the prediction was generated (e.g., "day-ahead at 06:00").
- **lead_time**: The gap between prediction generation and the target timestamp (e.g., 36 hours ahead).
- **time_windows**: Rolling evaluation periods for detecting performance drift.

Configurable Metrics
^^^^^^^^^^^^^^^^^^^^

Metrics are supplied as provider objects. BEAM includes providers such as
``RMAEProvider`` (Relative Mean Absolute Error) and ``RCRPSProvider`` (Relative
Continuous Ranked Probability Score) for probabilistic evaluation. You can implement
custom providers to add domain-specific metrics.

.. code-block:: python

   from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline
   from openstef_beam.evaluation.metric_providers import RMAEProvider, RCRPSProvider

   eval_pipeline = EvaluationPipeline(
       config=EvaluationConfig(),
       quantiles=forecaster.quantiles,
       window_metric_providers=[RMAEProvider(), RCRPSProvider()],
       global_metric_providers=[RMAEProvider(), RCRPSProvider()],
   )

The output is an :class:`~openstef_beam.evaluation.EvaluationReport` containing per-subset
metrics that can be inspected programmatically or passed to the analysis stage.

Analysis Stage
--------------

The :class:`~openstef_beam.analysis.analysis_pipeline.AnalysisPipeline` transforms
evaluation reports into HTML visualizations. It operates at configurable aggregation
levels:

.. list-table:: Aggregation Levels
   :header-rows: 1
   :widths: 30 70

   * - Level
     - Description
   * - NONE
     - Individual target, single run
   * - TARGET
     - Aggregate across runs for one target
   * - GROUP
     - Aggregate across targets within a group
   * - RUN_AND_NONE
     - Compare runs at individual target level
   * - RUN_AND_TARGET
     - Compare runs aggregated per target
   * - RUN_AND_GROUP
     - Compare runs aggregated per group

Visualization providers (e.g., ``SummaryTableVisualization``) are pluggable. Each
provider receives the relevant subset of evaluation data and produces a self-contained
HTML fragment.

BenchmarkPipeline: The Orchestrator
------------------------------------

:class:`~openstef_beam.benchmarking.benchmark_pipeline.BenchmarkPipeline` ties everything
together. For each target supplied by a ``TargetProvider``, it executes the three stages
sequentially and persists results via ``BenchmarkStorage``.

Key responsibilities:

- **Target iteration**: Acquires targets (with ground truth, predictors, and metric configuration) from a pluggable ``TargetProvider``.
- **Forecaster creation**: Uses a factory to instantiate a ``BacktestForecasterMixin`` per target.
- **Sequential execution**: Runs Backtest, then Evaluation, then Analysis for each target.
- **Parallel processing**: Supports processing multiple targets concurrently for efficiency.
- **Storage management**: Persists predictions, evaluation reports, and analysis outputs through a ``BenchmarkStorage`` backend (local filesystem, cloud, or in-memory).

Because results are persisted after each stage, you can use
:class:`~openstef_beam.benchmarking.benchmark_comparison_pipeline.BenchmarkComparisonPipeline`
to compare results across multiple benchmark runs without re-executing them.

Design Principles
-----------------

**Separation of concerns**: Each stage has a single responsibility and a well-defined
interface. This makes it possible to swap metric providers, visualization backends, or
storage implementations independently.

**Temporal integrity**: The ``RestrictedHorizonVersionedTimeSeries`` wrapper guarantees
that no stage can accidentally introduce future information into model training or
prediction.

**Reproducibility**: Configurations (``BacktestConfig``, ``EvaluationConfig``,
``AnalysisConfig``) are serializable data classes. Storing them alongside results ensures
any benchmark can be reproduced exactly.

**Scalability**: Parallel target processing and decoupled stages mean that large-scale
benchmarks (hundreds of targets, multiple models) remain tractable.

Relationship to Other Concepts
------------------------------

BEAM builds on the forecasting models described in :ref:`concept_models`. Any model that
can be wrapped in a :class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin`
can participate in a benchmark. The :ref:`concept_metalearning` system can use BEAM
results to inform model selection decisions across targets.

For a practical guide to setting up and running benchmarks, see :ref:`guide_backtesting`.