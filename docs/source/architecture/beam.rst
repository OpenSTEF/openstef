The openstef_beam Package
=========================

OpenSTEF BEAM (Backtesting, Evaluation, Analysis, and Metrics) is the evaluation
framework within the OpenSTEF ecosystem. It provides a structured pipeline for
answering the fundamental question: *how well does a forecasting model perform under
realistic operational conditions?*

BEAM's core dependency is ``openstef-core`` only—it works with any forecasting model
that implements its forecaster interface. The optional ``baselines`` extra
(``pip install openstef-beam[baselines]``) adds predefined benchmark forecasters built
on ``openstef-models`` and ``openstef-meta``, enabling out-of-the-box comparisons
against OpenSTEF's standard models.

.. mermaid:: /diagrams/architecture/beam_diagram_1.mmd

Pipeline Architecture
---------------------

BEAM decomposes model evaluation into three distinct phases, each handled by a
dedicated pipeline:

- **BacktestPipeline** — Replays historical data under realistic temporal constraints,
  producing forecasts as if the model were running in production.
- **EvaluationPipeline** — Compares forecasts against ground truth using configurable
  metrics, time windows, lead times, and data filters.
- **AnalysisPipeline** — Aggregates evaluation reports and generates visualizations
  at global, group, and individual target levels.

The **BenchmarkPipeline** orchestrates all three phases across multiple models and
targets, managing parallel execution and result storage. For comparing results across
separate benchmark runs, the **BenchmarkComparisonPipeline** operates on stored results
without re-running expensive computations.


Backtesting
-----------

The backtesting phase simulates operational forecasting by enforcing strict temporal
constraints—models never see future data during training or prediction. This prevents
data leakage and produces performance estimates that match real deployment.

.. code-block:: python

   from datetime import timedelta
   from openstef_beam.backtesting import BacktestConfig, BacktestPipeline

   config = BacktestConfig(
       prediction_sample_interval=timedelta(minutes=15),
   )

   pipeline = BacktestPipeline(config=config)

The ``BacktestPipeline`` generates ``BacktestEvent`` instances—discrete prediction
moments—and feeds each one to a forecaster through a
``RestrictedHorizonVersionedTimeSeries``. This wrapper ensures the forecaster can only
access data that would have been available at that point in time.

Custom Forecasters
^^^^^^^^^^^^^^^^^^

Any model can participate in backtesting by implementing the ``BacktestForecasterMixin``
interface:

.. code-block:: python

   from openstef_beam.backtesting.backtest_forecaster.mixins import (
       BacktestForecasterMixin,
   )
   from openstef_beam.backtesting.restricted_horizon_timeseries import (
       RestrictedHorizonVersionedTimeSeries,
   )
   from openstef_core.base_model import BaseModel
   from openstef_core.datasets import TimeSeriesDataset


   class MyCustomForecaster(BaseModel, BacktestForecasterMixin):
       """A custom forecaster for use with BEAM backtesting."""

       def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
           # Train your model using only historically-available data
           window = data.get_window(start=..., end=...)
           ...

       def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
           # Generate forecasts respecting the horizon restriction
           window = data.get_window(start=..., end=...)
           ...
           return forecast_dataset

This design means BEAM is **model-agnostic**. You can evaluate scikit-learn models,
neural networks, statistical methods, or any external forecasting library—as long as
you wrap it in the mixin interface.


Evaluation
----------

The evaluation phase applies metrics to backtest results, slicing performance across
multiple dimensions:

- **Time windows** — Compare accuracy across days, weeks, or seasons
- **Lead times** — Measure how accuracy degrades from 1-hour to 48-hour horizons
- **Data filtering** — Focus on specific conditions (peak hours, weekdays, etc.)

.. code-block:: python

   from openstef_beam.evaluation import (
       EvaluationConfig,
       EvaluationPipeline,
       EvaluationReport,
   )

   eval_config = EvaluationConfig()
   eval_pipeline = EvaluationPipeline(config=eval_config)

The pipeline produces ``EvaluationReport`` objects containing ``SubsetMetric`` values
organized by ``Window`` and ``Filtering`` criteria. These structured reports serve as
the input for the analysis phase.


Analysis
--------

The analysis phase transforms raw evaluation metrics into interpretable outputs.
It supports multiple aggregation levels through ``AnalysisScope``:

.. code-block:: python

   from openstef_beam.analysis import AnalysisConfig, AnalysisPipeline, AnalysisScope
   from openstef_beam.analysis.models import AnalysisAggregation

   analysis_config = AnalysisConfig(
       visualization_providers=[...],  # Custom visualization generators
       filterings=None,  # None means include all filterings
   )

   analysis_pipeline = AnalysisPipeline(config=analysis_config)

Visualization providers are pluggable—you can implement custom chart generators that
receive evaluation reports and produce ``VisualizationOutput`` objects.


Benchmarking: Orchestrating Complete Workflows
----------------------------------------------

The ``BenchmarkPipeline`` ties everything together, running the full
backtest → evaluate → analyze workflow across a matrix of targets and models:

.. code-block:: python

   from openstef_beam.benchmarking import BenchmarkPipeline

The benchmark pipeline follows this workflow:

1. **Target acquisition** — A ``TargetProvider`` supplies the list of forecasting
   targets (e.g., substations, grid nodes).
2. **Backtesting** — Each (target, model) pair is backtested under identical conditions.
3. **Evaluation** — Forecasts are scored against ground truth.
4. **Analysis** — Results are aggregated and visualized.
5. **Storage** — All artifacts are persisted via ``BenchmarkStorage`` for later
   comparison.

Using Predefined Baselines
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``baselines`` extra provides ready-made forecasters for benchmarking against
OpenSTEF's standard models:

.. code-block:: python

   # Requires: pip install openstef-beam[baselines]
   from openstef_beam.benchmarking.baselines.openstef4 import (
       create_openstef4_preset_backtest_forecaster,
   )

   # Create a factory that produces OpenSTEF4-based forecasters
   forecaster_factory = create_openstef4_preset_backtest_forecaster(
       workflow_config=my_workflow_config,
   )

This factory pattern allows the benchmark pipeline to instantiate fresh forecasters
for each target, ensuring clean state between evaluations.

For details on the models and workflows available through the baselines extra, see
the sibling pages on :doc:`models` and :doc:`meta`.

Comparing Benchmark Runs
^^^^^^^^^^^^^^^^^^^^^^^^^

After running multiple benchmarks (e.g., with different model configurations), use
``BenchmarkComparisonPipeline`` to analyze differences without re-running forecasts:

.. code-block:: python

   from openstef_beam.benchmarking.benchmark_comparison_pipeline import (
       BenchmarkComparisonPipeline,
   )
   from openstef_beam.benchmarking.storage import BenchmarkStorage

   comparison = BenchmarkComparisonPipeline(config=analysis_config)

   run_data = {
       "baseline_v1": BenchmarkStorage(path="results/run_baseline"),
       "new_model_v2": BenchmarkStorage(path="results/run_new"),
   }

   comparison.run(run_data=run_data)

This enables systematic evaluation of model improvements, hyperparameter tuning
effects, and cross-validation analysis from stored results.


Dependency Structure
--------------------

BEAM is intentionally lightweight in its core dependencies:

- **openstef-beam** → depends on ``openstef-core`` only
- **openstef-beam[baselines]** → additionally pulls in ``openstef-models`` and
  ``openstef-meta`` for predefined benchmark forecasters

This means you can use BEAM to evaluate *any* forecasting approach—not just OpenSTEF
models—by implementing the forecaster mixin against ``openstef-core`` types like
``TimeSeriesDataset`` and ``VersionedTimeSeriesDataset``.

.. note::

   For details on the core data types used throughout BEAM (``TimeSeriesDataset``,
   ``BaseConfig``, etc.), see :doc:`core`.