# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Benchmark pipeline for systematic forecasting model evaluation.

Provides a standardized framework for running comprehensive forecasting benchmarks
across multiple targets, models, and evaluation metrics. Coordinates backtesting,
evaluation, and analysis phases while managing parallel execution and result storage.

The benchmark pipeline follows a consistent workflow:
1. Target acquisition from configurable providers
2. Model training and backtesting for each target
3. Evaluation against ground truth with configurable metrics
4. Analysis and visualization of results
5. Storage of results for comparison and reporting

Key components:
- BenchmarkPipeline: Main orchestrator for benchmark execution
- ForecasterFactory: Factory pattern for creating target-specific models
- BenchmarkStorage: Pluggable storage backends for results
- BenchmarkCallback: Extensible event handling for monitoring and customization
"""

import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

from openstef_beam.analysis import AnalysisConfig, AnalysisPipeline, AnalysisScope
from openstef_beam.analysis.models import AnalysisAggregation, RunName, TargetMetadata
from openstef_beam.backtesting import BacktestConfig, BacktestPipeline
from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterMixin
from openstef_beam.benchmarking.callbacks import BenchmarkCallback, BenchmarkCallbackManager
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import InMemoryBenchmarkStorage
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline, EvaluationReport
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Quantile
from openstef_core.utils import run_parallel

_logger = logging.getLogger(__name__)


class BenchmarkContext(BaseConfig):
    """Context information passed to forecaster factories during benchmark execution.

    Provides runtime context that forecaster factories can use to customize
    model creation based on the benchmark environment and configuration.
    """

    run_name: str = "default"


type ForecasterFactory[T] = Callable[[BenchmarkContext, T], BacktestForecasterMixin]


class BenchmarkPipeline[T: BenchmarkTarget, F]:
    """Orchestrates comprehensive forecasting model benchmarks across multiple targets.

    Provides a standardized framework for systematic evaluation of forecasting models.
    Coordinates the entire benchmark workflow from data preparation through analysis,
    ensuring consistent evaluation methodology and result comparability.

    Core workflow:
    1. Target retrieval: Gets targets from configurable providers with optional filtering
    2. Model creation: Uses factory pattern to create target-specific forecasters
    3. Backtesting: Generates predictions using historical data with proper validation
    4. Evaluation: Computes performance metrics against ground truth
    5. Analysis: Creates visualizations and comparative reports
    6. Storage: Persists results for future analysis and comparison

    Key features:
    - Parallel execution support for efficient processing of large target sets
    - Pluggable storage backends (local filesystem, cloud storage, in-memory)
    - Extensible callback system for monitoring and custom processing
    - Automatic handling of data dependencies and validation
    - Consistent error handling and recovery mechanisms

    Example:
        Basic benchmark setup and execution:

        >>> from openstef_beam.benchmarking import BenchmarkPipeline
        >>> from openstef_beam.backtesting import BacktestConfig
        >>> from openstef_beam.evaluation import EvaluationConfig
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.evaluation.metric_providers import RMAEProvider, RCRPSProvider
        >>> from openstef_beam.analysis.visualizations import SummaryTableVisualization
        >>> from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
        >>> from datetime import timedelta
        >>> from pathlib import Path
        >>>
        >>> # Configure components
        >>> storage = LocalBenchmarkStorage(base_path=Path("./results"))
        >>> backtest_config = BacktestConfig(
        ...     horizon=timedelta(hours=24),
        ...     window_step=timedelta(days=1)
        ... )
        >>> evaluation_config = EvaluationConfig()
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[SummaryTableVisualization(name="summary")]
        ... )
        >>> # Create benchmark pipeline with target provider
        >>> pipeline = BenchmarkPipeline(
        ...     backtest_config=backtest_config,
        ...     evaluation_config=evaluation_config,
        ...     analysis_config=analysis_config,
        ...     storage=storage,
        ...     target_provider=...  # Your custom provider
        ... )
        >>>
        >>> # Define forecaster factory for target-specific models
        >>> def create_forecaster(context, target):
        ...     # Customize model configuration per target
        ...     return MyForecaster(config=target.get_model_config())
        >>>
        >>> # Execute complete benchmark with parallel processing
        >>> #pipeline.run(
        >>> #    forecaster_factory=create_forecaster,
        >>> #    run_name="baseline_comparison",
        >>> #    n_processes=4
        >>> #)
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        evaluation_config: EvaluationConfig,
        analysis_config: AnalysisConfig,
        target_provider: TargetProvider[T, F],
        storage: BenchmarkStorage | None = None,
        callbacks: list[BenchmarkCallback] | None = None,
    ) -> None:
        """Initializes the benchmark pipeline and sets up logging and configuration.

        Args:
            backtest_config: Configuration for the backtesting pipeline.
            evaluation_config: Configuration for the evaluation pipeline.
            analysis_config: Configuration for the analysis pipeline.
            target_provider: Provider that supplies benchmark targets and their data.
            storage: Storage backend for saving benchmark results. Defaults to in-memory storage.
            callbacks: Optional list of callbacks to manage benchmark events.
        """
        self.backtest_config = backtest_config
        self.evaluation_config = evaluation_config
        self.analysis_config = analysis_config
        self.target_provider = target_provider
        self.storage = cast(BenchmarkStorage, storage or InMemoryBenchmarkStorage())
        self.callback_manager = cast(BenchmarkCallback, BenchmarkCallbackManager(callbacks or []))

    def run(
        self,
        forecaster_factory: ForecasterFactory[T],
        run_name: str = "default",
        filter_args: F | None = None,
        n_processes: int | None = None,
    ) -> None:
        """Runs the benchmark for all targets, optionally filtered and in parallel.

        This is the main entry point for executing a benchmark. It:
        1. Gets all available targets from the target provider
        2. Optionally filters them based on provided criteria
        3. Processes each target sequentially or in parallel
        4. For each target, creates a forecastser and runs backtest and evaluation

        Args:
            forecaster_factory: Factory function that creates a forecaster for a target.
                               This allows customizing the model for each target.
            run_name: Name of the benchmark run, used for logging and result storage.
            filter_args: Optional filter criteria for targets. If provided, only targets
                        matching these criteria will be processed.
            n_processes: Number of processes to use for parallel execution. If None or 1,
                        targets are processed sequentially.
        """
        context = BenchmarkContext(run_name=run_name)

        targets = self.target_provider.get_targets(filter_args)

        if not self.callback_manager.on_benchmark_start(runner=self, targets=cast(list[BenchmarkTarget], targets)):
            return

        _logger.info("Running benchmark in parallel with %d processes", n_processes)
        run_parallel(
            process_fn=partial(self._run_for_target, context, forecaster_factory),
            items=targets,
            n_processes=n_processes,
            mode="fork",  # TODO: Change back to 'loky' after before commit
        )

        if not self.storage.has_analysis_output(
            AnalysisScope(
                aggregation=AnalysisAggregation.GROUP,
                run_name=context.run_name,
            )
        ):
            self.run_benchmark_analysis(
                context=context,
                targets=targets,
            )

        self.callback_manager.on_benchmark_complete(runner=self, targets=cast(list[BenchmarkTarget], targets))

    def _run_for_target(self, context: BenchmarkContext, model_factory: ForecasterFactory[T], target: T) -> None:
        """Run benchmark for a single target."""
        if not self.callback_manager.on_target_start(runner=self, target=target):
            _logger.info("Skipping target")
            return

        try:
            forecaster = model_factory(context, target)

            if not self.storage.has_backtest_output(target):
                _logger.info("Running backtest for target")
                self.run_backtest_for_target(target=target, forecaster=forecaster)

            if not self.storage.has_evaluation_output(target):
                _logger.info("Running evaluation for target")
                predictions = self.storage.load_backtest_output(target)
                self.run_evaluation_for_target(target=target, predictions=predictions, quantiles=forecaster.quantiles)

            if not self.storage.has_analysis_output(
                scope=AnalysisScope(
                    aggregation=AnalysisAggregation.TARGET,
                    target_name=target.name,
                    group_name=target.group_name,
                    run_name=context.run_name,
                )
            ):
                _logger.info("Running analysis for target")
                report = self.storage.load_evaluation_output(target)
                self.run_analysis_for_target(context=context, target=target, report=report)

        except Exception as e:
            _logger.exception("Error during benchmark for target")
            self.callback_manager.on_error(runner=self, target=target, error=e)

        _logger.info("Finished benchmark for target")
        self.callback_manager.on_target_complete(runner=self, target=target)

    def run_backtest_for_target(self, target: T, forecaster: BacktestForecasterMixin):
        """Runs the backtest for a single target and stores predictions."""
        if not self.callback_manager.on_backtest_start(runner=self, target=target):
            _logger.info("Skipping backtest for target", extra={"target_name": target.name})
            return

        pipeline = BacktestPipeline(
            config=self.backtest_config,
            forecaster=forecaster,
        )
        predictions = pipeline.run(
            ground_truth=self.target_provider.get_measurements_for_target(target),
            predictors=self.target_provider.get_predictors_for_target(target),
            start=target.benchmark_start,
            end=target.benchmark_end,
        )
        self.storage.save_backtest_output(target=target, output=predictions)
        self.callback_manager.on_backtest_complete(runner=self, target=target, predictions=predictions)

    def run_evaluation_for_target(
        self,
        target: T,
        quantiles: list[Quantile],
        predictions: TimeSeriesDataset,
    ) -> None:
        """Runs evaluation for a single target and stores results."""
        if not self.callback_manager.on_evaluation_start(runner=self, target=target):
            _logger.info("Skipping evaluation for target", extra={"target_name": target.name})
            return

        metrics = self.target_provider.get_metrics_for_target(target)
        pipeline = EvaluationPipeline(
            config=self.evaluation_config,
            quantiles=quantiles,
            window_metric_providers=metrics,
            global_metric_providers=metrics,
        )
        report = pipeline.run(
            ground_truth=self.target_provider.get_measurements_for_target(target),
            predictions=predictions,
            evaluation_mask=self.target_provider.get_evaluation_mask_for_target(target),
            target_column=self.target_provider.target_column,
        )
        self.storage.save_evaluation_output(target=target, output=report)
        self.callback_manager.on_evaluation_complete(runner=self, target=target, report=report)

    def run_analysis_for_target(
        self,
        context: BenchmarkContext,
        target: T,
        report: EvaluationReport,
    ):
        """Run analysis pipeline for a single target's evaluation results.

        Args:
            context: Benchmark execution context containing run metadata.
            target: Target that was evaluated.
            report: Evaluation report containing computed metrics.
        """
        pipeline = AnalysisPipeline(
            config=self.analysis_config,
        )

        metadata = TargetMetadata(
            name=target.name,
            group_name=target.group_name,
            run_name=context.run_name,
            filtering=None,
            limit=target.limit,
            upper_limit=target.upper_limit,
            lower_limit=target.lower_limit,
        )

        visualizations = pipeline.run_for_reports(
            reports=[(metadata, report)],
            scope=AnalysisScope(
                aggregation=AnalysisAggregation.NONE,
                target_name=target.name,
                group_name=target.group_name,
                run_name=context.run_name,
            ),
        )
        self.storage.save_analysis_output(output=visualizations)
        self.callback_manager.on_analysis_complete(runner=self, target=target, output=visualizations)

    def run_benchmark_analysis(
        self,
        context: BenchmarkContext,
        targets: Sequence[T],
    ):
        """Runs benchmark analysis for multiple targets."""
        reports = read_evaluation_reports(
            targets=targets,
            storage=self.storage,
            run_name=context.run_name,
        )

        pipeline = AnalysisPipeline(
            config=self.analysis_config,
        )

        # Visualizations aggregated by group
        analysis = pipeline.run_for_reports(
            reports=reports,
            scope=AnalysisScope(
                aggregation=AnalysisAggregation.GROUP,
                run_name=context.run_name,
            ),
        )
        self.storage.save_analysis_output(output=analysis)

        # For each group, create visualizations aggregated by target
        scope = AnalysisScope(
            aggregation=AnalysisAggregation.TARGET,
            run_name=context.run_name,
        )
        for output in pipeline.run_for_groups(reports=reports, scope=scope).values():
            self.storage.save_analysis_output(output=output)


def read_evaluation_reports[T: BenchmarkTarget](
    targets: Sequence[T], storage: BenchmarkStorage, run_name: RunName, *, strict: bool = True
) -> list[tuple[TargetMetadata, EvaluationReport]]:
    """Load evaluation reports for multiple targets from storage.

    Utility function for retrieving evaluation results from benchmark storage
    and formatting them for analysis workflows.

    Args:
        targets: Sequence of benchmark targets to load reports for.
        storage: Storage backend containing the evaluation outputs.
        run_name: Name of the benchmark run to load reports from.
        strict: If True, raises an error if any target's report is missing.
               If False, skips missing reports.

    Returns:
        List of tuples containing target metadata and evaluation reports.
    """
    return [
        (
            TargetMetadata(
                name=target.name,
                group_name=target.group_name,
                run_name=run_name,
                filtering=None,
                limit=target.limit,
                upper_limit=target.upper_limit,
                lower_limit=target.lower_limit,
            ),
            storage.load_evaluation_output(target),
        )
        for target in targets
        if storage.has_evaluation_output(target) or strict
    ]


__all__ = [
    "BenchmarkContext",
    "BenchmarkPipeline",
    "ForecasterFactory",
    "read_evaluation_reports",
]
