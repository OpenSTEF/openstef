# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-run benchmark comparison and analysis pipeline.

Provides tools for comparing results across multiple benchmark runs, enabling
systematic evaluation of model improvements, parameter tuning effects, and
cross-validation analysis. Supports aggregated analysis at global, group,
and individual target levels.

The comparison pipeline operates on existing benchmark results, allowing
retrospective analysis without re-running expensive computations.
"""

import logging
from collections import defaultdict

from openstef_beam.analysis import AnalysisConfig, AnalysisPipeline, AnalysisScope
from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, TargetMetadata, TargetName
from openstef_beam.benchmarking.benchmark_pipeline import read_evaluation_reports
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation import EvaluationReport

_logger = logging.getLogger(__name__)


class BenchmarkComparisonPipeline[T: BenchmarkTarget, F]:
    """Pipeline for comparing results across multiple benchmark runs.

    Enables systematic comparison of forecasting models by analyzing results from
    multiple benchmark runs side-by-side. Provides aggregated analysis at different
    levels (global, group, target) to identify performance patterns and improvements.

    Use cases:
    - Compare model variants (different hyperparameters, algorithms)
    - Evaluate performance before/after model updates
    - Cross-validation analysis across different time periods
    - A/B testing of forecasting approaches

    The pipeline operates on existing benchmark results, making it efficient for
    retrospective analysis without re-running expensive computations.

    Multi-level analysis:
        The pipeline automatically generates analysis at three aggregation levels:
        - Global: Overall performance across all runs and targets
        - Group: Performance comparison within target groups
        - Target: Individual target performance across runs

    This hierarchical approach helps identify whether improvements are consistent
    across the entire portfolio or specific to certain target types.

    Example:
        Comparing three model versions across all targets:

        >>> from openstef_beam.benchmarking import BenchmarkComparisonPipeline
        >>> from openstef_beam.analysis import AnalysisConfig
        >>> from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
        >>> from openstef_beam.analysis.visualizations import SummaryTableVisualization
        >>> from pathlib import Path
        >>>
        >>> from openstef_beam.analysis.visualizations import (
        ...     GroupedTargetMetricVisualization,
        ...     TimeSeriesVisualization
        ... )
        >>>
        >>> # Configure comprehensive analysis
        >>> analysis_config = AnalysisConfig(
        ...     visualization_providers=[
        ...         GroupedTargetMetricVisualization(name="model_comparison", metric="rCRPS"),
        ...         SummaryTableVisualization(name="performance_summary"),
        ...         TimeSeriesVisualization(name="prediction_quality")
        ...     ]
        ... )
        >>>
        >>> # Set up comparison pipeline
        >>> comparison = BenchmarkComparisonPipeline(
        ...     analysis_config=analysis_config,
        ...     target_provider=...,
        ...     storage=...
        ... )
        >>>
        >>> # Compare multiple model versions across all targets
        >>> run_data = {
        ...     "baseline_v1": LocalBenchmarkStorage("results/baseline"),
        ...     "improved_v2": LocalBenchmarkStorage("results/improved"),
        ...     "experimental_v3": LocalBenchmarkStorage("results/experimental")
        ... }
        >>>
        >>> # Generate comprehensive comparison analysis
        >>> # comparison.run(
        >>> #    run_data=run_data,
        >>> # )
    """

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        target_provider: TargetProvider[T, F],
        storage: BenchmarkStorage,
    ):
        """Initialize the comparison pipeline.

        Args:
            analysis_config: Configuration for analysis and visualization generation.
            target_provider: Provider that supplies targets for comparison.
            storage: Storage backend for saving comparison results.
        """
        super().__init__()
        self.analysis_config = analysis_config
        self.target_provider = target_provider
        self.storage = storage
        self.pipeline = AnalysisPipeline(
            config=self.analysis_config,
        )

    def run(
        self,
        run_data: dict[RunName, BenchmarkStorage],
        filter_args: F | None = None,
    ):
        """Execute comparison analysis across multiple benchmark runs.

        Orchestrates the complete comparison workflow: loads evaluation reports
        from all specified runs, then generates comparative analysis at global,
        group, and target levels.

        Args:
            run_data: Mapping from run names to their corresponding storage backends.
                     Each storage backend should contain evaluation results for the run.
            filter_args: Optional criteria for filtering targets. Only targets
                        matching these criteria will be included in the comparison.
        """
        targets = self.target_provider.get_targets(filter_args)

        # Read evaluation reports for each run
        reports: list[tuple[TargetMetadata, EvaluationReport]] = []
        for run_name, run_storage in run_data.items():
            run_reports = read_evaluation_reports(
                targets=targets,
                storage=run_storage,
                run_name=run_name,
                strict=True,
            )
            reports.extend(run_reports)

        self.run_global(reports)
        self.run_for_groups(reports)
        self.run_for_targets(reports)

    def run_global(self, reports: list[tuple[TargetMetadata, EvaluationReport]]):
        """Generate global comparison analysis across all runs and targets.

        Creates aggregate visualizations comparing performance across all runs
        and target groups, providing a high-level overview of model improvements.

        Args:
            reports: List of target metadata and evaluation report pairs from all runs.
        """
        scope = AnalysisScope(
            aggregation=AnalysisAggregation.RUN_AND_GROUP,
        )
        if self.storage.has_analysis_output(scope=scope):
            _logger.info("Skipping global analysis, already exists")
            return

        _logger.info("Running analysis comparison for runs across groups")
        analysis = self.pipeline.run_for_reports(
            reports=reports,
            scope=scope,
        )
        self.storage.save_analysis_output(output=analysis)

    def run_for_groups(
        self,
        reports: list[tuple[TargetMetadata, EvaluationReport]],
    ):
        """Generate group-level comparison analysis for each target group.

        Creates comparative visualizations within each target group, showing
        how different runs perform for similar types of targets.

        Args:
            reports: List of target metadata and evaluation report pairs from all runs.
        """
        grouped: dict[GroupName, list[tuple[TargetMetadata, EvaluationReport]]] = defaultdict(list)
        for metadata, report in reports:
            grouped[metadata.group_name].append((metadata, report))

        for group_name, report_subset in grouped.items():
            scope = AnalysisScope(
                aggregation=AnalysisAggregation.RUN_AND_TARGET,
                group_name=group_name,
            )
            if self.storage.has_analysis_output(scope=scope):
                _logger.info(
                    "Skipping analysis for group %s, already exists",
                    group_name,
                )
                continue

            _logger.info("Running analysis for group comparison")
            run_analysis = self.pipeline.run_for_reports(
                reports=report_subset,
                scope=scope,
            )
            self.storage.save_analysis_output(output=run_analysis)

    def run_for_targets(
        self,
        reports: list[tuple[TargetMetadata, EvaluationReport]],
    ):
        """Generate target-level comparison analysis for individual targets.

        Creates detailed comparative visualizations for each individual target,
        showing how different runs perform on the same forecasting challenge.

        Args:
            reports: List of target metadata and evaluation report pairs from all runs.
        """
        grouped: dict[tuple[GroupName, TargetName], list[tuple[TargetMetadata, EvaluationReport]]] = defaultdict(list)
        for metadata, report in reports:
            grouped[metadata.group_name, metadata.name].append((metadata, report))

        for (group_name, target_name), report_subset in grouped.items():
            scope = AnalysisScope(
                aggregation=AnalysisAggregation.RUN_AND_NONE,
                target_name=target_name,
                group_name=group_name,
            )
            if self.storage.has_analysis_output(scope=scope):
                _logger.info(
                    "Skipping analysis for target %s in group %s, already exists",
                    target_name,
                    group_name,
                )
                continue

            _logger.info("Running analysis for target comparison")
            run_analysis = self.pipeline.run_for_reports(
                reports=report_subset,
                scope=scope,
            )
            self.storage.save_analysis_output(output=run_analysis)


__all__ = ["BenchmarkComparisonPipeline"]
