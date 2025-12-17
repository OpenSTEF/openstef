# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Analysis pipeline for generating visualizations from evaluation reports.

This module provides the core pipeline that orchestrates visualization generation
from evaluation reports at different aggregation levels. It integrates with the
benchmarking framework to provide consistent analysis outputs across benchmark runs.
"""

from collections import defaultdict
from collections.abc import Sequence

from pydantic import Field

from openstef_beam.analysis.models import (
    AnalysisAggregation,
    AnalysisOutput,
    AnalysisScope,
    GroupName,
    TargetMetadata,
    VisualizationOutput,
)
from openstef_beam.analysis.visualizations import ReportTuple, VisualizationProvider
from openstef_beam.evaluation import EvaluationReport
from openstef_beam.evaluation.models import Filtering
from openstef_core.base_model import BaseConfig
from openstef_core.utils.itertools import groupby


class AnalysisConfig(BaseConfig):
    """Configuration for the analytics pipeline."""

    visualization_providers: list[VisualizationProvider] = Field(
        default=[], description="List of visualization providers to use for generating analysis outputs"
    )


class AnalysisPipeline:
    """Orchestrates the generation of visualizations from evaluation reports.

    The pipeline processes evaluation reports at different aggregation levels:
    - Individual targets: Creates detailed visualizations for single targets
    - Multiple targets: Creates comparative visualizations across target groups

    It integrates with the benchmarking framework by being called from BenchmarkPipeline
    after evaluation is complete, ensuring consistent visualization generation across
    all benchmark runs.
    """

    def __init__(
        self,
        config: AnalysisConfig,
    ) -> None:
        """Initialize the analysis pipeline with configuration.

        Args:
            config: Analysis configuration containing visualization providers.
        """
        super().__init__()
        self.config = config

    @staticmethod
    def _group_by_filtering(
        reports: Sequence[tuple[TargetMetadata, EvaluationReport]],
    ) -> dict[Filtering, list[ReportTuple]]:
        """Group reports by their lead time filtering conditions.

        Organizes evaluation reports based on their lead time criteria (e.g.,
        1-hour ahead vs 24-hour ahead forecasts), enabling comparison of model
        performance across different forecasting horizons.

        Returns:
            Dictionary mapping lead time filtering conditions to lists of report tuples.
        """
        return groupby(
            (subset.filtering, (base_metadata.with_filtering(subset.filtering), subset))
            for base_metadata, report in reports
            for subset in report.subset_reports
        )

    def run_for_subsets(
        self,
        reports: list[ReportTuple],
        aggregation: AnalysisAggregation,
    ) -> list[VisualizationOutput]:
        """Generate visualizations for a set of evaluation reports at a specific aggregation level.

        Processes the provided evaluation reports through all configured visualization
        providers that support the requested aggregation level. Only providers that
        declare support for the aggregation are used.

        Args:
            reports: List of (metadata, evaluation_subset_report) tuples to visualize.
                The metadata provides context about the target and run, while the
                evaluation report contains the metrics and predictions to visualize.
            aggregation: The aggregation level determining how reports are grouped
                and compared in visualizations.

        Returns:
            List of visualization outputs from all applicable providers. Empty if
            no providers support the requested aggregation level.
        """
        return [
            provider.create(
                reports=reports,
                aggregation=aggregation,
            )
            for provider in self.config.visualization_providers
            if aggregation in provider.supported_aggregations
        ]

    def run_for_reports(
        self,
        reports: Sequence[tuple[TargetMetadata, EvaluationReport]],
        scope: AnalysisScope,
    ) -> AnalysisOutput:
        """Generate visualizations for evaluation reports within a specific scope.

        Groups reports by lead time filtering conditions and generates visualizations
        for each group using all configured visualization providers that support the
        scope's aggregation level. This enables comparing model performance across
        different forecasting horizons (e.g., short-term vs long-term predictions).

        Args:
            reports: List of (metadata, evaluation_report) tuples to visualize.
            scope: Analysis scope defining aggregation level and context.

        Returns:
            Analysis output containing all generated visualizations grouped by
            lead time filtering conditions.
        """
        grouped = self._group_by_filtering(reports)

        result: dict[Filtering, list[VisualizationOutput]] = defaultdict(list)
        for filtering, subset_reports in grouped.items():
            visualizations = self.run_for_subsets(
                reports=subset_reports,
                aggregation=scope.aggregation,
            )
            result[filtering].extend(visualizations)

        return AnalysisOutput(
            scope=scope,
            visualizations=result,
        )

    def run_for_groups(
        self,
        reports: Sequence[tuple[TargetMetadata, EvaluationReport]],
        scope: AnalysisScope,
    ) -> dict[GroupName, AnalysisOutput]:
        """Generate visualizations for multiple target groups at a specific aggregation level.

        This method processes all evaluation reports, grouping them by their target
        group names and generating visualizations for each group.

        Args:
            reports: List of (metadata, evaluation_report) tuples to visualize.
                The metadata provides context about the target and run, while the
                evaluation report contains the metrics and predictions to visualize.
            scope: The analytics scope defining how reports are grouped and aggregated.

        Returns:
            Dictionary mapping group names to their corresponding AnalyticsOutput.
        """
        reports_by_group: dict[GroupName, list[tuple[TargetMetadata, EvaluationReport]]] = groupby(
            (report[0].group_name, report) for report in reports
        )

        result: dict[GroupName, AnalysisOutput] = {}
        for group_name, group_reports in reports_by_group.items():
            if not group_reports:
                continue

            result[group_name] = self.run_for_reports(
                reports=group_reports,
                scope=AnalysisScope(
                    aggregation=scope.aggregation,
                    target_name=scope.target_name,
                    run_name=scope.run_name,
                    group_name=group_name,
                ),
            )

        return result
