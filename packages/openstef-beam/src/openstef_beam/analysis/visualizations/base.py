# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import abstractmethod

from pydantic import Field

from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, TargetMetadata, VisualizationOutput
from openstef_beam.evaluation import EvaluationSubsetReport
from openstef_core.base_model import BaseConfig
from openstef_core.types import Quantile
from openstef_core.utils.itertools import groupby, is_all_same

ReportTuple = tuple[TargetMetadata, EvaluationSubsetReport]


class VisualizationProvider(BaseConfig):
    """Abstract base class for creating visualizations from evaluation reports.

    Provides a unified interface for generating different types of visualizations
    at various aggregation levels. Subclasses must implement specific visualization
    logic for each supported aggregation type.
    """

    name: str = Field(description="Name of the visualization provider, used for identification in reports.")

    def create(
        self,
        reports: list[ReportTuple],
        aggregation: AnalysisAggregation,
    ) -> VisualizationOutput:
        """Creates a visualization based on evaluation reports and aggregation level.

        Validates the aggregation type, groups reports appropriately, and delegates
        to the specific creation method. Ensures data consistency constraints are
        met for each aggregation type.

        Args:
            reports: List of (metadata, evaluation_report) tuples containing the
                data to visualize.
            aggregation: The aggregation level determining how reports are grouped
                and visualized.

        Returns:
            A visualization output containing the generated plot or HTML content.

        Raises:
            ValueError: If aggregation is not supported by this provider, if the
                number of reports doesn't match the aggregation requirements, if
                reports have inconsistent run_name for GROUP aggregation, or if
                reports have inconsistent group_name for RUN aggregation.
        """
        # Validate aggregation support upfront
        self._validate_aggregation_support(aggregation)

        # Not NONE aggregations require non-empty reports
        if aggregation != AnalysisAggregation.NONE and len(reports) == 0:
            msg = f"No reports provided for {aggregation.value} aggregation."
            raise ValueError(msg)

        # Use match/case to dispatch aggregation handling
        match aggregation:
            case AnalysisAggregation.NONE:
                # Early return for unaggregated (single report) case
                if len(reports) != 1:
                    raise ValueError("Cannot create unaggregated visualization for multiple reports.")
                metadata, report = reports[0]
                return self.create_by_none(report=report, metadata=metadata)

            case AnalysisAggregation.TARGET:
                _validate_same_run_names(reports)
                # Create visualization for each target in the same run
                return self.create_by_target(reports=reports)

            case AnalysisAggregation.GROUP:
                _validate_same_run_names(reports)
                # Group by group_name to compare performance across target categories
                grouped_reports = groupby(((m.group_name, (m, r)) for m, r in reports))
                return self.create_by_group(reports=grouped_reports)

            case AnalysisAggregation.RUN_AND_NONE:
                _validate_same_group_names(reports)
                # Group by run_name to compare different models on the same target
                grouped_reports = groupby(((m.run_name, (m, r)) for m, r in reports))
                return self.create_by_run_and_none(reports=grouped_reports)

            case AnalysisAggregation.RUN_AND_TARGET:
                _validate_same_group_names(reports)
                # Group by run_name to compare different models on the same target
                grouped_reports = groupby(((m.run_name, (m, r)) for m, r in reports))
                return self.create_by_run_and_target(reports=grouped_reports)

            case AnalysisAggregation.RUN_AND_GROUP:
                # Group by both run_name and group_name for comprehensive comparison matrix
                grouped_reports = groupby((((m.run_name, m.group_name), (m, r)) for m, r in reports))
                return self.create_by_run_and_group(reports=grouped_reports)

    def create_by_none(
        self,
        report: EvaluationSubsetReport,
        metadata: TargetMetadata,
    ) -> VisualizationOutput:
        """Creates visualization for a single target from a single run.

        Generates detailed analysis for individual target performance, typically
        showing time series, detailed metrics, or target-specific insights.

        Returns:
            Visualization focused on the specific target's performance.
        """
        raise NotImplementedError

    def create_by_target(
        self,
        reports: list[ReportTuple],
    ) -> VisualizationOutput:
        """Creates visualization comparing multiple targets from the same run.

        Groups reports by target metadata and creates visualizations showing
        performance differences across individual targets within the same model run.

        Args:
            reports: List of (metadata, report) tuples for each target in the run.

        Returns:
            Visualization comparing performance across different targets.
        """
        raise NotImplementedError

    def create_by_group(self, reports: dict[GroupName, list[ReportTuple]]) -> VisualizationOutput:
        """Creates visualization comparing multiple targets from the same run.

        Groups targets by their group_name and creates comparative visualizations
        showing performance differences across target categories or types.

        Args:
            reports: Dictionary mapping group names to lists of (metadata, report)
                tuples for that group.

        Returns:
            Visualization comparing performance across different target groups.
        """
        raise NotImplementedError

    def create_by_run_and_none(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        """Creates visualization comparing multiple runs on the same target group.

        Groups reports by run_name and creates comparative visualizations showing
        how different models or configurations perform on the same targets.

        Args:
            reports: Dictionary mapping run names to lists of (metadata, report)
                tuples for that run.

        Returns:
            Visualization comparing different model runs on the same targets.
        """
        raise NotImplementedError

    def create_by_run_and_target(self, reports: dict[RunName, list[ReportTuple]]) -> VisualizationOutput:
        """Creates visualization comparing multiple runs on the same target group.

        Groups reports by run_name and creates comparative visualizations showing
        how different models or configurations perform on the same targets.

        Args:
            reports: Dictionary mapping run names to lists of (metadata, report)
                tuples for that run.

        Returns:
            Visualization comparing different model runs on the same targets.
        """
        raise NotImplementedError

    def create_by_run_and_group(
        self, reports: dict[tuple[RunName, GroupName], list[ReportTuple]]
    ) -> VisualizationOutput:
        """Creates comprehensive visualization across multiple runs and target groups.

        Creates matrix-style comparisons showing how different models perform
        across different target categories, enabling full comparative analysis.

        Args:
            reports: Dictionary mapping (run_name, group_name) tuples to lists
                of (metadata, report) tuples for that combination.

        Returns:
            Comprehensive visualization matrix comparing runs across target groups.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_aggregations(self) -> set[AnalysisAggregation]:
        """Returns the set of aggregation types supported by this provider.

        Returns:
            Set of supported VisualizationAggregation values.
        """
        raise NotImplementedError

    def _validate_aggregation_support(self, aggregation: AnalysisAggregation) -> None:
        """Validate that the aggregation type is supported by this provider."""
        if aggregation not in self.supported_aggregations:
            msg = f"Aggregation {aggregation} is not supported by this provider."
            raise ValueError(msg)


def _validate_same_run_names(
    reports: list[ReportTuple],
) -> None:
    """Validate that all reports have the same run name."""
    run_names = [metadata.run_name for metadata, _ in reports]
    if not is_all_same(run_names):
        raise ValueError("All reports must have the same run name.")


def _validate_same_group_names(
    reports: list[ReportTuple],
) -> None:
    """Validate that all reports have the same group name."""
    group_names = [metadata.group_name for metadata, _ in reports]
    if not is_all_same(group_names):
        raise ValueError("All reports must have the same group.")


type MetricIdentifier = str | tuple[str, Quantile]


__all__ = ["MetricIdentifier", "ReportTuple", "VisualizationProvider"]
