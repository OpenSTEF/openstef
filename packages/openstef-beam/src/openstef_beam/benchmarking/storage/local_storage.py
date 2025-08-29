# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Local file system storage implementation for benchmark results.

Provides file system-based storage for benchmark artifacts including predictions,
evaluations, and analysis visualizations. Organizes results in a structured
directory hierarchy that supports efficient retrieval and conditional processing.
"""

from pathlib import Path

from openstef_beam.analysis import AnalysisOutput, AnalysisScope
from openstef_beam.analysis.models import AnalysisAggregation
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.evaluation import EvaluationReport
from openstef_core.datasets import VersionedTimeSeriesPart


class LocalBenchmarkStorage(BenchmarkStorage):
    """File system-based storage implementation for benchmark results.

    Stores benchmark artifacts (predictions, evaluations, and visualizations) in a
    structured directory hierarchy on the local file system. Supports conditional
    skipping of existing files to avoid redundant processing.

    Directory structure:
        base_path/
        ├── backtest/
        │   └── group_name/
        │       └── target_name/
        │           └── predictions.parquet
        ├── evaluation/
        │   └── group_name/
        │       └── target_name/
        └── analysis/
            ├── group_name/
            │   ├── target_name/  # Target-specific visualizations
            │   └── global/       # Group-level aggregated visualizations
            └── global/           # Global aggregated visualizations
    """

    def __init__(
        self,
        base_path: Path,
        *,
        skip_when_existing: bool = True,
        predictions_filename: str = "predictions.parquet",
        backtest_dirname: str = "backtest",
        evaluations_dirname: str = "evaluation",
        analysis_dirname: str = "analysis",
    ):
        """Initialize local file system storage.

        Args:
            base_path: Root directory where all benchmark artifacts will be stored.
            skip_when_existing: When True, has_* methods consider existing files as
                valid and skip reprocessing. When False, always indicates missing data.
            predictions_filename: Name of the parquet file for storing backtest predictions.
            backtest_dirname: Directory name for backtest predictions within base_path.
            evaluations_dirname: Directory name for evaluation reports within each target.
            analysis_dirname: Directory name for analysis visualizations.
        """
        self.base_path = base_path
        self.skip_when_existing = skip_when_existing
        self.predictions_filename = predictions_filename
        self.backtest_dirname = backtest_dirname
        self.evaluations_dirname = evaluations_dirname
        self.analysis_dirname = analysis_dirname

    def save_backtest_output(self, target: BenchmarkTarget, output: VersionedTimeSeriesPart) -> None:
        """Save backtest predictions to a parquet file."""
        predictions_path = self.get_predictions_path_for_target(target)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(predictions_path)

    def load_backtest_output(self, target: BenchmarkTarget) -> VersionedTimeSeriesPart:
        """Load backtest predictions from a parquet file.

        Returns:
            VersionedTimeSeriesPart: The loaded prediction data.
        """
        return VersionedTimeSeriesPart.read_parquet(
            path=self.get_predictions_path_for_target(target),
        )

    def has_backtest_output(self, target: BenchmarkTarget) -> bool:
        """Check if backtest output exists for the target.

        Returns:
            bool: True if backtest output exists and skip_when_existing is True.
        """
        return self.get_predictions_path_for_target(target).exists() and self.skip_when_existing

    def save_evaluation_output(self, target: BenchmarkTarget, output: EvaluationReport) -> None:
        """Save evaluation report to storage."""
        output.to_parquet(path=self.get_evaluations_path_for_target(target))

    def load_evaluation_output(self, target: BenchmarkTarget) -> EvaluationReport:
        """Load evaluation report from storage.

        Returns:
            EvaluationReport: The loaded evaluation report.
        """
        return EvaluationReport.read_parquet(path=self.get_evaluations_path_for_target(target))

    def has_evaluation_output(self, target: BenchmarkTarget) -> bool:
        """Check if evaluation output exists for the target.

        Returns:
            bool: True if evaluation output exists and skip_when_existing is True.
        """
        return self.get_evaluations_path_for_target(target).exists() and self.skip_when_existing

    def save_analysis_output(self, output: AnalysisOutput) -> None:
        """Save analysis visualizations to HTML files."""
        for filtering, visualizations in output.visualizations.items():
            output_dir = self.get_analysis_path(output.scope) / str(filtering)
            output_dir.mkdir(parents=True, exist_ok=True)

            for visualization in visualizations:
                visualization.write_html(output_dir / f"{visualization.name}.html")

    def has_analysis_output(self, scope: AnalysisScope) -> bool:
        """Check if analysis output exists for the given scope.

        Returns:
            bool: True if analysis output exists and skip_when_existing is True.
        """
        return self.get_analysis_path(scope).exists() and self.skip_when_existing

    def get_predictions_path_for_target(self, target: BenchmarkTarget) -> Path:
        """Returns the path for storing predictions for a target."""
        return (
            self.base_path
            / self.backtest_dirname
            / str(target.group_name)
            / str(target.name)
            / self.predictions_filename
        )

    def get_evaluations_path_for_target(self, target: BenchmarkTarget) -> Path:
        """Returns the path for storing evaluation results for a target."""
        return self.base_path / self.evaluations_dirname / str(target.group_name) / str(target.name)

    def get_analysis_path(self, scope: AnalysisScope) -> Path:
        """Get the file path for storing analysis output based on aggregation scope.

        Returns:
            Path: Directory path where analysis results should be stored.
        """
        base_dir = self.base_path / self.analysis_dirname
        if scope.aggregation == AnalysisAggregation.NONE:
            output_dir = base_dir / str(scope.group_name) / str(scope.target_name)
        elif scope.aggregation == AnalysisAggregation.TARGET:
            output_dir = base_dir / str(scope.group_name) / "global"
        elif scope.aggregation == AnalysisAggregation.GROUP:
            output_dir = base_dir / "global"
        elif scope.aggregation == AnalysisAggregation.RUN_AND_NONE:
            output_dir = base_dir / str(scope.group_name) / str(scope.target_name)
        elif scope.aggregation == AnalysisAggregation.RUN_AND_GROUP:
            output_dir = base_dir
        elif scope.aggregation == AnalysisAggregation.RUN_AND_TARGET:
            output_dir = base_dir / str(scope.group_name) / "global"
        else:
            # Default case for any new or unexpected aggregation types
            output_dir = base_dir

        return output_dir


__all__ = ["LocalBenchmarkStorage"]
