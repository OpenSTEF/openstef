# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Base classes and interfaces for benchmark result storage.

Defines the storage abstraction for benchmark artifacts including backtest outputs,
evaluation reports, and analysis visualizations. Provides a unified interface
that can be implemented for different storage backends (local filesystem, cloud
storage, databases, in-memory).

The storage interface ensures consistent data access patterns across different
deployment environments while maintaining data integrity and enabling efficient
retrieval for analysis and comparison workflows.
"""

from abc import ABC, abstractmethod
from typing import override

from openstef_beam.analysis import AnalysisOutput, AnalysisScope
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.evaluation import EvaluationReport
from openstef_core.datasets import VersionedTimeSeriesDataset


class BenchmarkStorage(ABC):
    """Abstract base class for storing and retrieving benchmark results.

    Provides a unified interface for persisting all benchmark artifacts across different
    storage backends. The primary responsibility is ensuring consistent storage and
    retrieval of data, maintaining the temporal versioning semantics of forecasts.

    Storage responsibilities:
    - Backtest outputs: Time series predictions with temporal versioning (forecasts
      made at different times for the same target period, becoming more accurate
      closer to the actual time)
    - Evaluation reports: Performance metrics and analysis results
    - Analysis outputs: Visualizations and comparative analysis artifacts

    Implementation requirements:
    - Consistent data storage and retrieval patterns
    - Preserve temporal versioning information in the stored data
    - Handle data organization schemes appropriate for the storage backend
    - Provide reliable error handling for missing or corrupted data

    Example:
        Using storage in a benchmark pipeline:

        >>> from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
        >>> from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
        >>> from pathlib import Path
        >>>
        >>> # Configure storage backend (testable)
        >>> storage = LocalBenchmarkStorage(
        ...     base_path=Path("./benchmark_results")
        ... )
        >>>
        >>> # Test storage creation
        >>> isinstance(storage, LocalBenchmarkStorage)
        True
        >>> storage.base_path.name
        'benchmark_results'

        Integration with benchmark pipeline: # doctest: +SKIP

        >>> from openstef_beam.benchmarking import BenchmarkPipeline
        >>>
        >>> # Use in complete benchmark setup
        >>> pipeline = BenchmarkPipeline(
        ...     backtest_config=...,
        ...     evaluation_config=...,
        ...     analysis_config=...,
        ...     target_provider=...,
        ...     storage=storage  # Handles all result persistence
        ... )
        >>>
        >>> # Storage automatically manages:
        >>> # - Backtest outputs (predictions with temporal versioning)
        >>> # - Evaluation reports (metrics across time windows)
        >>> # - Analysis visualizations (charts and summary tables)
        >>> # pipeline.run(forecaster_factory=my_factory)

    Custom storage implementation:

        >>> class DatabaseStorage(BenchmarkStorage):
        ...     def __init__(self, db_connection):
        ...         self.db = db_connection
        ...
        ...     def save_backtest_output(self, target, output):
        ...         # Store forecast data preserving temporal versioning
        ...         self.db.save_predictions(
        ...             target_id=target.name,
        ...             predictions=output,  # Contains timestamp + available_at columns
        ...             metadata=target.metadata
        ...         )
        ...
        ...     def load_backtest_output(self, target):
        ...         # Retrieve data maintaining temporal versioning structure
        ...         return self.db.load_predictions(target_id=target.name)

    The storage interface enables seamless switching between local development,
    cloud deployment, and custom enterprise systems while preserving the temporal
    nature of forecast data across all backends.
    """

    @abstractmethod
    def save_backtest_output(self, target: BenchmarkTarget, output: VersionedTimeSeriesDataset) -> None:
        """Save the backtest output for a specific benchmark target.

        Stores the results of a backtest execution, associating it with the target
        configuration. Must handle overwrites of existing data gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def load_backtest_output(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        """Load previously saved backtest output for a benchmark target.

        Returns:
            The stored backtest results as a VersionedTimeSeriesDataset.

        Raises:
            KeyError: When no backtest output exists for the given target.
        """
        raise NotImplementedError

    @abstractmethod
    def has_backtest_output(self, target: BenchmarkTarget) -> bool:
        """Check if backtest output exists for the given benchmark target.

        Returns:
            True if backtest output is stored for the target, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def save_evaluation_output(self, target: BenchmarkTarget, output: EvaluationReport) -> None:
        """Save the evaluation report for a specific benchmark target.

        Stores the evaluation metrics and analysis results, associating them with
        the target configuration. Must handle overwrites of existing data gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def load_evaluation_output(self, target: BenchmarkTarget) -> EvaluationReport:
        """Load previously saved evaluation report for a benchmark target.

        Returns:
            The stored evaluation report containing metrics and analysis results.

        Raises:
            KeyError: When no evaluation output exists for the given target.
        """
        raise NotImplementedError

    @abstractmethod
    def has_evaluation_output(self, target: BenchmarkTarget) -> bool:
        """Check if evaluation output exists for the given benchmark target.

        Returns:
            True if evaluation output is stored for the target, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def save_analysis_output(self, output: AnalysisOutput) -> None:
        """Save analysis output, optionally associated with a benchmark target.

        Args:
            output: The analysis results to store, typically containing insights
        """
        raise NotImplementedError

    @abstractmethod
    def has_analysis_output(self, scope: AnalysisScope) -> bool:
        """Check if analysis output exists for the given target or global scope.

        Args:
            scope: The scope of the analysis output to check.

        Returns:
            True if analysis output exists for the specified scope, False otherwise.
        """
        raise NotImplementedError


class InMemoryBenchmarkStorage(BenchmarkStorage):
    """In-memory implementation of BenchmarkStorage for testing and temporary use.

    Stores all benchmark data in memory using dictionaries. Data is lost when the
    instance is destroyed. Does not support analysis output storage.

    Note:
        This implementation is not suitable for production use with large datasets
        or when persistence across sessions is required.
    """

    def __init__(self):
        """Initialize empty in-memory storage containers."""
        self._backtest_outputs: dict[str, VersionedTimeSeriesDataset] = {}
        self._evaluation_outputs: dict[str, EvaluationReport] = {}
        self._analytics_outputs: dict[AnalysisScope, AnalysisOutput] = {}

    @override
    def save_backtest_output(self, target: BenchmarkTarget, output: VersionedTimeSeriesDataset) -> None:
        self._backtest_outputs[target.name] = output

    @override
    def load_backtest_output(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        return self._backtest_outputs[target.name]

    @override
    def has_backtest_output(self, target: BenchmarkTarget) -> bool:
        return target.name in self._backtest_outputs

    @override
    def save_evaluation_output(self, target: BenchmarkTarget, output: EvaluationReport) -> None:
        self._evaluation_outputs[target.name] = output

    @override
    def load_evaluation_output(self, target: BenchmarkTarget) -> EvaluationReport:
        return self._evaluation_outputs[target.name]

    @override
    def has_evaluation_output(self, target: BenchmarkTarget) -> bool:
        return target.name in self._evaluation_outputs

    @override
    def save_analysis_output(self, output: AnalysisOutput) -> None:
        self._analytics_outputs[output.scope] = output

    @override
    def has_analysis_output(self, scope: AnalysisScope) -> bool:
        """Always returns False - analysis output is not supported in memory storage."""
        return scope in self._analytics_outputs

    @property
    def analysis_data(self) -> dict[AnalysisScope, AnalysisOutput]:
        return self._analytics_outputs


__all__ = ["BenchmarkStorage", "InMemoryBenchmarkStorage"]
