# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Callback system for benchmark pipeline event handling.

Provides extensible hooks into the benchmark execution workflow, allowing
custom monitoring, logging, result processing, and external integrations.
Callbacks receive notifications at key points in the benchmark lifecycle.

The callback system enables:
- Progress monitoring and reporting
- Custom logging and metrics collection
- Integration with external systems (databases, monitoring tools)
- Early termination based on custom conditions
- Custom result processing and analysis
"""

from typing import TYPE_CHECKING, Any, override

from openstef_beam.analysis import AnalysisOutput
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.evaluation import EvaluationReport
from openstef_core.datasets import VersionedTimeSeriesPart

if TYPE_CHECKING:
    from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline


class BenchmarkCallback:
    """Base class for benchmark execution callbacks.

    Provides hooks into the benchmark pipeline lifecycle, enabling custom monitoring,
    logging, progress tracking, and external integrations. Callbacks receive notifications
    at key execution points and can influence the benchmark flow.

    Callback methods can return boolean values to control execution flow:
    - Returning False from start methods (on_benchmark_start, on_target_start, etc.)
      will skip that phase of execution
    - Complete methods are purely informational and don't affect flow control

    Example:
        Creating a custom progress monitoring callback:

        >>> from openstef_beam.benchmarking.callbacks import BenchmarkCallback
        >>> import logging
        >>>
        >>> class ProgressCallback(BenchmarkCallback):
        ...     def __init__(self):
        ...         self.logger = logging.getLogger("benchmark.progress")
        ...         self.total_targets = 0
        ...         self.completed_targets = 0
        ...
        ...     def on_benchmark_start(self, runner, targets):
        ...         self.total_targets = len(targets)
        ...         self.logger.info(f"Starting benchmark for {self.total_targets} targets")
        ...         return True
        ...
        ...     def on_target_complete(self, runner, target):
        ...         self.completed_targets += 1
        ...         progress = (self.completed_targets / self.total_targets) * 100
        ...         self.logger.info(f"Completed {target.name} ({progress:.1f}%)")
        ...
        ...     def on_error(self, runner, target, error):
        ...         self.logger.error(f"Error processing {target.name}: {error}")

        Early termination based on conditions:

        >>> class QualityGateCallback(BenchmarkCallback):
        ...     def __init__(self, max_mae_threshold=100.0):
        ...         self.threshold = max_mae_threshold
        ...
        ...     def on_evaluation_complete(self, runner, target, report):
        ...         mae = report.get_metric("MAE")
        ...         if mae > self.threshold:
        ...             logging.warning(f"Target {target.name} exceeds MAE threshold")
        ...             # Could trigger alerts or early termination logic

    The callback system enables extensive customization while maintaining
    clean separation between benchmark execution and monitoring concerns.
    """

    def on_benchmark_start(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> bool:
        """Called when benchmark starts.

        Returns:
            bool: True if benchmark should start, False to skip.
        """
        _ = self, runner, targets  # Suppress unused variable warning
        return True

    def on_target_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called when processing a target begins.

        Returns:
            bool: True if target processing should start, False to skip.
        """
        _ = self, runner, target  # Suppress unused variable warning
        return True

    def on_backtest_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called before backtest execution.

        Returns:
            bool: True if backtesting should start, False to skip.
        """
        _ = self, runner, target  # Suppress unused variable warning
        return True

    def on_backtest_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, predictions: VersionedTimeSeriesPart
    ) -> None:
        """Called after backtest completes."""
        _ = self, runner, target, predictions  # Suppress unused variable warning

    def on_evaluation_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called before evaluation starts.

        Returns:
            bool: True if evaluation should start, False to skip.
        """
        _ = self, runner, target  # Suppress unused variable warning
        return True

    def on_evaluation_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, report: EvaluationReport
    ) -> None:
        """Called after evaluation completes."""
        _ = self, runner, target, report  # Suppress unused variable warning

    def on_target_complete(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> None:
        """Called when target processing finishes."""
        _ = self, runner, target  # Suppress unused variable warning

    def on_benchmark_complete(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> None:
        """Called when entire benchmark finishes."""
        _ = self, runner, targets  # Suppress unused variable warning

    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        """Called when an error occurs."""
        _ = self, runner, target, error  # Suppress unused variable warning

    def on_analysis_complete(
        self,
        runner: "BenchmarkPipeline[Any, Any]",
        target: BenchmarkTarget | None,
        output: AnalysisOutput,
    ) -> None:
        """Called after analysis (visualization) completes for a target."""
        _ = self, runner, target, output  # Suppress unused variable warning


class BenchmarkCallbackManager(BenchmarkCallback):
    """Group of callbacks that can be used to aggregate multiple callbacks."""

    def __init__(self, callbacks: list[BenchmarkCallback] | None = None):
        """Initialize the callback manager.

        Args:
            callbacks: List of callbacks to manage. If None, starts with empty list.
        """
        self.callbacks = callbacks or []

    def add_callback(self, callback: BenchmarkCallback) -> None:
        """Add a new callback to the manager."""
        self.callbacks.append(callback)

    def add_callbacks(self, callbacks: list[BenchmarkCallback]) -> None:
        """Add multiple callbacks to the manager."""
        self.callbacks.extend(callbacks)

    @override
    def on_benchmark_start(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> bool:
        return all(callback.on_benchmark_start(runner=runner, targets=targets) for callback in self.callbacks)

    @override
    def on_target_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_target_start(runner=runner, target=target) for callback in self.callbacks)

    @override
    def on_backtest_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_backtest_start(runner=runner, target=target) for callback in self.callbacks)

    @override
    def on_backtest_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, predictions: VersionedTimeSeriesPart
    ) -> None:
        for callback in self.callbacks:
            callback.on_backtest_complete(runner=runner, target=target, predictions=predictions)

    @override
    def on_evaluation_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_evaluation_start(runner=runner, target=target) for callback in self.callbacks)

    @override
    def on_evaluation_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, report: EvaluationReport
    ) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_complete(runner=runner, target=target, report=report)

    @override
    def on_analysis_complete(
        self,
        runner: "BenchmarkPipeline[Any, Any]",
        target: BenchmarkTarget | None,
        output: AnalysisOutput,
    ) -> None:
        for callback in self.callbacks:
            callback.on_analysis_complete(runner=runner, target=target, output=output)

    @override
    def on_target_complete(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> None:
        for callback in self.callbacks:
            callback.on_target_complete(runner=runner, target=target)

    @override
    def on_benchmark_complete(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> None:
        for callback in self.callbacks:
            callback.on_benchmark_complete(runner=runner, targets=targets)

    @override
    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        for callback in self.callbacks:
            callback.on_error(runner=runner, target=target, error=error)


__all__ = ["BenchmarkCallback", "BenchmarkCallbackManager"]
