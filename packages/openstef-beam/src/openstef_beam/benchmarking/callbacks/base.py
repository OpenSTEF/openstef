# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING, Any

from openstef_beam.analysis import AnalysisOutput
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.evaluation import EvaluationReport
from openstef_core.datasets import VersionedTimeSeriesDataset

if TYPE_CHECKING:
    from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline


class BenchmarkCallback:
    """Base class for benchmark execution callbacks."""

    def on_benchmark_start(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> bool:
        """Called when benchmark starts.

        Returns:
            bool: True if benchmark should start, False to skip.
        """
        _ = runner, targets  # Suppress unused variable warning
        return True

    def on_target_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called when processing a target begins."""
        _ = runner, target  # Suppress unused variable warning
        return True

    def on_backtest_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called before backtest execution.

        Returns:
            bool: True if backtesting should start, False to skip.
        """
        _ = runner, target  # Suppress unused variable warning
        return True

    def on_backtest_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, predictions: VersionedTimeSeriesDataset
    ) -> None:
        """Called after backtest completes."""
        _ = runner, target, predictions  # Suppress unused variable warning

    def on_evaluation_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        """Called before evaluation starts.

        Returns:
            bool: True if evaluation should start, False to skip.
        """
        _ = runner, target  # Suppress unused variable warning
        return True

    def on_evaluation_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, report: EvaluationReport
    ) -> None:
        """Called after evaluation completes."""
        _ = runner, target, report  # Suppress unused variable warning

    def on_target_complete(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> None:
        """Called when target processing finishes."""
        _ = runner, target  # Suppress unused variable warning

    def on_benchmark_complete(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> None:
        """Called when entire benchmark finishes."""
        _ = runner, targets  # Suppress unused variable warning

    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        """Called when an error occurs."""
        _ = runner, target, error  # Suppress unused variable warning

    def on_analysis_complete(
        self,
        runner: "BenchmarkPipeline[Any, Any]",
        target: BenchmarkTarget | None,
        output: AnalysisOutput,
    ) -> None:
        """Called after analysis (visualization) completes for a target."""
        _ = runner, target, output  # Suppress unused variable warning


class BenchmarkCallbackManager(BenchmarkCallback):
    """Group of callbacks that can be used to aggregate multiple callbacks."""

    def __init__(self, callbacks: list[BenchmarkCallback] | None = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: BenchmarkCallback) -> None:
        """Add a new callback to the manager."""
        self.callbacks.append(callback)

    def add_callbacks(self, callbacks: list[BenchmarkCallback]) -> None:
        """Add multiple callbacks to the manager."""
        self.callbacks.extend(callbacks)

    def on_benchmark_start(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> bool:
        return all(callback.on_benchmark_start(runner=runner, targets=targets) for callback in self.callbacks)

    def on_target_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_target_start(runner=runner, target=target) for callback in self.callbacks)

    def on_backtest_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_backtest_start(runner=runner, target=target) for callback in self.callbacks)

    def on_backtest_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, predictions: VersionedTimeSeriesDataset
    ) -> None:
        for callback in self.callbacks:
            callback.on_backtest_complete(runner=runner, target=target, predictions=predictions)

    def on_evaluation_start(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> bool:
        return all(callback.on_evaluation_start(runner=runner, target=target) for callback in self.callbacks)

    def on_evaluation_complete(
        self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, report: EvaluationReport
    ) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_complete(runner=runner, target=target, report=report)

    def on_analysis_complete(
        self,
        runner: "BenchmarkPipeline[Any, Any]",
        target: BenchmarkTarget | None,
        output: AnalysisOutput,
    ) -> None:
        for callback in self.callbacks:
            callback.on_analysis_complete(runner=runner, target=target, output=output)

    def on_target_complete(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget) -> None:
        for callback in self.callbacks:
            callback.on_target_complete(runner=runner, target=target)

    def on_benchmark_complete(self, runner: "BenchmarkPipeline[Any, Any]", targets: list[BenchmarkTarget]) -> None:
        for callback in self.callbacks:
            callback.on_benchmark_complete(runner=runner, targets=targets)

    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        for callback in self.callbacks:
            callback.on_error(runner=runner, target=target, error=error)


__all__ = ["BenchmarkCallback", "BenchmarkCallbackManager"]
