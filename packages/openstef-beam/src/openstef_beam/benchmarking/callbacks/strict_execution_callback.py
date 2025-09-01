# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Strict execution callback for benchmark pipelines.

Provides a callback that enforces strict error handling, causing the benchmark
pipeline to immediately terminate on any error rather than continuing with
remaining targets. Useful for development and validation workflows where
all targets must succeed.
"""

from typing import TYPE_CHECKING, Any, override

from openstef_beam.benchmarking.callbacks.base import BenchmarkCallback
from openstef_beam.benchmarking.models import BenchmarkTarget

if TYPE_CHECKING:
    from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline


class StrictExecutionCallback(BenchmarkCallback):
    """Callback to ensure strict benchmark execution with immediate error termination.

    This callback enforces strict error handling by re-raising any errors that occur
    during benchmark execution, causing the pipeline to immediately terminate rather
    than continuing with remaining targets.

    Use this callback during development, testing, or validation workflows where
    you need to ensure all targets complete successfully before proceeding.
    """

    @override
    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        """Re-raise any error to immediately terminate benchmark execution."""
        raise error


__all__ = ["StrictExecutionCallback"]
