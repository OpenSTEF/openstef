# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING, Any, override

from openstef_beam.benchmarking.callbacks.base import BenchmarkCallback
from openstef_beam.benchmarking.models import BenchmarkTarget

if TYPE_CHECKING:
    from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline


class StrictExecutionCallback(BenchmarkCallback):
    """Callback to ensure that the execution of the benchmark is strict.
    This means that the benchmark will not continue until the current step is completed.
    """

    @override
    def on_error(self, runner: "BenchmarkPipeline[Any, Any]", target: BenchmarkTarget, error: Exception) -> None:
        raise error


__all__ = ["StrictExecutionCallback"]
