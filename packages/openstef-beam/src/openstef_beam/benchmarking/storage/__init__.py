# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Storage backends for benchmark results and analysis outputs."""

from openstef_beam.benchmarking.storage.base import BenchmarkStorage, InMemoryBenchmarkStorage
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_beam.benchmarking.storage.s3_storage import S3BenchmarkStorage

__all__ = [
    "BenchmarkStorage",
    "InMemoryBenchmarkStorage",
    "LocalBenchmarkStorage",
    "S3BenchmarkStorage",
]
