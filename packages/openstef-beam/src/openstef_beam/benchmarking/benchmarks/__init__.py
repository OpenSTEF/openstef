# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Built in benchmarks to run with OpenSTEF BEAM.

This module provides predefined benchmark runners that set up common benchmarking scenarios.
"""

from .liander2024 import create_liander2024_benchmark_runner

__all__ = [
    "create_liander2024_benchmark_runner",
]
