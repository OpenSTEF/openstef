# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Runs complete model comparison studies across multiple forecasting targets.

Comparing forecasting models properly requires testing them on many different forecasting
scenarios (equipment types, consumption/prosumption, solar/wind parks, regions, seasons).
This module automates the entire process: training models, running backtests, calculating
metrics, generating reports, and storing results for comparison.

The complete workflow:
    - Model training: Train different forecasting approaches on each target
    - Backtesting: Test all models under realistic conditions
    - Evaluation: Calculate performance metrics across different scenarios
    - Analysis: Generate comparison reports and visualizations
    - Storage: Save results for later analysis and sharing
"""

from openstef_beam.benchmarking.benchmark_comparison_pipeline import BenchmarkComparisonPipeline
from openstef_beam.benchmarking.benchmark_pipeline import (
    BenchmarkContext,
    BenchmarkPipeline,
    ForecasterFactory,
    read_evaluation_reports,
)
from openstef_beam.benchmarking.callbacks import BenchmarkCallback, BenchmarkCallbackManager, StrictExecutionCallback
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import (
    BenchmarkStorage,
    InMemoryBenchmarkStorage,
    LocalBenchmarkStorage,
    S3BenchmarkStorage,
)
from openstef_beam.benchmarking.target_provider import TargetProvider, TargetProviderConfig

__all__ = [
    "BenchmarkCallback",
    "BenchmarkCallbackManager",
    "BenchmarkComparisonPipeline",
    "BenchmarkContext",
    "BenchmarkPipeline",
    "BenchmarkStorage",
    "BenchmarkTarget",
    "ForecasterFactory",
    "InMemoryBenchmarkStorage",
    "LocalBenchmarkStorage",
    "S3BenchmarkStorage",
    "StrictExecutionCallback",
    "TargetProvider",
    "TargetProviderConfig",
    "read_evaluation_reports",
]
