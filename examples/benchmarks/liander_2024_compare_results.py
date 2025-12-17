"""Example for comparing benchmark results from different runs on the Liander 2024 dataset."""
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from openstef_beam.analysis.models import RunName
from openstef_beam.benchmarking import BenchmarkComparisonPipeline, LocalBenchmarkStorage
from openstef_beam.benchmarking.benchmarks import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.benchmarks.liander2024 import LIANDER2024_ANALYSIS_CONFIG
from openstef_beam.benchmarking.storage import BenchmarkStorage

BASE_DIR = Path()

OUTPUT_PATH = BASE_DIR / "./benchmark_results_comparison"

BENCHMARK_DIR_GBLINEAR = BASE_DIR / "benchmark_results" / "GBLinear"
BENCHMARK_DIR_XGBOOST = BASE_DIR / "benchmark_results" / "XGBoost"
BENCHMARK_DIR_GBLINEAR_OPENSTEF3 = BASE_DIR / "benchmark_results_openstef3" / "gblinear"
BENCHMARK_DIR_XGBOOST_OPENSTEF3 = BASE_DIR / "benchmark_results_openstef3" / "xgboost"

check_dirs = [
    BENCHMARK_DIR_GBLINEAR,
    BENCHMARK_DIR_XGBOOST,
    BENCHMARK_DIR_GBLINEAR_OPENSTEF3,
    BENCHMARK_DIR_XGBOOST_OPENSTEF3,
]
for dir_path in check_dirs:
    if not dir_path.exists():
        msg = f"Benchmark directory not found: {dir_path}. Make sure to run the benchmarks first."
        raise FileNotFoundError(msg)

run_storages: dict[RunName, BenchmarkStorage] = {
    "gblinear": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_GBLINEAR),
    "gblinear_openstef3": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_GBLINEAR_OPENSTEF3),
    "xgboost": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_XGBOOST),
    "xgboost_openstef3": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_XGBOOST_OPENSTEF3),
}

target_provider = create_liander2024_benchmark_runner(
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
).target_provider

comparison_pipeline = BenchmarkComparisonPipeline(
    analysis_config=LIANDER2024_ANALYSIS_CONFIG,
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
    target_provider=target_provider,
)
comparison_pipeline.run(run_data=run_storages)
