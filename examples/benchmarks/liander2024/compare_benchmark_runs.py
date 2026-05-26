# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare Benchmark Runs
#
# Generate side-by-side comparison plots from multiple benchmark runs on the
# Liander 2024 dataset.
#
# **Prerequisites:** Run at least two models first (e.g. XGBoost + GBLinear via
# the *XGBoost & GBLinear* notebook).
#
# **What this does:**
#
# 1. Loads results from multiple model runs (each stored in its own directory)
# 2. Computes metrics across all targets using
#    [`BenchmarkComparisonPipeline`](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.BenchmarkComparisonPipeline.html)
# 3. Produces comparison visualizations (boxplots, ranking tables, per-target breakdowns)

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %% [markdown]
# ## Setup
#
# Point at the result directories from your benchmark runs.

# %%
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

# %% [markdown]
# ## Load run results
#
# Each run is identified by a name and backed by a `LocalBenchmarkStorage` that
# points at the directory where that model's results were saved.

# %%
check_dirs = [
    BENCHMARK_DIR_GBLINEAR,
    BENCHMARK_DIR_XGBOOST,
]
for dir_path in check_dirs:
    if not dir_path.exists():
        msg = f"Benchmark directory not found: {dir_path}. Make sure to run the benchmarks first."
        raise FileNotFoundError(msg)

run_storages: dict[RunName, BenchmarkStorage] = {
    "gblinear": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_GBLINEAR),
    "xgboost": LocalBenchmarkStorage(base_path=BENCHMARK_DIR_XGBOOST),
}

# %% [markdown]
# ## Run comparison
#
# The pipeline loads predictions from each run, re-evaluates them with the
# Liander 2024 analysis config, and produces comparison visualizations.

# %%
target_provider = create_liander2024_benchmark_runner(
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
).target_provider

comparison_pipeline = BenchmarkComparisonPipeline(
    analysis_config=LIANDER2024_ANALYSIS_CONFIG,
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
    target_provider=target_provider,
)
comparison_pipeline.run(run_data=run_storages)
