# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare Benchmark Runs
#
# Generate side-by-side comparison plots from multiple benchmark runs.
# Uses [`BenchmarkComparisonPipeline`](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.BenchmarkComparisonPipeline.html)
# to produce global, per-group, and per-target HTML visualizations.
#
# **Prerequisites:** Run at least two models first (e.g. via `run_liander2024_benchmark.py`).

# %% tags=["remove-cell"]
"""Compare benchmark results from different runs on the Liander 2024 dataset.

Usage:
    1. First run at least two models with run_liander2024_benchmark.py
       (e.g. ExampleBaseline and GBLinear).
    2. Then run this script to generate side-by-side comparison plots.

Output is saved to ./benchmark_results_comparison/liander2024/.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %% [markdown]
# ## Setup
#
# Point at the result directories from your benchmark runs.

# %%

from pathlib import Path
from typing import cast

from openstef_beam.analysis.models import RunName
from openstef_beam.benchmarking import BenchmarkComparisonPipeline, LocalBenchmarkStorage
from openstef_beam.benchmarking.benchmarks import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.benchmarks.liander2024 import LIANDER2024_ANALYSIS_CONFIG
from openstef_beam.benchmarking.storage import BenchmarkStorage

# One storage per run — keys are human-readable labels shown in comparison plots.
run_storages: dict[RunName, BenchmarkStorage] = {
    "ExampleBaseline": LocalBenchmarkStorage(base_path=Path("./benchmark_results/ExampleBaseline")),
    "GBLinear": LocalBenchmarkStorage(base_path=Path("./benchmark_results/GBLinear")),
}

# Check that results exist.
for name, storage in run_storages.items():
    base_path = cast(LocalBenchmarkStorage, storage).base_path
    if not base_path.exists():
        msg = f"Benchmark directory not found for '{name}': {base_path}. Run the benchmarks first."
        raise FileNotFoundError(msg)

# %% [markdown]
# ## Run comparison
#
# The pipeline loads predictions from each run, re-evaluates them, and produces
# comparison visualizations.

# %%
# Reuse the Liander 2024 target provider.
OUTPUT_PATH = Path("./benchmark_results_comparison/liander2024")
target_provider = create_liander2024_benchmark_runner(
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
).target_provider

# Run the comparison — generates global, group, and per-target HTML plots.
comparison = BenchmarkComparisonPipeline(
    analysis_config=LIANDER2024_ANALYSIS_CONFIG,
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
    target_provider=target_provider,
)
comparison.run(run_data=run_storages)
