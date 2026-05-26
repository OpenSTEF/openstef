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
# # Run Liander 2024 Benchmark (Custom Forecaster)
#
# Entry point: test your custom forecaster on the built-in
# [Liander 2024 dataset](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark)
# (auto-downloaded from HuggingFace).
#
# Uses [`create_liander2024_benchmark_runner()`](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.benchmarks.liander2024.html)
# which pre-configures backtest settings, evaluation windows, metrics, and target definitions.
#
# **See also:** [Custom Forecaster template](./custom_forecaster.ipynb) — define your model here.

# %% tags=["remove-cell"]
"""Example: run the built-in Liander 2024 benchmark with a custom baseline and GBLinear.

Uses create_liander2024_benchmark_runner() which pre-configures everything:
backtest settings, evaluation windows, metrics, analysis plots, and target
definitions. Data is auto-downloaded from HuggingFace.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# %% [markdown]
# ## Setup

# %%

import logging
import multiprocessing
from pathlib import Path

from examples.benchmarks.custom.custom_forecaster import ExampleBenchmarkForecaster
from openstef_beam.benchmarking import BenchmarkContext, BenchmarkTarget, LocalBenchmarkStorage, StrictExecutionCallback
from openstef_beam.benchmarking.baselines.openstef4 import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## Configuration
#
# Define output paths, quantiles, and the GBLinear model config.

# %%
OUTPUT_PATH = Path("./benchmark_results")
N_PROCESSES = int(os.environ.get("OPENSTEF_N_PROCESSES", str(multiprocessing.cpu_count())))

# Optional: filter to specific target categories (None = run all)
BENCHMARK_FILTER: list[Liander2024Category] | None = None

# Quantiles define the probabilistic forecast bands
# Q(0.05) = 5th percentile, Q(0.5) = median, Q(0.95) = 95th percentile
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# --- GBLinear model config ---
# Map column names in your data to what OpenSTEF expects
gblinear_config = ForecastingWorkflowConfig(
    model_id="liander_benchmark_",
    run_name=None,
    model="gblinear",
    horizons=[LeadTime.from_string("P3D")],
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=True,
    radiation_column="shortwave_radiation",
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    rolling_aggregate_features=["mean", "median", "max", "min"],
)


# %% [markdown]
# ## Forecaster factory
#
# The benchmark pipeline calls this function once per target. Return your custom forecaster.

# %%


def example_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> ExampleBenchmarkForecaster:
    """Create the example baseline forecaster.

    Returns:
        Configured ExampleBenchmarkForecaster instance.
    """
    return ExampleBenchmarkForecaster(predict_quantiles=PREDICTION_QUANTILES)


# %% [markdown]
# ## Run benchmark
#
# Run the custom baseline and GBLinear on all Liander 2024 targets.
# Results are saved to `./benchmark_results/<model_name>/`.

# %%
if __name__ == "__main__":
    # 1. Run custom baseline on Liander 2024
    # create_liander2024_benchmark_runner() sets up everything: data download, configs, metrics
    # LocalBenchmarkStorage writes results as parquet files to disk
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "ExampleBaseline"),
        callbacks=[StrictExecutionCallback()],  # Fail fast on errors
    ).run(
        forecaster_factory=example_factory,  # Your model factory (called per target)
        run_name="example_baseline",  # Label for this run
        n_processes=N_PROCESSES,  # Parallel targets
        filter_args=BENCHMARK_FILTER,  # None = all categories
    )

    # 2. Run GBLinear on Liander 2024
    # create_openstef4_preset_backtest_forecaster returns a factory that wraps OpenSTEF models
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / "GBLinear"),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=gblinear_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name="gblinear",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
