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
# # XGBoost & GBLinear Benchmark
#
# Run two models head-to-head on the
# [Liander 2024 STEF benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark)
# — an open dataset of Dutch energy grid measurements.
#
# **What this does:**
#
# 1. Downloads the Liander 2024 dataset from HuggingFace (automatic)
# 2. Trains XGBoost and GBLinear on each target using day-by-day backtesting
# 3. Produces probabilistic forecasts (7 quantiles) for a 3-day horizon
# 4. Saves results locally for comparison (see *Compare Results* notebook)
#
# **No code changes needed** — just run it. To benchmark your own model instead,
# see [Implement a Custom Forecaster](../custom/custom_forecaster.ipynb).
#
# ```{admonition} Runtime
# Expect 30-60 min on a laptop (uses all CPU cores).
# Set `OPENSTEF_N_PROCESSES=1` for easier debugging.
# ```

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# %% [markdown]
# ## Setup
#
# Import BEAM components and configure logging.

# %%
import logging
import multiprocessing
from pathlib import Path

from openstef_beam.benchmarking.baselines.openstef4 import (
    create_openstef4_preset_backtest_forecaster,
)
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.presets import (
    ForecastingWorkflowConfig,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## Configuration
#
# Define output paths, forecast horizons, and quantiles.
# The benchmark runs each model in parallel across all targets in the dataset.

# %%
OUTPUT_PATH = Path("./benchmark_results")

BENCHMARK_RESULTS_PATH_XGBOOST = OUTPUT_PATH / "XGBoost"
BENCHMARK_RESULTS_PATH_GBLINEAR = OUTPUT_PATH / "GBLinear"
N_PROCESSES = int(os.environ.get("OPENSTEF_N_PROCESSES", str(multiprocessing.cpu_count())))

# Forecast 3 days ahead, producing 7 quantile bands
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]
PREDICTION_QUANTILES = [
    Q(0.05),
    Q(0.1),
    Q(0.3),
    Q(0.5),
    Q(0.7),
    Q(0.9),
    Q(0.95),
]

# Set to a list of categories to run only a subset (e.g. [Liander2024Category.SOLAR])
BENCHMARK_FILTER: list[Liander2024Category] | None = None

# %% [markdown]
# ## Model configuration
#
# `ForecastingWorkflowConfig` defines how OpenSTEF trains and predicts.
# We create a shared base config and derive model-specific variants with `model_copy()`.
#
# Set `OPENSTEF_MLFLOW_STORAGE=true` to log experiment artifacts to MLflow.

# %%
USE_MLFLOW_STORAGE = os.environ.get("OPENSTEF_MLFLOW_STORAGE", "false").lower() == "true"

if USE_MLFLOW_STORAGE:
    storage = MLFlowStorage(
        tracking_uri=str(OUTPUT_PATH / "mlflow_artifacts"),
        local_artifacts_path=OUTPUT_PATH / "mlflow_tracking_artifacts",
    )
else:
    storage = None

common_config = ForecastingWorkflowConfig(
    model_id="common_model_",
    run_name=None,
    model="flatliner",
    horizons=FORECAST_HORIZONS,
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=True,
    mlflow_storage=storage,
    radiation_column="shortwave_radiation",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
)

xgboost_config = common_config.model_copy(update={"model": "xgboost"})
gblinear_config = common_config.model_copy(update={"model": "gblinear"})

# %% [markdown]
# ## Run the benchmark
#
# Each model gets its own output directory. `StrictExecutionCallback` raises on
# any target failure (remove it to skip failing targets silently).

# %%
if __name__ == "__main__":
    # --- XGBoost ---
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_XGBOOST),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=xgboost_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name="xgboost",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )

    # --- GBLinear ---
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_GBLINEAR),
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
