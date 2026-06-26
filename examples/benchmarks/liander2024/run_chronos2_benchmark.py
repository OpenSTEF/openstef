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
# # Chronos-2 Foundation-Model Benchmark
#
# Backtest the zero-shot [Chronos-2](https://huggingface.co/amazon/chronos-2)
# foundation model on the
# [Liander 2024 STEF benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark),
# using the same backtesting harness as the XGBoost and GBLinear benchmarks so the
# numbers are directly comparable.
#
# What this does:
#
# 1. Loads a local Chronos-2 ONNX checkpoint once and reuses it for every target
# 2. Runs day-by-day backtesting on a subset of the dataset (wind parks by default)
# 3. Produces probabilistic forecasts (7 quantiles) for a 3-day horizon
# 4. Stacks consecutive forecast windows into batched backend calls (`BATCH_SIZE`)
# 5. Saves results locally for comparison (see the *Compare Results* notebook)
#
# **Note**:
#
# Chronos-2 is zero-shot, so the workflow and its loaded ONNX session are built
# once and shared across every target. This only holds when the benchmark runs
# sequentially (`N_PROCESSES = 1`): separate worker processes cannot share a live
# ONNX session and would load one copy of the model each.


# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# %% [markdown]
# ## Setup
#
# Import the relevant components, and configure logging.

# %%
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext
from openstef_beam.benchmarking.benchmarks.liander2024 import (
    Liander2024Category,
    Liander2024TargetProvider,
    create_liander2024_benchmark_runner,
)
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster
from openstef_foundation_models.models import CheckpointVariant, Chronos2
from openstef_foundation_models.models.checkpoint import CheckpointRef, LocalCheckpoint
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.utils.feature_selection import Include

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## Configuration
#
# Pick which categories to benchmark, the forecast horizon, and the quantiles.
# Wind parks are the default subset; add more categories to widen the run.

# %%
OUTPUT_PATH = Path("./benchmark_results")

# Number of forecast windows Chronos-2 stacks into a single backend call. The
# backtest pipeline groups consecutive windows per target into batches of this
# size; 1 forecasts one window at a time. Larger batches mean fewer, heavier
# ONNX calls. Results are written per batch size so runs do not overwrite.
# Note that more batching is not always better, it depends on your system.
# For CPU inference you should usually set this to 1.
BATCH_SIZE = 16

BENCHMARK_RESULTS_PATH_CHRONOS2 = OUTPUT_PATH / "Chronos2"

# Use the published Chronos-2 checkpoint from the HuggingFace Hub by default. Set
# CHRONOS2_ONNX_PATH to benchmark a local ONNX export (with its `.metadata.json`) instead.
LOCAL_CHECKPOINT_PATH = os.environ.get("CHRONOS2_ONNX_PATH")

# Run sequentially so the loaded model is reused across every target (see the note
# at the top). A value > 1 would load one model copy per worker process.
N_PROCESSES = 1

# Which Liander2024 categories to benchmark. Start with wind parks; add more here,
# e.g. ["wind_park", "solar_park"]. Set to None to run every category.
BENCHMARK_FILTER: list[Liander2024Category] | None = ["wind_park"]

# Forecast 3 days ahead, producing 7 quantile bands.
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# %% [markdown]
# ## Select the checkpoint
#
# By default we pull the published `Chronos2.BASE` checkpoint from the HuggingFace Hub
# (`recommended()` picks the variant best suited to this host). Its metadata, which
# describes the tensor names, native quantile grid, and context/horizon sizing, is
# downloaded alongside the weights. Set `CHRONOS2_ONNX_PATH` to benchmark a local
# export instead; its `.metadata.json` is discovered next to the weights.

# %%
checkpoint: CheckpointRef = (
    LocalCheckpoint(path=Path(LOCAL_CHECKPOINT_PATH))
    if LOCAL_CHECKPOINT_PATH
    else Chronos2.BASE.checkpoint(CheckpointVariant.recommended())
)

# %% [markdown]
# ## Build the workflow once
#
# `create_forecasting_workflow` resolves the checkpoint, builds the ONNX Runtime
# session once, and wraps a `Chronos2Forecaster` in a workflow that selects the
# target plus weather covariates and sorts quantiles. This single workflow instance
# is shared across every target below.

# %%
workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=checkpoint,
        quantiles=PREDICTION_QUANTILES,
        horizons=FORECAST_HORIZONS,
        target_column="load",
        # Keep the target plus the known-future weather covariates; every kept
        # non-target column is forwarded to Chronos-2 as a covariate.
        selected_features=Include(
            "load",
            "shortwave_radiation",
            "wind_speed_80m",
            "temperature_2m",
        ),
        # No `backend` override: the default provider policy reads the checkpoint
        # metadata (precision, static shapes) and the host to pick a performant
        # chain automatically. Pass an explicit `backend=OnnxBackendConfig(...)`
        # only to force a specific chain.
    )
)


# %% [markdown]
# ## Forecaster factory
#
# The benchmark calls this factory once per target. It wraps the shared workflow in
# a backtest adapter without rebuilding it, so the loaded ONNX session is reused for
# every location. Passing `batch_size` lets the pipeline forecast several windows in
# a single backend call instead of one at a time.


# %%
def chronos2_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> FoundationModelBacktestForecaster:
    """Return a backtest forecaster wrapping the shared, pre-built workflow."""
    return FoundationModelBacktestForecaster.from_workflow(workflow, batch_size=BATCH_SIZE)


# %% [markdown]
# ## Run the benchmark
#
# Downloads the dataset (cached after the first run), then backtests Chronos-2 on the
# selected subset. `StrictExecutionCallback` raises on any target failure (remove it
# to skip failing targets silently).

# %%
if __name__ == "__main__":
    data_dir = Path(snapshot_download(repo_id="OpenSTEF/liander2024-stef-benchmark", repo_type="dataset"))

    create_liander2024_benchmark_runner(
        data_dir=data_dir,
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_CHRONOS2),
        target_provider=Liander2024TargetProvider(data_dir=data_dir),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=chronos2_factory,
        run_name="chronos2",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
