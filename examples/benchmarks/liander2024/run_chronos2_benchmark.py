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
# Backtest the **zero-shot** [Chronos-2](https://huggingface.co/amazon/chronos-2)
# foundation model on the
# [Liander 2024 STEF benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark),
# using the same backtesting harness as the XGBoost & GBLinear benchmark so the
# numbers are directly comparable.
#
# **What this does:**
#
# 1. Loads a local Chronos-2 ONNX checkpoint **once** and reuses it for every target
# 2. Runs day-by-day backtesting on a subset of the dataset (wind parks by default)
# 3. Produces probabilistic forecasts (7 quantiles) for a 3-day horizon
# 4. Saves results locally for comparison (see *Compare Results* notebook)
#
# ```{admonition} The model stays loaded across targets
# :class: tip
# Chronos-2 is zero-shot, so the workflow (and its loaded ONNX session) is built
# **once** and shared across every target — switching from one location to the next
# never reloads the model. This load-once pattern only holds when the benchmark
# runs **sequentially** (`N_PROCESSES = 1`): separate worker processes each have
# their own memory and cannot share a live ONNX session, so any parallel run would
# load one copy of the model *per worker*. For more throughput the right lever is
# batching series into a single backend call (planned), not multiprocessing.
# ```
#
# ```{warning}
# This benchmark loads a **local** ONNX export of Chronos-2 that is not published
# yet, so it is **not executed** during the docs build. To run it yourself, export
# the checkpoint with the `chronos-onnx-lab` script and run this notebook locally.
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
# Import the benchmarking harness, the foundation-model adapter, and configure logging.

# %%
import logging
from datetime import timedelta
from pathlib import Path
from typing import override

from huggingface_hub import snapshot_download
from pydantic import Field

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
from openstef_foundation_models.inference.providers import CoreMLProvider
from openstef_foundation_models.integrations.backtesting import (
    FoundationModelBacktestForecaster,
    create_foundation_model_backtest_forecaster,
)
from openstef_foundation_models.models.checkpoint import LocalCheckpoint
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    OnnxBackendConfig,
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
OUTPUT_PATH = Path("./benchmark_results_gpu")
BENCHMARK_RESULTS_PATH_CHRONOS2 = OUTPUT_PATH / "Chronos2"

# Path to the local Chronos-2 ONNX export (with its `.metadata.json` sidecar).
CHECKPOINT_PATH = Path(os.environ.get("CHRONOS2_ONNX_PATH", "chronos-onnx-lab/artifacts/chronos-2.onnx"))
CHECKPOINT_PATH = Path(os.environ.get("CHRONOS2_ONNX_PATH", "chronos-onnx-lab/artifacts/chronos-2_static.onnx"))

# Run sequentially so the loaded model is reused across every target (see the note
# at the top). A value > 1 would load one model copy per worker process.
N_PROCESSES = 1

# Which Liander2024 categories to benchmark. Start with wind parks; add more here,
# e.g. ["wind_park", "solar_park"]. Set to None to run every category.
BENCHMARK_FILTER: list[Liander2024Category] | None = ["wind_park"]

# Forecast 3 days ahead, producing 7 quantile bands (matches the XGBoost benchmark).
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# Smoke test: run a single target over a few days to verify the wiring end-to-end
# without committing to the full benchmark. Toggle with `OPENSTEF_SMOKE_TEST=1`.
SMOKE_TEST = os.environ.get("OPENSTEF_SMOKE_TEST", "false").lower() in {"1", "true", "yes"}
SMOKE_TEST = False
SMOKE_MAX_TARGETS = 1
SMOKE_BENCHMARK_DAYS = 14


# %% [markdown]
# ## Subset target provider
#
# A thin wrapper over the standard Liander2024 provider that can cap the number of
# targets and shorten each benchmark window — used only for the smoke test. With
# both limits unset it behaves exactly like the default provider.


# %%
class SubsetLiander2024TargetProvider(Liander2024TargetProvider):
    """Liander2024 provider that optionally limits the targets and benchmark window.

    Used to run a quick smoke test over a single target and a few days instead of
    the whole dataset. With both limits set to ``None`` it is a drop-in for the
    default provider.
    """

    max_targets: int | None = Field(default=None, description="Keep at most this many targets, in dataset order.")
    max_benchmark_days: int | None = Field(
        default=None,
        description="Clamp each target's benchmark window to this many days from its start.",
    )

    @override
    def get_targets(self, filter_args: list[Liander2024Category] | None = None) -> list[BenchmarkTarget]:
        targets = super().get_targets(filter_args)
        if self.max_benchmark_days is not None:
            for target in targets:
                target.benchmark_end = min(
                    target.benchmark_end,
                    target.benchmark_start + timedelta(days=self.max_benchmark_days),
                )
        if self.max_targets is not None:
            targets = targets[: self.max_targets]
        return targets


# %% [markdown]
# ## Locate the checkpoint
#
# A *checkpoint* is the ONNX weights file plus a small `CheckpointMetadata` sidecar
# (`<weights>.metadata.json`) describing the model's tensor names, native quantile
# grid, and context/horizon sizing. The sidecar is discovered automatically next to
# the weights.

# %%
if not CHECKPOINT_PATH.is_file():
    msg = (
        f"Chronos-2 ONNX artifact not found at {CHECKPOINT_PATH}. "
        "Export it with the chronos-onnx-lab script, or set CHRONOS2_ONNX_PATH."
    )
    raise FileNotFoundError(msg)

checkpoint = LocalCheckpoint(path=CHECKPOINT_PATH)

# %% [markdown]
# ## Build the workflow once
#
# `create_forecasting_workflow` resolves the checkpoint, builds the ONNX Runtime
# session **once**, and wraps a `Chronos2Forecaster` in a workflow that selects the
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
        backend=OnnxBackendConfig(providers=[CoreMLProvider()], strict_providers=True),  # For Mac GPU
    )
)


# %% [markdown]
# ## Forecaster factory
#
# The benchmark calls this factory once per target. It wraps the **shared** workflow
# in a backtest adapter without rebuilding it, so the loaded ONNX session is reused
# for every location.


# %%
def chronos2_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> FoundationModelBacktestForecaster:
    """Return a backtest forecaster wrapping the shared, pre-built workflow."""
    return create_foundation_model_backtest_forecaster(
        workflow=workflow,
        predict_length=FORECAST_HORIZONS[0].value,
    )


# %% [markdown]
# ## Run the benchmark
#
# Downloads the dataset (cached after the first run), then backtests Chronos-2 on the
# selected subset. `StrictExecutionCallback` raises on any target failure (remove it
# to skip failing targets silently).

# %%
if __name__ == "__main__":
    data_dir = Path(snapshot_download(repo_id="OpenSTEF/liander2024-stef-benchmark", repo_type="dataset"))

    target_provider = SubsetLiander2024TargetProvider(
        data_dir=data_dir,
        max_targets=SMOKE_MAX_TARGETS if SMOKE_TEST else None,
        max_benchmark_days=SMOKE_BENCHMARK_DAYS if SMOKE_TEST else None,
    )

    create_liander2024_benchmark_runner(
        data_dir=data_dir,
        storage=LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_CHRONOS2),
        target_provider=target_provider,
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=chronos2_factory,
        run_name="chronos2",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
