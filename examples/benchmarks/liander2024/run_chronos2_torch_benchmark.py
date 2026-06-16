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
# # Chronos-2 Foundation-Model Benchmark (PyTorch backend)
#
# Backtest the **zero-shot** [Chronos-2](https://huggingface.co/amazon/chronos-2)
# foundation model on the
# [Liander 2024 STEF benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark)
# using the **original PyTorch weights** instead of the ONNX export. This is the
# Torch counterpart of `run_chronos2_benchmark.py`: same harness, same subset, same
# horizon and quantiles — only the inference backend differs. Run both and compare
# the result folders to verify the ONNX export matches PyTorch end-to-end.
#
# **What this does:**
#
# 1. Loads the Chronos-2 **PyTorch** module **once** and reuses it for every target
# 2. Runs day-by-day backtesting on a subset of the dataset (wind parks by default)
# 3. Produces probabilistic forecasts (7 quantiles) for a 3-day horizon
# 4. Saves results locally for comparison against the ONNX run
#
# ```{admonition} The model stays loaded across targets
# :class: tip
# Chronos-2 is zero-shot, so the workflow (and its loaded Torch module) is built
# **once** and shared across every target — switching from one location to the next
# never reloads the model. This load-once pattern only holds when the benchmark
# runs **sequentially** (`N_PROCESSES = 1`): separate worker processes each have
# their own memory and cannot share a live Torch module, so any parallel run would
# load one copy of the model *per worker*.
# ```
#
# ```{warning}
# This benchmark needs the optional `[torch]` extra (PyTorch + chronos-forecasting)
# and downloads the Chronos-2 weights from HuggingFace on first run, so it is **not
# executed** during the docs build. It still reads the ONNX checkpoint's metadata
# sidecar (quantile grid, context/horizon sizing) so the two backends stay aligned.
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
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.integrations.backtesting import (
    FoundationModelBacktestForecaster,
    create_foundation_model_backtest_forecaster,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata, LocalCheckpoint
from openstef_foundation_models.models.forecasting.chronos2_forecaster import Chronos2Forecaster
from openstef_models.models import ForecastingModel
from openstef_models.transforms.general import Selector
from openstef_models.transforms.postprocessing import QuantileSorter
from openstef_models.utils.feature_selection import Include
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## Configuration
#
# Pick which categories to benchmark, the forecast horizon, and the quantiles.
# Wind parks are the default subset; add more categories to widen the run.

# %%
OUTPUT_PATH = Path("./benchmark_results_torch")
BENCHMARK_RESULTS_PATH_CHRONOS2 = OUTPUT_PATH / "Chronos2Torch"

# Path to the local Chronos-2 ONNX export. Only its `.metadata.json` sidecar is
# used here (native quantile grid + context/horizon sizing); the Torch weights are
# downloaded from HuggingFace. Keeping the same sidecar guarantees both backends
# run with identical sizing so their results are comparable.
CHECKPOINT_PATH = Path(os.environ.get("CHRONOS2_ONNX_PATH", "chronos-onnx-lab/artifacts/chronos-2.onnx"))

# HuggingFace model id for the original Chronos-2 Torch weights.
TORCH_MODEL_ID = os.environ.get("CHRONOS2_TORCH_ID", "amazon/chronos-2")

# Torch device: "cpu", "cuda", or "mps" (Apple). CPU is the safe default.
TORCH_DEVICE = os.environ.get("CHRONOS2_TORCH_DEVICE", "cpu")

# Run sequentially so the loaded model is reused across every target (see the note
# at the top). A value > 1 would load one model copy per worker process.
N_PROCESSES = 1

# Which Liander2024 categories to benchmark. Start with wind parks; add more here,
# e.g. ["wind_park", "solar_park"]. Set to None to run every category.
BENCHMARK_FILTER: list[Liander2024Category] | None = ["wind_park"]

# Forecast 3 days ahead, producing 7 quantile bands (matches the ONNX benchmark).
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# Smoke test: run a single target over a few days to verify the wiring end-to-end
# without committing to the full benchmark. Toggle with `OPENSTEF_SMOKE_TEST=1`.
SMOKE_TEST = os.environ.get("OPENSTEF_SMOKE_TEST", "false").lower() in {"1", "true", "yes"}
SMOKE_TEST = True
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
# ## Load the Chronos-2 Torch module
#
# `TorchBackend` runs any module that follows the named-tensor contract
# (`context`, `group_ids`, `attention_mask`, `future_covariates`,
# `future_covariates_mask` → quantile tensor). The raw `Chronos2Model.forward`
# uses different kwarg names and needs `num_output_patches`, so we wrap it in a
# small adapter — this mirrors `Chronos2OnnxModule` in the export lab so the Torch
# and ONNX backends receive identical inputs.


# %%
def build_torch_backend(metadata: CheckpointMetadata, model_id: str, device: str):  # noqa: ANN201
    """Load the Chronos-2 Torch module and wrap it in a backend-compatible adapter.

    Args:
        metadata: Checkpoint metadata (drives the frozen horizon via ``horizon_patches``).
        model_id: HuggingFace model id to load the Torch weights from.
        device: Torch device to run on (``"cpu"``, ``"cuda"``, or ``"mps"``).

    Returns:
        A ``TorchBackend`` wrapping the original Chronos-2 module.

    Raises:
        ImportError: When the ``[torch]`` extra (torch + chronos-forecasting) is missing.
    """
    try:
        import torch  # noqa: PLC0415
        from chronos import Chronos2Pipeline  # noqa: PLC0415

        from openstef_foundation_models.inference.torch_backend import TorchBackend  # noqa: PLC0415
    except ImportError as exc:
        msg = "This benchmark needs the [torch] extra: install torch + chronos-forecasting."
        raise ImportError(msg) from exc

    num_output_patches = metadata.horizon_patches

    class _Chronos2TorchAdapter(torch.nn.Module):
        """Flatten Chronos-2's interface to the forecaster's named-tensor contract."""

        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            context: torch.Tensor,
            group_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            future_covariates: torch.Tensor,
            future_covariates_mask: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.model(
                context=context,
                group_ids=group_ids,
                context_mask=attention_mask,
                future_covariates=future_covariates,
                future_covariates_mask=future_covariates_mask,
                num_output_patches=num_output_patches,
            )
            return outputs.quantile_preds

    logging.getLogger(__name__).info("Loading Torch Chronos-2 '%s' on %s", model_id, device)
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    return TorchBackend(metadata=metadata, module=_Chronos2TorchAdapter(pipeline.model), device=device)


# %% [markdown]
# ## Build the workflow once
#
# Resolve the checkpoint metadata, load the Torch backend **once**, and wrap a
# `Chronos2Forecaster` in a `CustomForecastingWorkflow` that selects the target
# plus weather covariates and sorts quantiles. This is what
# `create_forecasting_workflow` does for the ONNX backend; we build it by hand here
# because the preset is ONNX-only. This single workflow instance is shared across
# every target below.

# %%
if not CHECKPOINT_PATH.is_file():
    msg = (
        f"Chronos-2 metadata sidecar not found next to {CHECKPOINT_PATH}. "
        "Export it with the chronos-onnx-lab script, or set CHRONOS2_ONNX_PATH."
    )
    raise FileNotFoundError(msg)

resolved_checkpoint = LocalCheckpoint(path=CHECKPOINT_PATH).resolve()

SELECTED_FEATURES = Include("load", "shortwave_radiation", "wind_speed_80m", "temperature_2m")

torch_backend = build_torch_backend(resolved_checkpoint.metadata, TORCH_MODEL_ID, TORCH_DEVICE)

workflow = CustomForecastingWorkflow(
    model=ForecastingModel(
        preprocessing=TransformPipeline(transforms=[Selector(selection=SELECTED_FEATURES)]),
        forecaster=Chronos2Forecaster(
            backend=torch_backend,
            quantiles=PREDICTION_QUANTILES,
            horizons=FORECAST_HORIZONS,
        ),
        postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
        target_column="load",
    ),
    model_id="chronos2-torch",
)


# %% [markdown]
# ## Forecaster factory
#
# The benchmark calls this factory once per target. It wraps the **shared** workflow
# in a backtest adapter without rebuilding it, so the loaded Torch module is reused
# for every location.


# %%
def chronos2_torch_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> FoundationModelBacktestForecaster:
    """Return a backtest forecaster wrapping the shared, pre-built Torch workflow."""
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
        forecaster_factory=chronos2_torch_factory,
        run_name="chronos2-torch",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
        skip_analysis=False,
    )
