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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


# %% [markdown]
# # Foundation Model Forecasting
#
# This guide covers the practical settings for running `openstef-foundation-models`:
# installing an ONNX runtime, choosing a checkpoint, picking the execution provider for your
# hardware, and backtesting with openstef-beam. It uses Chronos-2, the first model family in
# the package; the same steps apply to families added later.
#
# For an end-to-end forecast with a plot, see the
# {doc}`Foundation Model Forecasting Quickstart </user_guide/getting_started/foundation_model_forecasting_quickstart>`.
# For what a foundation model is and when to use one, see the
# {ref}`Foundation Models concept page <concept_foundation_models>`.
#
# Every code cell below runs during the docs build against the current API, so each output is
# the live result. The model runs on CPU with the small Chronos-2 checkpoint, which keeps the
# guide fast and lets it execute on any machine, with or without a GPU.

# %% tags=["remove-cell"]
import warnings

warnings.filterwarnings("ignore")

from openstef_core.testing import setup_notebook_logging

logger = setup_notebook_logging(
    __name__,
    suppress=("huggingface_hub", "fsspec", "filelock", "openstef_core.datasets"),
)


# %% [markdown]
# ## Install one ONNX runtime
#
# `[cpu]` and `[gpu]` are mutually exclusive: `onnxruntime` and `onnxruntime-gpu` collide in
# the same environment, so pick one. uv enforces the choice through conflicting extras; pip
# does not, so choose it yourself.
#
# ```bash
# # CPU (what the meta-package installs by default)
# pip install "openstef-foundation-models[cpu]"
#
# # NVIDIA CUDA GPU: installs onnxruntime-gpu plus the pinned NVIDIA CUDA 12 / cuDNN 9
# # wheels, so no system CUDA install is required.
# pip install "openstef-foundation-models[gpu]"
#
# # Add TensorRT (only if you pin TensorRTProvider): the CUDA 12 runtime on top of [gpu].
# pip install "openstef-foundation-models[gpu]" tensorrt-cu12
# ```
#
# `[cpu]` installs only `onnxruntime`. `[gpu]` is heavier: `onnxruntime-gpu` carries the CUDA
# execution-provider plugin but not the CUDA runtime it loads at session creation, so the
# extra also pulls the matching NVIDIA CUDA 12 and cuDNN 9 wheels and no system CUDA install
# is required:
#
# - `onnxruntime-gpu` (the CUDA execution provider)
# - `nvidia-cuda-runtime-cu12`
# - `nvidia-cublas-cu12`
# - `nvidia-cufft-cu12`
# - `nvidia-curand-cu12`
# - `nvidia-cudnn-cu12`
#
# These are Linux and Windows x86-64 wheels. Apple Silicon and AMD GPUs use `[cpu]` (see the
# hardware table below), so they never install the CUDA wheels.
#
# Neither extra installs TensorRT. `TensorRTProvider` loads the TensorRT runtime at session
# creation and expects the TensorRT libraries (NVIDIA's `tensorrt` wheels or a system install)
# on top of `[gpu]`. Install them yourself only if you pin TensorRT; the default policy never
# picks it.
#
# Through the meta-package, `openstef[foundation-models]` installs the CPU runtime. For the
# GPU runtime, install `openstef-foundation-models[gpu]` directly, or in the uv workspace use
# the `dev-gpu` group (`uv sync --no-default-groups --group dev-gpu`).

# %% [markdown]
# ## Pick a checkpoint
#
# Use the `Chronos2` catalog instead of writing repo ids by hand. Each entry resolves to a
# published checkpoint reference; printing it shows the repo id and file the workflow will
# pull. A `ForecastingWorkflowConfig` with no checkpoint defaults to the base, dynamic-shape
# build shown first below.

# %%
from openstef_foundation_models.models import CheckpointVariant, Chronos2

catalog = {
    # Default: base model, dynamic shapes.
    "default (base, dynamic)": Chronos2.BASE.checkpoint(),
    # Smaller model, faster to download and run.
    "small": Chronos2.SMALL.checkpoint(),
    # Static shapes, which the default policy needs to pick CoreML on macOS.
    "base, static": Chronos2.BASE.checkpoint(CheckpointVariant.STATIC),
    # Let the host choose: static on macOS, dynamic elsewhere.
    "base, recommended": Chronos2.BASE.checkpoint(CheckpointVariant.recommended()),
}
for label, ref in catalog.items():
    print(f"{label:24} -> {ref.repo_id} :: {ref.filename}")


# %% [markdown]
# ## Run a checkpoint already on disk
#
# A file you exported or downloaded yourself is described by `LocalCheckpoint`. It needs the
# `.onnx` weights and the matching `.metadata.json` file, which records the model's input
# names, quantiles, context length, horizon, and precision. Building the config does not read
# the files, so this validates the wiring without a model present.

# %%
from pathlib import Path

from openstef_foundation_models.models.checkpoint import LocalCheckpoint
from openstef_foundation_models.presets import ForecastingWorkflowConfig

local_config = ForecastingWorkflowConfig(
    checkpoint=LocalCheckpoint(
        path=Path("artifacts/chronos-2.onnx"),
        metadata_path=Path("artifacts/chronos-2.metadata.json"),
    ),
)
print(local_config.checkpoint)


# %% [markdown]
# ## Choose the execution provider for your hardware
#
# A foundation model runs its ONNX graph through an *execution provider*: plain CPU, NVIDIA
# CUDA, or CoreML on Apple Silicon. By default OpenSTEF reads your machine and the checkpoint
# and picks a sensible chain, falling back to CPU when no accelerator fits. The table shows
# what to install and what gets chosen:
#
# | Hardware | Install | Provider chosen | Notes |
# | --- | --- | --- | --- |
# | NVIDIA GPU | `[gpu]` | CUDA, then CPU | Fastest option. The extra installs the CUDA 12 / cuDNN 9 wheels. |
# | Apple Silicon | `[cpu]` | CoreML, then CPU | Needs a static-shape checkpoint (`CheckpointVariant.STATIC` or `recommended()`). |
# | AMD GPU | `[cpu]` | CPU | No supported GPU provider today, so it runs on CPU. See "Extend the backend" below. |
# | CPU only | `[cpu]` | CPU | Prefer `Chronos2.SMALL`, and an `int8` checkpoint where one is published. |
#
# To pin the chain yourself, pass an explicit `providers` list on the backend config. An
# explicit list is strict: a missing accelerator raises instead of silently falling back to
# CPU. The provider configs are `CpuProvider`, `CudaProvider`, `TensorRTProvider`, and
# `CoreMLProvider`. TensorRT is never chosen by the default policy, so name it explicitly if
# you want it, and install the TensorRT runtime yourself since `[gpu]` does not include it.

# %%
from openstef_foundation_models.inference import CpuProvider, CudaProvider, ExecutionProvider
from openstef_foundation_models.presets import OnnxBackendConfig

providers: list[ExecutionProvider] = [CudaProvider(device_id=0), CpuProvider()]
gpu_config = ForecastingWorkflowConfig(backend=OnnxBackendConfig(providers=providers))
ordered_providers = [provider.to_ort()[0] for provider in providers]
print(f"ONNX Runtime will try these providers in order: {ordered_providers}")
print(f"Pinned on the workflow config: {gpu_config.backend.providers is providers}")


# %% [markdown]
# The config is only the recipe. `create_forecasting_workflow` turns it into a running
# forecaster with that provider chain baked in, so you set the hardware once here and never
# pass it again at predict time. Below we pin `CpuProvider` so the workflow runs anywhere,
# including this docs build, then forecast one short window to confirm the wiring end to end.

# %%
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.presets import create_forecasting_workflow
from openstef_models.utils.feature_selection import Include

HORIZON = LeadTime.from_string("P7D")

cpu_workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        checkpoint=Chronos2.SMALL.checkpoint(),
        quantiles=[Q(0.5)],
        horizons=[HORIZON],
        target_column="load",
        selected_features=Include("load"),
        backend=OnnxBackendConfig(providers=[CpuProvider()]),
    ),
)
# Zero-shot: the model is ready on construction, nothing to train.
print(f"is_fitted: {cpu_workflow.model.is_fitted}  quantiles: {cpu_workflow.model.quantiles}")


# %% tags=["remove-cell"]
# Helper: a small load window from the Liander benchmark dataset to drive the demos below.
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset

_dataset = load_liander_dataset()
_origin = datetime.fromisoformat("2024-11-15T00:00:00Z")
demo_window = _dataset.filter_by_range(start=_origin - timedelta(days=60), end=_origin + HORIZON.value)
demo_origins = [_origin, _origin + timedelta(days=14)]
demo_windows = [
    _dataset.filter_by_range(start=origin - timedelta(days=60), end=origin + HORIZON.value) for origin in demo_origins
]


# %%
forecast = cpu_workflow.predict(demo_window, forecast_start=demo_origins[0])
print(f"Forecast rows: {len(forecast.data)}  quantiles: {forecast.quantiles}")


# %% [markdown]
# ### Extend the backend
#
# The forecaster does not call ONNX Runtime directly; it goes through the `InferenceBackend`
# protocol, and `OnnxBackend` is the one implementation today. To use a provider OpenSTEF does
# not select for you, such as ROCm on an AMD GPU, add a custom `ExecutionProvider` and pass it
# in the `providers` list, or implement your own `InferenceBackend`. OpenSTEF does not test
# these paths, so their quality is on you; ROCm in particular is niche and unsupported. See the
# {doc}`API reference </api/foundation_models>` for the protocol.

# %% [markdown]
# ## Batch many windows
#
# A foundation model forecasts many windows in a single backend call. Instead of calling
# `predict` in a loop, collect the windows and pass them to `predict_batch`: it concatenates
# them, runs the ONNX session once, and returns one forecast per window in input order. The
# results match a serial loop; only throughput changes. Batching pays off when you forecast many
# locations or targets at once, or sweep many forecast origins across history. Here we reuse the
# CPU workflow and the two demo windows from above.

# %%
batched = cpu_workflow.predict_batch(demo_windows, forecast_start=demo_origins)
print(f"Forecasts returned: {len(batched)} (one backend call for {len(demo_windows)} windows)")


# %% [markdown]
# The quickstart plots a batched run against actual load in its
# {doc}`batched section </user_guide/getting_started/foundation_model_forecasting_quickstart>`.
# In backtesting you do not batch by hand: `FoundationModelBacktestForecaster` takes a
# `batch_size`, and beam stacks that many consecutive windows into one call, shown next.

# %% [markdown]
# ## Backtest with openstef-beam
#
# `FoundationModelBacktestForecaster` wraps a workflow so beam can drive it over historical
# windows. It reuses the workflow's ONNX session for every window; `batch_size` sets how many
# consecutive windows beam stacks into a single call. Plug the adapter into a `BacktestPipeline`
# the same way as any other backtest forecaster; see the
# {ref}`Backtesting concept page <concept_beam>` for the pipeline. Here we wrap the CPU workflow
# from above.

# %%
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster

adapter = FoundationModelBacktestForecaster.from_workflow(cpu_workflow, batch_size=16)
print(f"Adapter forecasts quantiles {adapter.quantiles} with batch size {adapter.batch_size}")
