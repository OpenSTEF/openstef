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
# 1. Downloads the chosen Chronos-2 ONNX checkpoint from the HuggingFace Hub once and
#    reuses the loaded session for every target
# 2. Runs day-by-day backtesting on a subset of the dataset (wind parks by default)
# 3. Produces probabilistic forecasts (7 quantiles) for a 3-day horizon
# 4. Saves results locally for comparison (see the *Compare Results* notebook)
#
# The model size, on-disk precision, execution provider, and batch size are all
# selectable via environment variables (see the *Configuration* section), so you can
# sweep performance setups without editing code:
#
# ```bash
# CHRONOS2_SIZE=small CHRONOS2_PROVIDER=tensorrt-fp16 CHRONOS2_BATCH_SIZE=16 \
#     python run_chronos2_benchmark.py
# ```
#
# ```{admonition} The model stays loaded across targets
# :class: tip
# Chronos-2 is zero-shot, so the workflow and its loaded ONNX session are built
# once and shared across every target. This only holds when the benchmark runs
# sequentially (`N_PROCESSES = 1`): separate worker processes cannot share a live
# ONNX session and would load one copy of the model each.
# ```
#
# ```{note}
# This benchmark downloads the published Chronos-2 ONNX checkpoint from the
# HuggingFace Hub (`OpenSTEF/chronos-2-onnx` or `OpenSTEF/chronos-2-small-onnx`), so
# it needs network access on first run but no local export. It is not run during the
# docs build.
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
from openstef_foundation_models.inference.providers import CpuProvider, CudaProvider, TensorRTProvider
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster
from openstef_foundation_models.models.checkpoint import HubCheckpoint
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    OnnxBackendConfig,
    create_forecasting_workflow,
)
from openstef_models.utils.feature_selection import Include

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("chronos2_benchmark")

# %% [markdown]
# ## Configuration
#
# Pick which categories to benchmark, the forecast horizon, and the quantiles.
# Wind parks are the default subset; add more categories to widen the run.

# %%
OUTPUT_PATH = Path("./benchmark_results")

# --- Model setup switches (all env-overridable for easy A/B testing) ---------
#
# Mix and match these three knobs to sweep performance setups without editing code:
#
#   CHRONOS2_SIZE=base|small      python run_chronos2_benchmark.py
#   CHRONOS2_PRECISION=fp32|fp32-static|int8
#   CHRONOS2_PROVIDER=auto|cuda|tensorrt-fp16|tensorrt-fp32|cpu
#   CHRONOS2_BATCH_SIZE=<int>
#
# Model size. "base" -> OpenSTEF/chronos-2-onnx, "small" -> OpenSTEF/chronos-2-small-onnx.
# The small model has far fewer FLOPs per forecast (faster, slightly less accurate).
MODEL_SIZE = os.environ.get("CHRONOS2_SIZE", "small")

# On-disk precision/shape variant of the published ONNX weights:
#   "fp32"        -> dynamic-shape FP32 (chronos-2.onnx)            [default, portable]
#   "fp32-static" -> frozen-shape FP32  (chronos-2_static.onnx)    [enables more graph opt / TensorRT specialisation]
#   "int8"        -> 8-bit quantized    (chronos-2_int8.onnx)      [smallest/fastest on CPU, accuracy-dependent]
# Note: FP16 is not a separate file - get it at runtime via CHRONOS2_PROVIDER=tensorrt-fp16.
PRECISION = os.environ.get("CHRONOS2_PRECISION", "fp32")

# ONNX Runtime execution-provider chain:
#   "auto"          -> let the default policy pick from checkpoint metadata + host
#   "cuda"          -> CUDA EP (FP32 math) with CPU fallback
#   "tensorrt-fp16" -> TensorRT EP with FP16 (fastest on NVIDIA), CUDA/CPU fallback
#   "tensorrt-fp32" -> TensorRT EP at FP32, CUDA/CPU fallback
#   "cpu"           -> CPU only
# TensorRT builds an engine on first run; it is cached under ./trt_cache for reuse.
PROVIDER = os.environ.get("CHRONOS2_PROVIDER", "auto")

# Identifies this setup so sweep runs land in separate result folders and run names.
RUN_TAG = f"{MODEL_SIZE}-{PRECISION}-{PROVIDER}-b{os.environ.get('CHRONOS2_BATCH_SIZE', '1')}"
BENCHMARK_RESULTS_PATH_CHRONOS2 = OUTPUT_PATH / "Chronos2" / RUN_TAG

# Run sequentially so the loaded model is reused across every target (see the note
# at the top). A value > 1 would load one model copy per worker process.
N_PROCESSES = 1

# Which Liander2024 categories to benchmark. Start with wind parks; add more here,
# e.g. ["wind_park", "solar_park"]. Set to None to run every category.
BENCHMARK_FILTER: list[Liander2024Category] | None = ["wind_park"]

# Forecast 3 days ahead, producing 7 quantile bands.
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

# Batch size for inference. The backtest adapter groups this many forecast origins
# into a single `predict_batch` call, which Chronos-2 serves with one backend
# invocation instead of one per origin. Larger values trade memory for throughput.
# Set to 1 to fall back to the serial, one-origin-at-a-time path. Override via the
# CHRONOS2_BATCH_SIZE environment variable.
BATCH_SIZE = int(os.environ.get("CHRONOS2_BATCH_SIZE", "48"))

# %% [markdown]
# ## Resolve the setup into a checkpoint and a backend
#
# The three switches above are turned into a `HubCheckpoint` (which weights + metadata
# to download from the Hub) and an `OnnxBackendConfig` (how to run them). Both the
# weights and their `<weights>.metadata.json` are downloaded and cached on first use.
# Keeping these as small builders makes it trivial to sweep setups from the
# environment without touching the workflow below.

# %%
_MODEL_SLUGS = {"base": "chronos-2", "small": "chronos-2-small"}
_PRECISION_SUFFIX = {"fp32": "", "fp32-static": "_static", "int8": "_int8"}


def build_checkpoint(size: str, precision: str) -> HubCheckpoint:
    """Build the Hub checkpoint reference for a model *size* and *precision* variant.

    Args:
        size: Model size key, one of ``_MODEL_SLUGS`` (``base`` or ``small``).
        precision: Precision/shape variant key, one of ``_PRECISION_SUFFIX``.

    Returns:
        A ``HubCheckpoint`` pointing at the selected published weights and metadata.

    Raises:
        ValueError: If ``size`` or ``precision`` is not a recognised option.
    """
    if size not in _MODEL_SLUGS:
        msg = f"Unknown CHRONOS2_SIZE={size!r}; choose one of {sorted(_MODEL_SLUGS)}."
        raise ValueError(msg)
    if precision not in _PRECISION_SUFFIX:
        msg = f"Unknown CHRONOS2_PRECISION={precision!r}; choose one of {sorted(_PRECISION_SUFFIX)}."
        raise ValueError(msg)
    slug = _MODEL_SLUGS[size]
    # The metadata filename (e.g. chronos-2_int8.metadata.json) is auto-discovered.
    return HubCheckpoint(repo_id=f"OpenSTEF/{slug}-onnx", filename=f"{slug}{_PRECISION_SUFFIX[precision]}.onnx")


def build_backend(provider: str) -> OnnxBackendConfig:
    """Build the ONNX backend config for the selected execution-provider chain.

    Args:
        provider: Provider-chain key (``auto``, ``cuda``, ``tensorrt-fp16``,
            ``tensorrt-fp32``, or ``cpu``).

    Returns:
        An ``OnnxBackendConfig`` with the resolved provider chain (``None`` for
        ``auto``, which defers to the default policy).

    Raises:
        ValueError: If ``provider`` is not a recognised option.
    """
    trt_cache = OUTPUT_PATH / "trt_cache"
    chains: dict[str, list | None] = {
        # None lets the default policy choose from checkpoint metadata + host.
        "auto": None,
        "cuda": [CudaProvider(), CpuProvider()],
        "tensorrt-fp16": [TensorRTProvider(fp16=True, engine_cache_dir=trt_cache), CudaProvider(), CpuProvider()],
        "tensorrt-fp32": [TensorRTProvider(fp16=False, engine_cache_dir=trt_cache), CudaProvider(), CpuProvider()],
        "cpu": [CpuProvider()],
    }
    if provider not in chains:
        msg = f"Unknown CHRONOS2_PROVIDER={provider!r}; choose one of {sorted(chains)}."
        raise ValueError(msg)
    return OnnxBackendConfig(providers=chains[provider])


checkpoint = build_checkpoint(MODEL_SIZE, PRECISION)
backend = build_backend(PROVIDER)
logger.info(
    "Chronos-2 setup: size=%s precision=%s provider=%s batch_size=%s", MODEL_SIZE, PRECISION, PROVIDER, BATCH_SIZE
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
        # Compute backend (execution-provider chain) chosen by the CHRONOS2_PROVIDER
        # switch above. Pass `providers=None` ("auto") to let the default policy read
        # the checkpoint metadata and host to pick a chain automatically.
        backend=backend,
    )
)


# %% [markdown]
# ## Forecaster factory
#
# The benchmark calls this factory once per target. It wraps the shared workflow in
# a backtest adapter without rebuilding it, so the loaded ONNX session is reused for
# every location.


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
        run_name=f"chronos2-{RUN_TAG}",
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )
