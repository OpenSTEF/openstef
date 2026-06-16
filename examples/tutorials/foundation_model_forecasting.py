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
# # Foundation-Model Forecasting with Chronos-2
#
# Produce a **zero-shot** probabilistic load forecast with the pretrained
# [Chronos-2](https://huggingface.co/amazon/chronos-2) foundation model — no
# training, no feature engineering — using OpenSTEF's ONNX inference backend.
#
# **What you'll learn:**
#
# - Point at a local Chronos-2 ONNX checkpoint and describe it with a metadata sidecar
# - Build a `Chronos2Forecaster` from a declarative config via `create_foundation_forecaster`
# - Feed raw load history (the model owns its own normalization) and read raw-scale quantiles
# - Generate and visualize a P10 / P50 / P90 forecast
#
# ```{note}
# Chronos-2 is **zero-shot**: it is pretrained and needs no `fit()`. You feed it a
# window of recent load and it returns a probabilistic forecast directly.
# ```
#
# ```{warning}
# This tutorial loads a **local** ONNX export of Chronos-2 that is not published yet,
# so it is **not executed** during the docs build. To run it yourself, export the
# checkpoint with the `chronos-onnx-lab` script (or set `OPENSTEF_CHRONOS2_ONNX_PATH`
# to your own export) and run the notebook locally.
# ```

# %% tags=["remove-cell"]
import warnings
from typing import Any, cast

warnings.filterwarnings("ignore")

from openstef_core.testing import configure_notebook_display, setup_notebook_logging

configure_notebook_display()
logger = setup_notebook_logging(
    __name__,
    suppress=(
        "choreographer",
        "kaleido",
        "httpx",
        "huggingface_hub",
        "fsspec",
        "filelock",
        "openstef_core.datasets",
    ),
)

# %% [markdown]
# ## Locate the checkpoint
#
# A *checkpoint* is the ONNX weights file plus a small `CheckpointMetadata` sidecar
# describing the model's tensor names, native quantile grid, and context/horizon
# sizing. Keeping these specifics in **data** (not code) is what lets the same
# generic inference backend serve any foundation model.
#
# Here we point at a local export produced by the `chronos-onnx-lab` script and
# write its metadata sidecar to a temp file. Once the checkpoint is published to
# the HuggingFace Hub, this whole cell becomes a one-line `HubCheckpoint(...)`.

# %%
import os
import tempfile
from pathlib import Path

from openstef_foundation_models.models.checkpoint import CheckpointMetadata, LocalCheckpoint

# Native quantile grid Chronos-2 emits (21 levels), from its `chronos_config.quantiles`.
NATIVE_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]  # fmt: skip

# Default to the lab export; override with OPENSTEF_CHRONOS2_ONNX_PATH.
artifact_path = Path(os.environ.get("OPENSTEF_CHRONOS2_ONNX_PATH", "chronos-onnx-lab/artifacts/chronos-2.onnx"))
if not artifact_path.is_file():
    msg = (
        f"Chronos-2 ONNX artifact not found at {artifact_path}. "
        "Export it with the chronos-onnx-lab script or set OPENSTEF_CHRONOS2_ONNX_PATH."
    )
    raise FileNotFoundError(msg)

# Describe the exported graph (60-day context, 7-day horizon at 15-minute resolution).
metadata = CheckpointMetadata(
    model_family="chronos2",
    input_names=["context", "group_ids", "attention_mask"],
    output_name="quantile_preds",
    native_quantiles=NATIVE_QUANTILES,
    context_length=5760,
    output_patch_size=16,
    horizon_patches=42,
    resolution_minutes=15,
)
metadata_path = Path(tempfile.mkdtemp()) / "chronos-2.metadata.json"
metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

checkpoint = LocalCheckpoint(path=artifact_path, metadata_path=metadata_path)
print(f"Checkpoint: {artifact_path.name} ({artifact_path.stat().st_size / 1e6:.0f} MB)")
print(f"Horizon:    {metadata.horizon_length} steps  ·  context: {metadata.context_length} steps")

# %% [markdown]
# ## Build the forecaster
#
# `FoundationForecasterConfig` declares the model family, the backend that runs it,
# and the quantiles/horizons to predict. `create_foundation_forecaster` resolves the
# checkpoint, builds the ONNX Runtime session **once**, and composes it into a
# `Chronos2Forecaster`. The ONNX dependency is imported lazily inside the backend's
# `build()`, so importing the config alone stays light.

# %%
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.presets.forecasting_workflow import (
    FoundationForecasterConfig,
    OnnxBackendConfig,
    create_foundation_forecaster,
)

HORIZON = LeadTime.from_string("PT48H")

forecaster = create_foundation_forecaster(
    FoundationForecasterConfig(
        model="chronos2",
        backend=OnnxBackendConfig(checkpoint=checkpoint),
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[HORIZON],
    )
)

# Zero-shot: the model is "fitted" on construction — there is nothing to train.
print(f"is_fitted: {forecaster.is_fitted}")
print(f"quantiles: {forecaster.quantiles}")

# %% [markdown]
# ## Load real load history
#
# We reuse the [Liander 2024 benchmark](https://huggingface.co/datasets/Alliander/MSL_Benchmark_Dataset)
# dataset for a realistic medium-voltage feeder load series. Chronos-2 only needs the
# **target** column as context — no weather features, lags, or calendar encodings.
#
# We take 60 days of history up to a chosen forecast start and wrap it in a
# `ForecastInputDataset`. The raw load is fed unscaled: Chronos-2 normalizes each
# series internally and returns predictions on the original scale.

# %%
from datetime import datetime, timedelta

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

forecast_start = datetime.fromisoformat("2024-04-15T00:00:00Z")
context_start = forecast_start - timedelta(days=60)

history = dataset.filter_by_range(start=context_start, end=forecast_start)
input_data = ForecastInputDataset.from_timeseries(
    history,
    target_column="load",
    forecast_start=forecast_start,
)

print(
    f"Context:  {input_data.target_series.notna().sum():,} observed load points, "
    f"{context_start:%Y-%m-%d} to {forecast_start:%Y-%m-%d}"
)

# %% [markdown]
# ## Forecast
#
# `predict` runs the ONNX session once and post-processes the output: it slices the
# model's frozen horizon to the requested 48 hours and resamples Chronos-2's native
# 21-quantile grid onto the requested P10 / P50 / P90.

# %%
forecast = forecaster.predict(input_data)

print(f"Forecast rows: {len(forecast.data)}")
print(f"Quantiles:     {forecast.quantiles}")
forecast.data.head()

# %% tags=["remove-cell"]
assert len(forecast.data) > 1, "Expected a multi-step forecast"
assert forecast.quantiles == [Q(0.1), Q(0.5), Q(0.9)], "Quantiles should match the request"

# %% [markdown]
# ## Visualize the forecast
#
# [`ForecastTimeSeriesPlotter`](https://openstef.github.io/openstef/api/generated/openstef_beam.analysis.plots.ForecastTimeSeriesPlotter.html)
# overlays the actual load against the median forecast with a shaded P10-P90 band.

# %% tags=["hide-input"]
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

actuals = dataset.filter_by_range(
    start=forecast_start - timedelta(days=3),
    end=forecast_start + HORIZON.value,
).data["load"]

fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=actuals)
    .add_model(
        model_name="Chronos-2",
        forecast=forecast.median_series,
        quantiles=forecast.quantiles_data,
    )
    .plot()
)
fig = cast(Any, fig)
fig.update_layout(
    title="Chronos-2 zero-shot forecast vs actuals",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=500,
)
fig.show()

# %% [markdown]
# ## Next steps
#
# - {doc}`/tutorials/forecasting_quickstart` — train a classical gradient-boosted
#   model and compare it against this zero-shot baseline.
# - {doc}`/tutorials/backtesting_quickstart` — evaluate a forecaster over historical
#   windows. The `FoundationModelBacktestForecaster` adapter (the `[benchmarking]`
#   extra) runs Chronos-2 through the same backtesting pipeline, loading the ONNX
#   session once and reusing it across every window.
