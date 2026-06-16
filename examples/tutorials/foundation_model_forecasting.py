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
# [Chronos-2](https://huggingface.co/amazon/chronos-2) foundation model - no
# training - using OpenSTEF's ONNX inference backend, and condition it on known
# weather **covariates**.
#
# **What you'll learn:**
#
# - Point at a local Chronos-2 ONNX checkpoint described by a metadata sidecar
# - Assemble a forecasting workflow from a declarative config via `create_forecasting_workflow`
# - Feed raw load history plus known-future weather covariates and read raw-scale quantiles
# - Generate and visualize a P10 / P50 / P90 forecast
#
# ```{note}
# Chronos-2 is **zero-shot**: it is pretrained and needs no `fit()`. You feed it a
# window of recent load (and optional known-future covariates) and it returns a
# probabilistic forecast directly.
# ```
#
# ```{note}
# In OpenSTEF, covariates span the **whole** time range - history *and* future.
# Chronos-2 sees each weather series' recent history as context and its known
# horizon values as a known-future covariate, so the forecast can react to, say,
# an incoming cold snap.
# ```
#
# ```{warning}
# This tutorial loads a **local** ONNX export of Chronos-2 that is not published yet,
# so it is **not executed** during the docs build. To run it yourself, export the
# checkpoint with the `chronos-onnx-lab` script and run the notebook locally.
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
# (`<weights>.metadata.json`) describing the model's tensor names, native quantile
# grid, and context/horizon sizing. Keeping these specifics in **data** (not code)
# is what lets the same generic inference backend serve any foundation model.
#
# Here we point at a local export produced by the `chronos-onnx-lab` script; its
# metadata sidecar sits next to the weights and is discovered automatically. Once
# the checkpoint is published to the HuggingFace Hub, this becomes a one-line
# `HubCheckpoint(...)`.

# %%
from pathlib import Path

from openstef_foundation_models.models.checkpoint import LocalCheckpoint

artifact_path = Path("chronos-onnx-lab/artifacts/chronos-2.onnx")
if not artifact_path.is_file():
    msg = (
        f"Chronos-2 ONNX artifact not found at {artifact_path}. "
        "Export it with the chronos-onnx-lab script."
    )
    raise FileNotFoundError(msg)

# The metadata sidecar (chronos-2.metadata.json) is auto-discovered next to the weights.
checkpoint = LocalCheckpoint(path=artifact_path)
print(f"Checkpoint: {artifact_path.name} ({artifact_path.stat().st_size / 1e6:.0f} MB)")

# %% [markdown]
# ## Assemble the workflow
#
# `ForecastingWorkflowConfig` declares the model family, the checkpoint that backs
# it, the quantiles/horizons to predict, and which columns are the target and the
# weather covariates. `create_forecasting_workflow` resolves the checkpoint, builds
# the ONNX Runtime session **once**, and wraps a `Chronos2Forecaster` in a
# `CustomForecastingWorkflow` (a `Selector` that picks the target + covariates, the
# forecaster, and a `QuantileSorter`). The ONNX dependency is imported lazily, so
# importing the config alone stays light.

# %%
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)

HORIZON = LeadTime.from_string("PT48H")

workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=checkpoint,
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[HORIZON],
        target_column="load",
        radiation_column="shortwave_radiation",
        wind_speed_column="wind_speed_80m",
        temperature_column="temperature_2m",
    )
)

# Zero-shot: the model is "fitted" on construction - there is nothing to train.
print(f"is_fitted: {workflow.model.is_fitted}")
print(f"quantiles: {workflow.model.quantiles}")

# %% [markdown]
# ## Load real load history and weather
#
# We reuse the [Liander 2024 benchmark](https://huggingface.co/datasets/Alliander/MSL_Benchmark_Dataset)
# dataset for a realistic medium-voltage feeder load series together with its
# weather forecasts. The workflow's `Selector` keeps the target (`load`) and the
# three weather covariates; everything else is ignored.
#
# We take 60 days of history up to a chosen forecast start **and** keep the weather
# columns running through the 48-hour horizon, so Chronos-2 can use the known-future
# weather as a covariate. The raw load is fed unscaled: Chronos-2 normalizes each
# series internally and returns predictions on the original scale.

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

forecast_start = datetime.fromisoformat("2024-04-15T00:00:00Z")
context_start = forecast_start - timedelta(days=60)

# The window spans history + horizon: load history conditions the model, while the
# weather columns are known across the whole range (history and future).
window = dataset.filter_by_range(start=context_start, end=forecast_start + HORIZON.value)

print(
    f"Window:   {context_start:%Y-%m-%d} to {forecast_start + HORIZON.value:%Y-%m-%d}, "
    f"{len(window.data):,} rows"
)

# %% [markdown]
# ## Forecast
#
# `workflow.predict` selects the target and covariates, runs the ONNX session once,
# and post-processes the output: it slices the model's frozen horizon to the
# requested 48 hours and resamples Chronos-2's native 21-quantile grid onto the
# requested P10 / P50 / P90.

# %%
forecast = workflow.predict(window, forecast_start=forecast_start)

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
