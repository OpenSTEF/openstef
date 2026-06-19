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
# # Batch Inference with Chronos-2
#
# This tutorial forecasts several independent series in a single call with
# [Chronos-2](https://huggingface.co/amazon/chronos-2). Instead of looping over each
# forecast origin and calling `predict` once per item, you hand the whole batch to
# `workflow.predict_batch` and the foundation model serves it with a single backend
# invocation.
#
# What you'll do:
#
# - Build a zero-shot Chronos-2 forecasting workflow (no training)
# - Assemble several context windows, each with its own forecast origin
# - Forecast them all at once with `predict_batch`
# - Confirm the batched result is identical to looping `predict`
#
# ```{note}
# Batching is where foundation models shine: a `predict_batch` call concatenates the
# windows and runs the ONNX session **once** for the whole batch, rather than once
# per window. The classical `predict` loop still works and returns the same numbers,
# but issues one backend call per item. For large backtests or many feeders this is
# the difference between one inference and hundreds.
# ```
#
# ```{note}
# This tutorial reads a local ONNX export of Chronos-2, so it is not run during the
# docs build. Point `artifact_path` at your own export to run it locally.
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
# A checkpoint is the ONNX weights file plus a small `CheckpointMetadata` JSON file
# (`<weights>.metadata.json`) describing the model's tensor names, native quantile
# grid, and context/horizon sizing. The metadata file sits next to the weights and
# is discovered automatically.

# %%
from pathlib import Path

from openstef_foundation_models.models.checkpoint import LocalCheckpoint

artifact_path = Path("chronos-onnx-lab/artifacts/chronos-2.onnx")
if not artifact_path.is_file():
    msg = f"Chronos-2 ONNX artifact not found at {artifact_path}. Export it with the chronos-onnx-lab script."
    raise FileNotFoundError(msg)

# The metadata JSON (chronos-2.metadata.json) is auto-discovered next to the weights.
checkpoint = LocalCheckpoint(path=artifact_path)
print(f"Checkpoint: {artifact_path.name} ({artifact_path.stat().st_size / 1e6:.0f} MB)")

# %% [markdown]
# ## Assemble the workflow
#
# The workflow is identical to the single-forecast tutorial: a zero-shot Chronos-2
# forecaster that selects the target plus weather covariates and sorts quantiles. The
# same instance serves both `predict` and `predict_batch`.

# %%
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.utils.feature_selection import Include

HORIZON = LeadTime.from_string("P7D")

workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=checkpoint,
        quantiles=[Q(0.3), Q(0.5), Q(0.7)],
        horizons=[HORIZON],
        target_column="load",
        selected_features=Include(
            "load",
            "shortwave_radiation",
            "wind_speed_80m",
            "temperature_2m",
        ),
    )
)

# Zero-shot: the model is "fitted" on construction - there is nothing to train.
print(f"is_fitted: {workflow.model.is_fitted}")

# %% [markdown]
# ## Build a batch of windows
#
# We reuse the [Liander 2024 benchmark](https://huggingface.co/datasets/Alliander/MSL_Benchmark_Dataset)
# dataset and carve out several forecast origins, two weeks apart. Each origin yields
# one context window (60 days of history plus known-future weather through the
# horizon) and its own forecast start. In practice these windows would just as easily
# be different feeders or substations - one entry per series you want to forecast.

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

# Four forecast origins, two weeks apart - one independent forecasting task each.
forecast_starts = [
    datetime.fromisoformat("2024-09-15T00:00:00Z"),
    datetime.fromisoformat("2024-09-29T00:00:00Z"),
    datetime.fromisoformat("2024-10-13T00:00:00Z"),
    datetime.fromisoformat("2024-10-27T00:00:00Z"),
]

# Each window spans 60 days of history plus the 7-day horizon of known-future weather.
windows = [
    dataset.filter_by_range(start=start - timedelta(days=60), end=start + HORIZON.value) for start in forecast_starts
]

print(f"Batch size: {len(windows)} windows")
for start, win in zip(forecast_starts, windows, strict=True):
    print(f"  origin {start:%Y-%m-%d}: {len(win.data):,} rows")

# %% [markdown]
# ## Forecast the whole batch at once
#
# `predict_batch` takes the list of windows and a matching list of forecast origins.
# It selects the target and covariates for every window, concatenates them, and runs
# the ONNX session a **single** time for the entire batch. The result is one forecast
# per window, in input order.

# %%
batched = workflow.predict_batch(windows, forecast_start=forecast_starts)

print(f"Forecasts returned: {len(batched)}")
for start, forecast in zip(forecast_starts, batched, strict=True):
    print(f"  origin {start:%Y-%m-%d}: {len(forecast.data)} rows, quantiles {forecast.quantiles}")

# %% [markdown]
# ## Confirm it matches the serial loop
#
# Batching is a pure throughput optimization - it must not change the numbers. We
# forecast the same windows one at a time with `predict` and check the results are
# identical.

# %%
serial = [
    workflow.predict(window, forecast_start=start) for window, start in zip(windows, forecast_starts, strict=True)
]

# %% tags=["remove-cell"]
from openstef_core.testing import assert_timeseries_equal

assert len(batched) == len(serial), "Batched and serial runs should return the same number of forecasts"
for batch_item, serial_item in zip(batched, serial, strict=True):
    assert_timeseries_equal(batch_item, serial_item)

# %%
all_equal = all(
    batch_item.data.equals(serial_item.data) for batch_item, serial_item in zip(batched, serial, strict=True)
)
print(f"Batched result matches serial loop: {all_equal}")

# %% [markdown]
# ## Visualize the batch
#
# Each window is an independent 7-day forecast. We overlay all four median forecasts
# against the actual load to see how the same zero-shot model tracks the series at
# different points in time.

# %% tags=["hide-input"]
import plotly.graph_objects as go

actuals = dataset.filter_by_range(
    start=forecast_starts[0] - timedelta(days=3),
    end=forecast_starts[-1] + HORIZON.value,
).data["load"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=actuals.index, y=actuals.to_numpy(), name="Actual", line={"color": "#444"}))
for start, forecast in zip(forecast_starts, batched, strict=True):
    median = forecast.median_series
    fig.add_trace(go.Scatter(x=median.index, y=median.to_numpy(), name=f"Forecast {start:%b %d}"))

fig = cast(Any, fig)
fig.update_layout(
    title="Chronos-2 batched zero-shot forecasts vs actuals",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=500,
)
fig.show()

# %% [markdown]
# ## Next steps
#
# - {doc}`/tutorials/foundation_model_forecasting` — the single-window version of
#   this tutorial, with a closer look at how Chronos-2 uses covariates.
# - {doc}`/tutorials/backtesting_quickstart` — evaluate a forecaster over historical
#   windows. The `FoundationModelBacktestForecaster` adapter batches forecast origins
#   through this same `predict_batch` path; set its `batch_size` to control how many
#   windows share each backend call.
