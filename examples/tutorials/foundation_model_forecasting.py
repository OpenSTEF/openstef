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
# This tutorial produces a zero-shot probabilistic load forecast with the pretrained
# [Chronos-2](https://huggingface.co/amazon/chronos-2) model using OpenSTEF's ONNX
# inference backend. No training is involved, and the forecast is conditioned on
# known weather covariates.
#
# What you'll do:
#
# - Select a published Chronos-2 checkpoint from the HuggingFace Hub
# - Build a forecasting workflow from a config with `create_forecasting_workflow`
# - Feed load history plus known-future weather and read the predicted quantiles
# - Plot a P30 / P50 / P70 forecast
# - Forecast several origins at once with a single `predict_batch` call
#
# ```{note}
# Chronos-2 is zero-shot: it is pretrained and needs no `fit()`. You give it a
# window of recent load (and optional known-future covariates) and it returns a
# probabilistic forecast directly. Covariates cover the whole time range, so the
# model sees each weather series both as history and as known future values and can
# react to, say, an incoming cold snap.
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
# ## Assemble the workflow
#
# `ForecastingWorkflowConfig` declares the model family, the checkpoint that backs it,
# the quantiles/horizons to predict, and which columns are the target and the weather
# covariates. OpenSTEF publishes Chronos-2 as ONNX checkpoints on the HuggingFace Hub;
# the `Chronos2` catalog turns a size into a checkpoint reference, downloaded and cached
# on first use. The config defaults to the full `Chronos2.BASE`, so the checkpoint is
# optional — here we pick the compact `Chronos2.SMALL` so the tutorial stays fast and
# runs in the docs build (pass a `LocalCheckpoint(path=...)` to run a file on disk).
# `create_forecasting_workflow` then resolves the checkpoint, builds the ONNX Runtime
# session once, and wraps a `Chronos2Forecaster` in a `CustomForecastingWorkflow` (a
# `Selector` that picks the target and covariates, the forecaster, and a `QuantileSorter`).

# %%
from openstef_core.types import LeadTime, Q
from openstef_foundation_models.models import Chronos2
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.utils.feature_selection import Include

HORIZON = LeadTime.from_string("P7D")

workflow = create_forecasting_workflow(
    ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=Chronos2.SMALL.checkpoint(),
        quantiles=[Q(0.3), Q(0.5), Q(0.7)],
        horizons=[HORIZON],
        target_column="load",
        # Keep the target plus the three known-future weather covariates; every
        # kept non-target column is forwarded to Chronos-2 as a covariate.
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
print(f"quantiles: {workflow.model.quantiles}")


# %% [markdown]
# ## Load real load history and weather
#
# We reuse the [Liander 2024 benchmark](https://huggingface.co/datasets/Alliander/MSL_Benchmark_Dataset)
# dataset for a realistic medium-voltage feeder load series together with its
# weather forecasts. The workflow's `Selector` keeps the target (`load`) and the
# three weather covariates; everything else is ignored.
#
# We take 60 days of history up to a chosen forecast start and keep the weather
# columns running through the 7-day horizon, so Chronos-2 can use the known-future
# weather as a covariate.

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

forecast_start = datetime.fromisoformat("2024-11-15T00:00:00Z")
context_start = forecast_start - timedelta(days=60)

# The window spans history + horizon: load history conditions the model, while the
# weather columns are known across the whole range (history and future).
window = dataset.filter_by_range(start=context_start, end=forecast_start + HORIZON.value)

print(f"Window:   {context_start:%Y-%m-%d} to {forecast_start + HORIZON.value:%Y-%m-%d}, {len(window.data):,} rows")


# %% [markdown]
# ## Forecast
#
# `workflow.predict` selects the target and covariates, runs the ONNX session once,
# and post-processes the output: it slices the model's frozen horizon to the
# requested 7 days and resamples Chronos-2's native quantile grid onto the
# requested P30 / P50 / P70.

# %%
forecast = workflow.predict(window, forecast_start=forecast_start)

print(f"Forecast rows: {len(forecast.data)}")
print(f"Quantiles:     {forecast.quantiles}")
forecast.data.head()


# %% tags=["remove-cell"]
assert len(forecast.data) > 1, "Expected a multi-step forecast"
assert forecast.quantiles == [Q(0.3), Q(0.5), Q(0.7)], "Quantiles should match the request"


# %% [markdown]
# ## Visualize the forecast
#
# [`ForecastTimeSeriesPlotter`](https://openstef.github.io/openstef/api/generated/openstef_beam.analysis.plots.ForecastTimeSeriesPlotter.html)
# overlays the actual load against the median forecast with a shaded quantile band.

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
# ## Forecast many origins in one call
#
# Foundation models shine when you forecast **many** series or origins at once.
# Instead of looping `predict` per window, hand the whole batch to `predict_batch`:
# it concatenates the windows and runs the ONNX session a **single** time, returning
# one forecast per window in input order. The numbers are identical to the serial
# loop, batching is purely a throughput optimization.
#
# Here we carve four forecast origins two weeks apart out of the same dataset. While this is useful for backtesting, in a live setting you would typically forecast many different locations or targets at once. Each
# window keeps its own 60 days of history plus the 7-day horizon of known-future
# weather.

# %%
forecast_starts = [
    datetime.fromisoformat("2024-09-15T00:00:00Z"),
    datetime.fromisoformat("2024-09-29T00:00:00Z"),
    datetime.fromisoformat("2024-10-13T00:00:00Z"),
    datetime.fromisoformat("2024-10-27T00:00:00Z"),
]
windows = [
    dataset.filter_by_range(start=start - timedelta(days=60), end=start + HORIZON.value) for start in forecast_starts
]

batched = workflow.predict_batch(windows, forecast_start=forecast_starts)
print(f"Forecasts returned: {len(batched)} (one backend call for the whole batch)")


# %% tags=["remove-cell"]
from openstef_core.testing import assert_timeseries_equal

serial = [
    workflow.predict(window, forecast_start=start) for window, start in zip(windows, forecast_starts, strict=True)
]
assert len(batched) == len(serial), "Batched and serial runs should return the same number of forecasts"
for batch_item, serial_item in zip(batched, serial, strict=True):
    assert_timeseries_equal(batch_item, serial_item)


# %% [markdown]
# Each window is an independent 7-day forecast. We overlay the four median forecasts
# against the actual load to see how the same zero-shot model tracks the series at
# different points in time.

# %% tags=["hide-input"]
batch_actuals = dataset.filter_by_range(
    start=forecast_starts[0] - timedelta(days=3),
    end=forecast_starts[-1] + HORIZON.value,
).data["load"]

batch_plotter = ForecastTimeSeriesPlotter().add_measurements(measurements=batch_actuals)
for start, batch_forecast in zip(forecast_starts, batched, strict=True):
    batch_plotter = batch_plotter.add_model(
        model_name=f"Chronos-2 {start:%b %d}",
        forecast=batch_forecast.median_series,
        quantiles=batch_forecast.quantiles_data,
    )

fig = cast(Any, batch_plotter.plot())
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
# - {doc}`/tutorials/forecasting_quickstart` — train a classical gradient-boosted
#   model and compare it against this zero-shot baseline.
# - {doc}`/tutorials/backtesting_quickstart` — evaluate a forecaster over historical
#   windows. The `FoundationModelBacktestForecaster` adapter (the `[benchmarking]`
#   extra) runs Chronos-2 through the same backtesting pipeline, loading the ONNX
#   session once and reusing it across every window.
