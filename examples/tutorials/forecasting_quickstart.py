# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# # Forecasting Quickstart
#
# Train a GBLinear model on real energy data and generate probabilistic forecasts
# with confidence intervals — all in under a minute.
#
# **What you'll learn:**
#
# - Load the Liander 2024 benchmark dataset
# - Configure a forecasting workflow with `ForecastingWorkflowConfig`
# - Train a model and inspect evaluation metrics
# - Generate quantile forecasts (P10 / P50 / P90)
# - Visualize predictions against actuals
#
# ```{note}
# This tutorial uses a small data slice for fast execution.
# See `examples/benchmarks/` for production-scale runs.
# ```
#
# **Key API references:**
# [`ForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.ForecastingWorkflowConfig.html)
# [`create_forecasting_workflow`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.create_forecasting_workflow.html)
# · [`LeadTime`](https://openstef.github.io/openstef/api/generated/openstef_core.types.LeadTime.html)
# · [`Q`](https://openstef.github.io/openstef/api/generated/openstef_core.types.Quantile.html)

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
# ## Load the dataset
#
# The [Liander 2024 benchmark](https://huggingface.co/datasets/Alliander/MSL_Benchmark_Dataset)
# dataset contains load measurements, versioned weather forecasts, EPEX prices, and
# load profiles for a medium-voltage feeder in the Netherlands.
#
# We split the data into:
#
# - **45 days** of training data
# - **7 days** for forecasting
#
# The predict window includes **14 days of history** before the forecast start so
# that lag features (e.g. `load_lag_P7D`) can be computed during prediction.

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
train_end = train_start + timedelta(days=45)
forecast_end = train_end + timedelta(days=7)

train_dataset = dataset.filter_by_range(start=train_start, end=train_end)

# Include 14 days of history before forecast start for lag feature computation
predict_dataset = dataset.filter_by_range(
    start=train_end - timedelta(days=14),
    end=forecast_end,
)

print(
    f"Training:  {train_dataset.data.shape[0]:,} rows, "
    f"{train_dataset.data.index.min():%Y-%m-%d} to {train_dataset.data.index.max():%Y-%m-%d}"
)
print(
    f"Predict:   {predict_dataset.data.shape[0]:,} rows, "
    f"{predict_dataset.data.index.min():%Y-%m-%d} to {predict_dataset.data.index.max():%Y-%m-%d}"
)

# %% tags=["hide-input"]
# Quick look at the target variable
fig = cast(Any, train_dataset.data[["load"]].plot(title="Training period — energy load"))
fig.update_layout(yaxis_title="Load (MW)", xaxis_title="Time")
fig.show()

# %% [markdown]
# ## Configure the workflow
#
# [`ForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.ForecastingWorkflowConfig.html) bundles all settings — model type, horizons, quantiles,
# and feature columns — into a single object.  [`create_forecasting_workflow`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.create_forecasting_workflow.html) turns
# it into a ready-to-use pipeline with preprocessing, training, and postprocessing.
#
# We pick **GBLinear** (gradient-boosted linear model) for its speed and
# ability to extrapolate beyond training data.

# %%
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster

workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="quickstart_gblinear",
        model="gblinear",
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=[Q(0.5), Q(0.1), Q(0.9)],
        target_column="load",
        # Weather features available in the Liander dataset
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        wind_speed_column="wind_speed_10m",
        radiation_column="shortwave_radiation",
        pressure_column="surface_pressure",
        verbosity=0,
        mlflow_storage=None,
        gblinear_hyperparams=GBLinearForecaster.HyperParams(n_steps=50),
    )
)

# %% [markdown]
# ## Train the model
#
# `workflow.fit()` runs the full pipeline: feature engineering, data validation,
# model training, and evaluation on a held-out test split.

# %%
result = workflow.fit(train_dataset)

if result is not None:
    print("Training metrics:")
    print(result.metrics_full.to_dataframe())

    if result.metrics_test is not None:
        print("\nTest-set metrics:")
        print(result.metrics_test.to_dataframe())

# %% tags=["remove-cell"]
assert result is not None, "Training should produce a result"
assert result.metrics_full is not None, "Full metrics should be present"

# %% [markdown]
# ## Generate forecasts
#
# The trained workflow produces a [`ForecastDataset`](https://openstef.github.io/openstef/api/generated/openstef_core.datasets.ForecastDataset.html) with point predictions and
# quantile bands.  The P10-P90 interval covers 80 % of expected outcomes.
# To improve the reliability of these quantile estimates, see
# {doc}`/tutorials/quantile_calibration`.

# %%
from openstef_core.datasets import ForecastDataset

forecast: ForecastDataset = workflow.predict(predict_dataset, forecast_start=train_end)

print(f"Forecast rows: {len(forecast.data)}")
print(f"Quantiles:     {forecast.quantiles}")
forecast.data.tail()

# %% tags=["remove-cell"]
assert len(forecast.data) > 100, f"Expected >100 forecast rows, got {len(forecast.data)}"
assert forecast.quantiles is not None, "Quantile data should be present"

# %% [markdown]
# ## Visualize the results
#
# [`ForecastTimeSeriesPlotter`](https://openstef.github.io/openstef/api/generated/openstef_beam.analysis.plots.ForecastTimeSeriesPlotter.html) overlays measurements and predictions with shaded
# confidence bands in a single interactive chart.

# %% tags=["hide-input"]
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=predict_dataset.data["load"].loc[train_end:])
    .add_model(
        model_name="GBLinear",
        forecast=forecast.median_series,
        quantiles=forecast.quantiles_data,
    )
    .plot()
)
fig.update_layout(
    title="Forecast vs actuals",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=500,
)
fig.show()

# %% [markdown]
# ## Next steps
#
# - {doc}`/tutorials/backtesting_quickstart` — evaluate how this model performs on
#   historical data with realistic temporal constraints.
# - {doc}`/tutorials/custom_pipeline` — build a model from individual transforms when
#   presets don't cover your use case.
