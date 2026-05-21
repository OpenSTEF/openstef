# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # Concept Figure Generation
#
# This script generates static images used by concept pages in the documentation.
# It uses real data from the Liander dataset and actual OpenSTEF model predictions.

# %% tags=["remove-cell"]
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from pathlib import Path

# Output directory (relative to this file's location during build)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images" / "concepts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Figure 1: Weekly Load Profile
#
# Shows one week of real substation load with clear daily and weekly patterns,
# using OpenSTEF's ForecastTimeSeriesPlotter.

# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.testing import load_liander_dataset

dataset = load_liander_dataset()

# Select a representative week (mid-February, includes weekday/weekend)
week_start = "2024-02-12"
week_end = "2024-02-19"
week_data = dataset.data.loc[week_start:week_end, "load"]

plotter = ForecastTimeSeriesPlotter()
plotter.add_measurements(measurements=week_data)
fig = plotter.plot()
fig.update_layout(
    title="Substation Load Profile: One Week at 15-Minute Resolution",
    xaxis_title="Time",
    yaxis_title="Load (MW)",
    height=400,
    width=900,
)
fig.write_image(str(OUTPUT_DIR / "weekly_load_profile.svg"))
print(f"Saved: {OUTPUT_DIR / 'weekly_load_profile.svg'}")

# %% [markdown]
# ## Figure 2: Forecast with Confidence Bands
#
# Trains a real GBLinear model and plots its quantile predictions using
# OpenSTEF's ForecastTimeSeriesPlotter.

# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster

# Train on 45 days, predict 7 days ahead
train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
train_end = train_start + timedelta(days=45)
forecast_end = train_end + timedelta(days=7)

train_dataset = dataset.filter_by_range(start=train_start, end=train_end)
predict_dataset = dataset.filter_by_range(
    start=train_end - timedelta(days=14),
    end=forecast_end,
)

workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="docs_figure",
        model="gblinear",
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=[Q(0.5), Q(0.1), Q(0.9)],
        target_column="load",
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

workflow.fit(train_dataset)
forecast = workflow.predict(predict_dataset, forecast_start=train_end)

# Use OpenSTEF's ForecastTimeSeriesPlotter
plotter = ForecastTimeSeriesPlotter()
plotter.add_measurements(measurements=predict_dataset.data["load"].loc[train_end:])
plotter.add_model(
    model_name="GBLinear",
    forecast=forecast.median_series,
    quantiles=forecast.quantiles_data,
)

fig = plotter.plot()
fig.update_layout(
    title="7-Day Forecast with Confidence Bands (GBLinear, 36h horizon)",
    xaxis_title="Time",
    yaxis_title="Load (MW)",
    height=400,
    width=900,
)
fig.write_image(str(OUTPUT_DIR / "forecast_confidence_bands.svg"))
print(f"Saved: {OUTPUT_DIR / 'forecast_confidence_bands.svg'}")

# %% [markdown]
# ## Figure 3: Aggregation Level Effects
#
# Compares a single solar park with clear curtailment (on/off switching during
# production hours) against a large transformer over the same days.

# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Solar park near Rotterdam has clear on/off curtailment patterns
solar_dataset = load_liander_dataset(
    target="solar_park/Within 20 kilometers of Rotterdam_normalized"
)
transformer_dataset = load_liander_dataset(target="transformer/OS Amsterdam Hemweg")

# May 14-17: includes days with 6 and 8 on/off switches (clear curtailment)
agg_start = "2024-05-14"
agg_end = "2024-05-18"
solar_days = solar_dataset.data.loc[agg_start:agg_end, "load"]
transformer_days = transformer_dataset.data.loc[agg_start:agg_end, "load"] / 1e6  # W to MW

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Single Solar Park (normalized) - on/off curtailment visible",
        "Transformer OS Amsterdam Hemweg (MW)",
    ),
)

fig.add_trace(
    go.Scatter(x=solar_days.index, y=solar_days.values, mode="lines",
               line={"color": "blue", "width": 1}, name="Solar park"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=transformer_days.index, y=transformer_days.values, mode="lines",
               line={"color": "red", "width": 1}, name="Transformer"),
    row=2, col=1,
)

fig.update_layout(
    title="Aggregation Level: Low vs High",
    height=500,
    width=900,
    showlegend=False,
)
fig.update_yaxes(title_text="Generation (norm.)", row=1, col=1)
fig.update_yaxes(title_text="Load (MW)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)

fig.write_image(str(OUTPUT_DIR / "aggregation_comparison.svg"))
print(f"Saved: {OUTPUT_DIR / 'aggregation_comparison.svg'}")
