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
# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %% [markdown]
# # Guide Figure Generation
#
# Generates static images used by user-guide pages in the documentation.
# Uses real Liander data and OpenSTEF's own plotters. Run by hand; outputs go
# to ``docs/source/images/guides/``. The figures are not regenerated during
# normal docs builds.

# %% tags=["remove-cell"]
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images" / "guides"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Figure 1: Quantile fan chart (probabilistic forecast output)
#
# Trains a real GBLinear model with seven quantiles and shows the full
# predictive distribution as nested confidence bands plus the median line.
# This is the visual answer to "what does a probabilistic forecast look
# like".

# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster

dataset = load_liander_dataset()

train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
train_end = train_start + timedelta(days=45)
forecast_end = train_end + timedelta(days=5)

train_dataset = dataset.filter_by_range(start=train_start, end=train_end)
predict_dataset = dataset.filter_by_range(
    start=train_end - timedelta(days=14),
    end=forecast_end,
)

# Seven quantiles so the fan chart has three visible bands.
quantiles = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="docs_guide_fan",
        model="gblinear",
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=quantiles,
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

plotter = ForecastTimeSeriesPlotter()
plotter.add_measurements(measurements=predict_dataset.data["load"].loc[train_end:])
plotter.add_model(
    model_name="GBLinear",
    forecast=forecast.median_series,
    quantiles=forecast.quantiles_data,
)
fig = plotter.plot()
fig.update_layout(
    title="Probabilistic forecast: median plus seven quantile bands",
    xaxis_title="Time",
    yaxis_title="Load (MW)",
    height=400,
    width=900,
)
fig.write_image(str(OUTPUT_DIR / "probabilistic_fan_chart.svg"))
print(f"Saved: {OUTPUT_DIR / 'probabilistic_fan_chart.svg'}")

# %% [markdown]
# ## Figure 2: Calibration plot (raw model)
#
# For a well-calibrated probabilistic forecast, a prediction at the p-th
# quantile should have exactly p fraction of actual observations below it.
# This figure plots the empirical (observed) probability against the
# nominal (forecast) probability for each quantile of the model above.
# Points on the diagonal indicate perfect calibration; systematic
# deviations indicate over- or under-confidence.

# %%
from openstef_beam.analysis.plots import QuantileProbabilityPlotter
from openstef_beam.metrics.metrics_probabilistic import observed_probability

# Predict on a longer evaluation window so the empirical probabilities are
# stable. Use the post-training section of predict_dataset.
eval_dataset = dataset.filter_by_range(
    start=train_end,
    end=train_end + timedelta(days=30),
)
eval_forecast = workflow.predict(
    dataset.filter_by_range(
        start=train_end - timedelta(days=14),
        end=train_end + timedelta(days=30),
    ),
    forecast_start=train_end,
)

# Align forecasts to actuals.
actuals = eval_dataset.data["load"].dropna()
forecast_df = eval_forecast.quantiles_data.dropna()
common_index = actuals.index.intersection(forecast_df.index)
actuals = actuals.loc[common_index].to_numpy()

observed = []
for q in quantiles:
    col = q.format()  # e.g., "quantile_P10"
    preds = forecast_df.loc[common_index, col].to_numpy()
    observed.append(Q(observed_probability(actuals, preds)))

plotter = QuantileProbabilityPlotter()
plotter.add_model("GBLinear (raw)", quantiles, observed)
fig = plotter.plot(title="Calibration plot: forecasted vs observed probability")
fig.update_layout(
    height=450,
    width=600,
)
fig.write_image(str(OUTPUT_DIR / "calibration_plot.svg"))
print(f"Saved: {OUTPUT_DIR / 'calibration_plot.svg'}")

# %% [markdown]
# ## Figure 3: Flatline detection and fallback
#
# Shows a synthetic flatline scenario: a real load signal where the last 36
# hours are stuck at the last reported value (sensor failure). The primary
# model, trained on healthy history, would happily extrapolate the normal
# daily pattern (a confidently wrong forecast). The FlatlinerForecaster
# fallback recognizes the condition and emits zero across all horizons.

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cccccc",
        "font.size": 10,
    }
)

# Color palette aligned with ForecastTimeSeriesPlotter.
COLOR_MEASURED = "#d62728"  # Red: realized measurements
COLOR_PRIMARY = "#1f77b4"   # Blue: primary model prediction
COLOR_FALLBACK = "#ff7f0e"  # Orange: fallback prediction
COLOR_FLATLINE = "#f4d35e"  # Yellow shading: flatline region

# Take 7 days of real load data, then synthesize a flatline at the end.
flatline_start = datetime.fromisoformat("2024-04-08T12:00:00Z")
window_start = flatline_start - timedelta(days=5)
window_end = flatline_start + timedelta(days=2)
load = dataset.data.loc[window_start:window_end, "load"].astype(float) / 1e6  # W to MW

# Synthesize the flatline: last 36 hours pinned to the value just before.
pin_time = flatline_start
pin_value = float(load.loc[:pin_time].iloc[-1])
load_with_flatline = load.copy()
load_with_flatline.loc[pin_time:] = pin_value

# Primary model prediction: extrapolates the regular daily pattern by reusing
# the previous week's profile. The point is illustrative; the real primary
# model behavior is similar.
weekly_lag = timedelta(days=1)  # use yesterday's same-hour as a stand-in
primary_forecast = load.shift(freq=weekly_lag).loc[pin_time:window_end]

# Fallback prediction: constant zero.
fallback_forecast = pd.Series(0.0, index=primary_forecast.index)

fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

# Shade the flatline detection region.
ax.axvspan(
    pin_time, window_end,
    alpha=0.18, color=COLOR_FLATLINE, zorder=0,
    label="Flatline detected",
)

# Measured load (with the synthetic flatline).
ax.plot(
    load_with_flatline.index, load_with_flatline.values,
    color=COLOR_MEASURED, linewidth=1.6, zorder=3,
    label="Measured load (sensor stuck)",
)

# Primary model: confidently wrong.
ax.plot(
    primary_forecast.index, primary_forecast.values,
    color=COLOR_PRIMARY, linewidth=1.4, linestyle="--", zorder=2,
    label="Primary model forecast (unaware)",
)

# Fallback model: explicit zero.
ax.plot(
    fallback_forecast.index, fallback_forecast.values,
    color=COLOR_FALLBACK, linewidth=1.8, zorder=4,
    label="FlatlinerForecaster output",
)

# Vertical marker at the detection time.
ax.axvline(pin_time, color="#666666", linewidth=1.0, linestyle=":", zorder=1)
ax.text(
    pin_time, ax.get_ylim()[1] * 0.95,
    "  meter freeze",
    ha="left", va="top", fontsize=9, color="#666666",
)

ax.set_xlim(window_start, window_end)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.set_xlabel("Datetime [UTC]")
ax.set_ylabel("Load [MW]")
ax.set_title(
    "Flatline scenario: primary model versus FlatlinerForecaster",
    fontsize=11, fontweight="bold",
)
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "flatline_detection.svg"))
plt.close(fig)
print(f"Saved: {OUTPUT_DIR / 'flatline_detection.svg'}")
