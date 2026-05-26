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

# Axis labels reused across figures.
AXIS_TIME = "Time"
AXIS_LOAD_MW = "Load (MW)"
AXIS_DATETIME_UTC = "Datetime [UTC]"
AXIS_LOAD_BRACKETS = "Load [MW]"

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
    xaxis_title=AXIS_TIME,
    yaxis_title=AXIS_LOAD_MW,
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
    xaxis_title=AXIS_TIME,
    yaxis_title=AXIS_LOAD_MW,
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
fig.update_yaxes(title_text=AXIS_LOAD_MW, row=2, col=1)
fig.update_xaxes(title_text=AXIS_TIME, row=2, col=1)

fig.write_image(str(OUTPUT_DIR / "aggregation_comparison.svg"))
print(f"Saved: {OUTPUT_DIR / 'aggregation_comparison.svg'}")

# %% [markdown]
# ## Figure 4: Backtesting Animation
#
# Animated GIF showing how BEAM's BacktestPipeline steps through time with
# periodic retraining and more frequent predictions. The animation illustrates
# the restricted horizon concept: at each point in time, the model can only
# see data that would have been available in production.

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Style matching ForecastTimeSeriesPlotter (plotly_white aesthetic)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#cccccc",
        "font.size": 10,
    }
)

# Colors aligned with ForecastTimeSeriesPlotter palette
COLOR_AVAILABLE = "#1f77b4"  # Plotly blue (model forecast color)
COLOR_FUTURE = "#d9d9d9"  # Light gray
COLOR_HORIZON = "#d62728"  # Plotly red (realized/measurements color)
COLOR_TRAIN_WINDOW = "#2ca02c"  # Green for training window shading
COLOR_FORECAST = "#ff7f0e"  # Orange for forecast line (positive, warm)
COLOR_PRED_WINDOW = "#fff3e0"  # Warm light orange (prediction region)
LINEWIDTH = 1.5  # Matches ForecastTimeSeriesPlotter.stroke_width

# Load 4 weeks of data (zoomed in compared to 6 weeks)
dataset = load_liander_dataset()
anim_start = datetime.fromisoformat("2024-02-01T00:00:00Z")
anim_end = anim_start + timedelta(days=28)
anim_data = dataset.data.loc[anim_start:anim_end, "load"] / 1e6  # W to MW

# Backtesting parameters (matching BEAM defaults)
train_interval = timedelta(days=7)
predict_interval = timedelta(hours=6)
prediction_horizon = timedelta(hours=36)  # Each prediction looks 36h ahead

# Determine event timestamps
sim_start = anim_start + timedelta(days=7)  # Need initial training data
sim_end = anim_end - prediction_horizon  # Stop before we run out of future

train_times = []
t = sim_start
while t <= sim_end:
    train_times.append(t)
    t += train_interval

predict_times = []
t = sim_start
while t <= sim_end:
    predict_times.append(t)
    t += predict_interval

# Create frames: every 3rd predict event for smoother animation
frame_step = 3
frame_predict_times = predict_times[::frame_step]

fig, ax = plt.subplots(figsize=(10, 4), dpi=120)


def draw_frame(frame_idx):
    """Draw a single animation frame."""
    ax.clear()
    current_time = frame_predict_times[frame_idx]

    # Determine training window: from start to most recent train event
    past_trains = [t for t in train_times if t <= current_time]
    train_window_start = anim_data.index[0]

    # Training window highlight (green shading behind trained data)
    if past_trains:
        ax.axvspan(
            train_window_start, past_trains[-1],
            alpha=0.12, color=COLOR_TRAIN_WINDOW, zorder=0,
        )

    # Prediction window (warm orange shading)
    pred_end = current_time + prediction_horizon
    ax.axvspan(current_time, pred_end, alpha=0.18, color=COLOR_FORECAST, zorder=0)

    # Full time series (faded future)
    ax.plot(
        anim_data.index, anim_data.values,
        color=COLOR_FUTURE, linewidth=LINEWIDTH * 0.6, zorder=1,
    )

    # Available data (what the model can see) - blue line
    visible = anim_data.loc[:current_time]
    ax.plot(
        visible.index, visible.values,
        color=COLOR_AVAILABLE, linewidth=LINEWIDTH, zorder=2,
    )

    # Forecast line (orange, within prediction window)
    forecast_data = anim_data.loc[current_time:pred_end]
    if len(forecast_data) > 1:
        # Simulate a forecast with small random offset for realism
        rng = np.random.default_rng(seed=frame_idx * 7)
        noise = rng.normal(0, 0.02, size=len(forecast_data))
        forecast_values = forecast_data.values + noise
        ax.plot(
            forecast_data.index, forecast_values,
            color=COLOR_FORECAST, linewidth=LINEWIDTH * 1.2, zorder=4,
        )

    # Current horizon line (red vertical)
    ax.axvline(
        current_time, color=COLOR_HORIZON,
        linewidth=LINEWIDTH * 1.3, linestyle="-", zorder=5, alpha=0.9,
    )

    # Formatting (plotly_white style)
    ax.set_xlim(anim_data.index[0], anim_data.index[-1])
    y_min, y_max = anim_data.min(), anim_data.max()
    y_pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.set_xlabel(AXIS_DATETIME_UTC)
    ax.set_ylabel(AXIS_LOAD_BRACKETS)
    ax.set_title("BEAM Backtesting: Event-Driven Simulation", fontsize=11, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_TRAIN_WINDOW, alpha=0.25, label="Training data"),
        Line2D([0], [0], color=COLOR_AVAILABLE, linewidth=LINEWIDTH, label="Available data"),
        Line2D([0], [0], color=COLOR_FUTURE, linewidth=1, label="Future (hidden)"),
        Line2D([0], [0], color=COLOR_FORECAST, linewidth=LINEWIDTH * 1.2, label="Forecast"),
        Line2D([0], [0], color=COLOR_HORIZON, linewidth=LINEWIDTH * 1.3, label="Current horizon"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.95)

    fig.tight_layout()


# Generate animation
anim = FuncAnimation(fig, draw_frame, frames=len(frame_predict_times), interval=200)
gif_path = OUTPUT_DIR / "backtesting_simulation.gif"
anim.save(str(gif_path), writer=PillowWriter(fps=5))
plt.close(fig)
print(f"Saved: {gif_path}")
