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

# %%
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# pyright: basic

# %% [markdown]
# # 🔮 Forecasting with OpenSTEF 4.0 Workflow Presets
#
# This tutorial demonstrates how to use **OpenSTEF 4.0** to create energy load forecasts
# using the **Workflow Presets** pattern. You'll learn how to:
#
# 1. **Load real-world energy data** from the Liander 2024 benchmark dataset
# 2. **Configure a forecasting workflow** with weather features and prediction quantiles
# 3. **Train a model** and inspect its performance
# 4. **Generate probabilistic forecasts** with confidence intervals
# 5. **Visualize results** and explain feature importance
#
# > **OpenSTEF** (Short-Term Energy Forecasting) is a modular library for creating
# > accurate energy forecasts in the power grid domain.

# %%
# --- Setup: Logging and Display Configuration ---
# Configure logging to see training progress and plotly to render as PNG for VS Code compatibility
import logging
from typing import Any, cast

import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "png"  # Use PNG for VS Code notebook compatibility

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("choreographer").setLevel(logging.ERROR)
logging.getLogger("kaleido").setLevel(logging.ERROR)
logging.getLogger("choreographer").disabled = True
logging.getLogger("kaleido").disabled = True

# %% [markdown]
# ## 📦 Step 1: Download the Dataset
#
# We'll use the **Liander 2024 Energy Forecasting Benchmark** dataset from HuggingFace Hub. This dataset contains:
# - **Load measurements** — historical energy consumption from various installations (mv feeders, transformers, etc.)
# - **Weather forecasts** — versioned weather predictions (temperature, radiation, wind, etc.)
# - **EPEX prices** — day-ahead electricity market prices
# - **Profiles** — typical daily/weekly load patterns

# %%
# Download and combine the Liander benchmark dataset into a single TimeSeriesDataset.
# See data.py for the reusable helper that handles download + loading + combining.
from data import load_liander_dataset

dataset = load_liander_dataset()

print(f"Dataset shape: {dataset.data.shape}")
print(f"Date range: {dataset.data.index.min()} to {dataset.data.index.max()}")
dataset.data.head()

# %% [markdown]
# ## ✂️ Step 3: Split Data into Training and Forecast Periods
#
# We'll use:
# - **90 days** of historical data for training
# - **14 days** as the forecast period (where we'll generate predictions)

# %%
# Define training and forecast time periods
from datetime import datetime, timedelta

# Training period: 90 days of historical data
train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
train_end = train_start + timedelta(days=90)

# Forecast period: 14 days after training (this is where we'll predict)
forecast_start = train_end
forecast_end = forecast_start + timedelta(days=14)

# Split the dataset using time-based filtering
train_dataset = dataset.filter_by_range(start=train_start, end=train_end)
forecast_dataset = dataset.filter_by_range(start=forecast_start, end=forecast_end)

print(f"📈 Training period: {train_start.date()} to {train_end.date()} ({len(train_dataset.data)} samples)")
print(f"🔮 Forecast period: {forecast_start.date()} to {forecast_end.date()} ({len(forecast_dataset.data)} samples)")

# %%
# Visualize the training data
# The plot shows the 'load' column (energy consumption in MW) over time
# cast() needed: pandas returns plotly Figure at runtime (backend="plotly") but typed as Axes
fig = cast(Any, train_dataset.data[["load"]].plot(title="Training Data: Energy Load over Time"))
fig.update_layout(yaxis_title="Load (MW)", xaxis_title="Time")
fig.show()

# %% [markdown]
# ## ⚙️ Step 4: Configure the Forecasting Workflow
#
# OpenSTEF uses a **ForecastingWorkflowConfig** to define all aspects of the forecasting pipeline:
# - **Model type** — `gblinear` (gradient boosted linear model) or `xgboost`
# - **Forecast horizons** — how far ahead to predict (e.g., 36 hours)
# - **Quantiles** — prediction intervals for probabilistic forecasts
# - **Feature columns** — which weather variables to use
#
# The **GBLinear** model is particularly good for energy forecasting because:
# 1. It can extrapolate beyond training data (important for rare events)
# 2. It provides interpretable feature importance
# 3. It's fast to train and predict

# %%
# Import workflow components
from openstef_core.types import LeadTime, Q  # LeadTime: forecast horizon, Q: quantile
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster

# Configure the forecasting workflow
workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        # Model identification
        model_id="gblinear_demo_v1",
        model="gblinear",  # Use gradient boosted linear model
        # Forecast settings
        horizons=[LeadTime.from_string("PT36H")],  # Predict up to 36 hours ahead
        quantiles=[Q(0.5), Q(0.1), Q(0.9)],  # Median + 80% prediction interval
        # Target column (what we're predicting)
        target_column="load",
        # Weather feature columns (from the dataset)
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        wind_speed_column="wind_speed_10m",
        radiation_column="shortwave_radiation",  # Solar radiation
        pressure_column="surface_pressure",
        # Training settings
        verbosity=1,  # Show progress during training
        mlflow_storage=None,  # Disable MLflow tracking for this demo
        # Model-specific hyperparameters
        gblinear_hyperparams=GBLinearForecaster.HyperParams(
            n_steps=50  # Number of boosting iterations
        ),
    )
)

print("✅ Workflow configured successfully!")

# %% [markdown]
# ## 🏋️ Step 5: Train the Model
#
# The workflow's `fit()` method handles the entire training pipeline:
# 1. **Preprocessing** — feature engineering, data validation, scaling
# 2. **Training** — fit the model on historical data
# 3. **Evaluation** — compute metrics on training data

# %%
# Train the model on historical data
logger.info("🏋️ Starting model training...")

result = workflow.fit(train_dataset)

# Display training metrics
if result is not None:
    logger.info("✅ Training complete!")
    print("\n📊 Training Evaluation Metrics:")
    print(result.metrics_full.to_dataframe())

    if result.metrics_test is not None:
        print("\n📊 Test Set Metrics (held-out validation):")
        print(result.metrics_test.to_dataframe())

# %% [markdown]
# ## 🔮 Step 6: Generate Forecasts
#
# Now we use the trained model to predict energy load for the next 14 days.
# The output is a **ForecastDataset** containing:
# - **Median prediction** (`quantile_P50`)
# - **Lower bound** (`quantile_P10`) — 10th percentile
# - **Upper bound** (`quantile_P90`) — 90th percentile

# %%
# Generate probabilistic forecasts for the forecast period
from openstef_core.datasets import ForecastDataset

logger.info("🔮 Generating forecasts...")
forecast: ForecastDataset = workflow.predict(forecast_dataset)

# Display forecast summary
print(f"\n📈 Forecast generated for {len(forecast.data)} timestamps")
print(f"📊 Quantiles: {forecast.quantiles}")
print("\n🔍 Last 5 forecast values:")
print(forecast.data.tail())

# %% [markdown]
# ## 📈 Step 7: Visualize Forecast Results
#
# OpenSTEF-BEAM provides **ForecastTimeSeriesPlotter** for beautiful interactive visualizations:
# - Actual measurements shown as a line
# - Forecast median shown as another line
# - Prediction intervals shown as shaded areas

# %%
# Create an interactive forecast visualization
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

fig = (
    ForecastTimeSeriesPlotter()
    # Add actual measurements (ground truth)
    .add_measurements(measurements=forecast_dataset.data["load"])
    # Add model predictions with confidence bands
    .add_model(
        model_name="GBLinear",
        forecast=forecast.median_series,  # P50 prediction
        quantiles=forecast.quantiles_data,  # P10-P90 confidence band
    )
    .plot()
)

# Update layout for better presentation
fig.update_layout(
    title="🔮 Energy Load Forecast vs Actual",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=500,
)
fig.show()

# %% [markdown]
# ## 🔍 Step 8: Explain Feature Importance
#
# Understanding **why** the model makes certain predictions is crucial for trust
# and debugging. GBLinear models provide clear feature importance rankings.

# %%
# Visualize feature importance using the ExplainableForecaster interface
from typing import cast

from openstef_models.explainability import ExplainableForecaster
from openstef_models.models.forecasting_model import ForecastingModel

# The GBLinear model implements ExplainableForecaster, providing feature importance
forecaster = cast(ForecastingModel, workflow.model).forecaster
explainable_model = cast(ExplainableForecaster, forecaster)

# Create an interactive treemap of feature importances
# Larger boxes = more important features
fig = explainable_model.plot_feature_importances()
fig.update_layout(title="🔍 Feature Importance Treemap")
fig.show()

# %% [markdown]
# ---
#
# ## 🎯 Summary
#
# In this tutorial, you learned how to:
#
# 1. ✅ **Load energy data** from the Liander 2024 benchmark dataset
# 2. ✅ **Configure a workflow** with `ForecastingWorkflowConfig`
# 3. ✅ **Train a GBLinear model** for probabilistic forecasting
# 4. ✅ **Generate forecasts** with confidence intervals
# 5. ✅ **Visualize results** and feature importance
#
# ### 🚀 Next Steps
#
# - Try different models: `"xgboost"` for more complex patterns
# - Experiment with more quantiles for narrower prediction intervals
# - Use the **backtesting notebook** to evaluate model performance systematically
# - Explore MLflow integration for experiment tracking
