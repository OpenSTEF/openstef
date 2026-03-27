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

# %%
# --- Setup: Logging and Display Configuration ---
# Configure logging to see training progress and plotly to render as PNG for VS Code compatibility
import logging

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

# %%
# Download dataset from HuggingFace Hub
# The dataset is stored as parquet files for efficient loading
from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

from openstef_core.base_model import Path

repo_id = "OpenSTEF/liander2024-energy-forecasting-benchmark"  # Public benchmark dataset
local_dir = Path("./liander_dataset")
target = "mv_feeder/OS Gorredijk"  # Specific installation to focus on

# Download required files: load measurements, weather, prices, and profiles
files_to_download = [
    f"load_measurements/{target}.parquet",  # Energy consumption data
    f"weather_forecasts_versioned/{target}.parquet",  # Weather features
    "EPEX.parquet",  # Electricity prices (optional feature)
    "profiles.parquet",  # Standard load profiles (optional feature)
]

for filename in files_to_download:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False
    )  # pyright: ignore[reportCallIssue]
    print(f"✓ {filename} downloaded")

print("\n✅ All files downloaded successfully!")

# %%
# Load datasets using OpenSTEF's VersionedTimeSeriesDataset
# This class handles versioned data where each value has an "available_at" timestamp
from openstef_core.datasets import VersionedTimeSeriesDataset

# Load each data source from parquet files
load_dataset = VersionedTimeSeriesDataset.read_parquet(local_dir / f"load_measurements/{target}.parquet")
weather_dataset = VersionedTimeSeriesDataset.read_parquet(local_dir / f"weather_forecasts_versioned/{target}.parquet")
epex_dataset = VersionedTimeSeriesDataset.read_parquet(local_dir / "EPEX.parquet")
profiles_dataset = VersionedTimeSeriesDataset.read_parquet(local_dir / "profiles.parquet")

# Combine all datasets using left join (keep all load timestamps, match features where available)
# select_version() materializes the lazy dataset into a concrete TimeSeriesDataset
dataset = VersionedTimeSeriesDataset.concat(
    [load_dataset, weather_dataset, epex_dataset, profiles_dataset],
    mode="left",  # Left join keeps all timestamps from the first dataset
).select_version()

# Preview the combined dataset
print(f"Dataset shape: {dataset.data.shape}")
print(f"Date range: {dataset.data.index.min()} to {dataset.data.index.max()}")
dataset.data.head()

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
fig = train_dataset.data[["load"]].plot(title="Training Data: Energy Load over Time")
fig.update_layout(yaxis_title="Load (MW)", xaxis_title="Time")
fig.show()

# %% [markdown]
# ## Define a base config with inline search space
#
#

# %%
from openstef_core.param_ranges import FloatRange, IntRange
from openstef_core.types import Q
from openstef_models.utils.tuning import HyperparameterTuner

# %% [markdown]
# ## Inspect the resolved search space
#

# %%

# Get the search space from the hyperparams instance (resolve fills None bounds from class-level defaults).
resolved_space = config.xgboost_hyperparams.get_search_space()

print("Resolved search space:")
for name, param in resolved_space.items():
    if isinstance(param, (FloatRange, IntRange)):
        scale = "  [log]" if param.log else ""
        print(f"  {name:25s}: {type(param).__name__}  [{param.low} — {param.high}]{scale}")
    else:
        print(f"  {name:25s}: CategoricalRange  {param.choices}")

# %% [markdown]
# ## Run the Optuna study with `HyperparameterTuner`

# %%
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress per-trial logs

tuner = HyperparameterTuner(
    config=config,
    train_dataset=train_dataset,
    create_workflow=create_forecasting_workflow,
    target_quantile=Q(0.5),
    metric_name="R2",
    n_trials=20,
    seed=42,
)
tuning_result = tuner.fit_with_tuning()

print(f"Study complete: {len(tuning_result.study.trials)} trials")
print(f"Best value: {tuning_result.study.best_value:.4f}")
print(f"Best params: {tuning_result.study.best_params}")


# %%
# The best config is already applied inside fit_with_tuning.
# Here we inspect which hyperparameters were tuned vs kept at their default.
print("Final XGBoost hyperparameters (tuned values marked):")
final_hp = tuning_result.workflow.model.forecaster.hyperparams
baseline_hp = config.xgboost_hyperparams
best_params = tuning_result.study.best_params

for field in type(final_hp).model_fields:
    value = getattr(final_hp, field)
    baseline = getattr(baseline_hp, field)
    marker = " <- tuned" if field in best_params else ""
    print(f"  {field:25s}: {value}{marker}")


# %% [markdown]
# ## Full-set training metrics
#
# `fit_with_tuning()` trains the final model on the full training set with the best
# hyperparameters.  The fit result is available as `tuning_result.fit_result`.
#

# %%
print("Final model already trained by fit_with_tuning!")
print("Full-set metrics (tuned model):")
print(tuning_result.fit_result.metrics_full.to_dataframe())


# %% [markdown]
# ## Inspect the study and forecast
#
# 1. How did $R^2$ improved over trials?
# 2. Which parameters had the most impact?
# 3. Final tuned model predictions on the held-out forecast window.
#

# %%
from optuna.visualization import plot_optimization_history, plot_param_importances

study = tuning_result.study

# How the best score evolved over trials
fig = plot_optimization_history(study)
fig.update_layout(title="Optimization History: R² over Trials")
fig.show()

# Which hyperparameters mattered most (requires ≥ ~20 trials for reliable ranking)
fig2 = plot_param_importances(study)
fig2.update_layout(title="Hyperparameter Importances")
fig2.show()


# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

forecast = tuning_result.workflow.predict(forecast_dataset)

fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=forecast_dataset.data["load"])
    .add_model(
        model_name="XGBoost (tuned)",
        forecast=forecast.median_series,
        quantiles=forecast.quantiles_data,
    )
    .plot()
)
fig.update_layout(
    title="Tuned XGBoost Forecast vs Actual",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=500,
)
fig.show()


# %%
