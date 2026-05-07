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

# %%
# --- Setup: Logging and Display Configuration ---
# Configure logging and display settings for the notebook
from typing import Literal

from openstef_core.testing import configure_notebook_display, setup_notebook_logging

configure_notebook_display()
logger = setup_notebook_logging(__name__)

# %%
# Download and combine the Liander benchmark dataset into a single TimeSeriesDataset.
from openstef_core.testing import load_liander_dataset, prepare_tutorial_datasets

dataset = load_liander_dataset()

print(f"Dataset shape: {dataset.data.shape}")
print(f"Date range: {dataset.data.index.min()} to {dataset.data.index.max()}")
dataset.data.head()

# %%
# Split the dataset into training (90 days) and forecast (14 days) periods.
train_dataset, forecast_dataset = prepare_tutorial_datasets()

# %%
# Visualize the training data
# The plot shows the 'load' column (energy consumption in MW) over time
fig = train_dataset.data[["load"]].plot(title="Training Data: Energy Load over Time")
fig.update_layout(yaxis_title="Load (MW)", xaxis_title="Time")  # type: ignore[union-attr]  # plotly Figure
fig.show()  # type: ignore[union-attr]

# %% [markdown]
# ## Define a base config with inline search space
#
# Override default hyperparameters with `TuningRange(tune=True)` to mark them for tuning.
# Any parameter left as a plain value keeps its default during tuning.

# %%
from openstef_beam.evaluation.metric_providers import ObservedProbabilityProvider, RMAEProvider
from openstef_core.mixins.param_ranges import FloatRange, IntRange
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.optuna import HyperparameterTuner
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

config = ForecastingWorkflowConfig(
    model_id="tuning_demo",
    model="xgboost",
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
    # Hyperparameters to tune
    xgboost_hyperparams=XGBoostHyperParams(
        learning_rate=FloatRange(0.01, 0.3, log=True, tune=True),  # pyright: ignore[reportCallIssue]  # ranges accepted at runtime via Annotated
        n_estimators=IntRange(50, 500, tune=True),
        max_depth=IntRange(3, 10, tune=True),
        subsample=FloatRange(0.5, 1.0, tune=True),
        colsample_bytree=FloatRange(0.5, 1.0, tune=True),
    ),
    evaluation_metrics=[RMAEProvider(), ObservedProbabilityProvider()],
    mlflow_storage=None,  # Disable MLFlow tune to avoid reusing models between trials.
)

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
    metric_name="rMAE",
    direction="minimize",
    n_trials=20,
    seed=42,
)
tuning_result = tuner.fit_with_tuning()

print(f"Study complete: {len(tuning_result.study.trials)} trials")
print(f"Best value: {tuning_result.study.best_value:.4f}")
print(f"Best params: {tuning_result.study.best_params}")


# %%
# Inspect which hyperparameters were tuned vs kept at their default.
best_config = tuning_result.best_config  # type: ignore[union-attr]  # known to be ForecastingWorkflowConfig
print("Final XGBoost hyperparameters (tuned values marked):")
final_hp = best_config.xgboost_hyperparams
baseline_hp = config.xgboost_hyperparams
best_params = tuning_result.study.best_params

for field in type(final_hp).model_fields:
    value = getattr(final_hp, field)
    baseline = getattr(baseline_hp, field)
    marker: Literal[" <- tuned", ""] = " <- tuned" if field in best_params else ""
    print(f"  {field:25s}: {value}{marker}")


# %% [markdown]
# ## The fitted workflow
#
# `fit_with_tuning()` already trains a final workflow on the full training set using the best
# hyperparameters — no separate fit step is needed. The result is in `tuning_result.workflow`.
#

# %%
workflow = tuning_result.workflow

# %% [markdown]
# ## Inspect the study and forecast
#
# 1. How did $rMAE$ improve over trials?
# 2. Which parameters had the most impact?
# 3. Final tuned model predictions on the held-out forecast window.
#

# %%
from optuna.visualization import plot_optimization_history, plot_param_importances

study = tuning_result.study

# How the best score evolved over trials
fig = plot_optimization_history(study)
fig.update_layout(title="Optimization History: rMAE over Trials")
fig.show()

# Which hyperparameters mattered most (requires ≥ ~20 trials for reliable ranking)
fig2 = plot_param_importances(study)
fig2.update_layout(title="Hyperparameter Importances")
fig2.show()


# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

forecast = workflow.predict(forecast_dataset)

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
