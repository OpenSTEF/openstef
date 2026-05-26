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


# %% tags=["remove-cell"]
import warnings

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
        "lightgbm",
    ),
)

# %% [markdown]
# # Ensemble Forecasting
#
# OpenSTEF supports ensemble models that combine multiple base forecasters
# into a single prediction.  A **combiner** learns which base model performs
# best under different conditions and weights their outputs accordingly.
#
# **What you'll learn:**
#
# - Why combining tree-based and linear models improves forecasts
# - How to configure and train an ensemble with [`EnsembleForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_meta.presets.EnsembleForecastingWorkflowConfig.html)
# - How to inspect combiner behavior (which model does it prefer?)
# - How ensemble predictions compare to individual base models
#
# ```{note}
# This tutorial uses a small dataset for fast execution.
# See `examples/benchmarks/` for production-scale ensemble runs.
# ```
#
# **Key API references:**
# [`EnsembleForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_meta.presets.EnsembleForecastingWorkflowConfig.html)
# · [`create_ensemble_forecasting_workflow`](https://openstef.github.io/openstef/api/generated/openstef_meta.presets.create_ensemble_forecasting_workflow.html)
# · [`ForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.ForecastingWorkflowConfig.html)

# %% [markdown]
# ## Why ensemble forecasting?
#
# Different model types have complementary strengths:
#
# | Model | Strengths | Weaknesses |
# |-------|-----------|------------|
# | **Tree-based** (LightGBM, XGBoost) | Captures complex non-linear patterns, handles feature interactions well | Poor extrapolation — struggles with unseen peaks, seasonal shifts, or values outside training range |
# | **Linear** (GBLinear) | Extrapolates naturally to new ranges, captures seasonal/solar trends | Cannot model non-linear interactions |
#
# In energy forecasting, load peaks during extreme weather or seasonal
# transitions often fall outside the training distribution.  A tree-based
# model underestimates these peaks while a linear model captures the trend
# but misses finer patterns.  An **ensemble** combines both: the combiner
# learns *when* each model is more reliable and weights accordingly.
#
# ## How it works
#
# An ensemble workflow has three layers:
#
# 1. **Common preprocessing** — shared feature engineering (lags, holidays,
#    cyclic features, scaling) applied once to raw data.
# 2. **Base forecasters** — multiple models each trained on the preprocessed
#    data, with optional per-model transforms (e.g. GBLinear gets fewer lags
#    to avoid collinearity).
# 3. **Combiner** — learns to aggregate base forecaster outputs.  Two modes:
#    - *Learned weights*: a classifier predicts which base model will perform
#      best for each sample, then weights predictions accordingly.
#    - *Stacking*: a meta-regressor trained on base model outputs per quantile.

# %% [markdown]
# ## Load the dataset

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q

dataset = load_liander_dataset()

train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
train_end = train_start + timedelta(days=45)
forecast_end = train_end + timedelta(days=7)

train_dataset = dataset.filter_by_range(start=train_start, end=train_end)
predict_dataset = dataset.filter_by_range(
    start=train_end - timedelta(days=14),
    end=forecast_end,
)

print(f"Training:  {train_dataset.data.shape[0]:,} rows")
print(f"Predict:   {predict_dataset.data.shape[0]:,} rows")

# %% [markdown]
# ## Configure the ensemble
#
# [`EnsembleForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_meta.presets.EnsembleForecastingWorkflowConfig.html) sets up the full pipeline.
# Key parameters:
#
# - `base_models` — which forecasters to include
# - `ensemble_type` — how to combine them (`"learned_weights"` or `"stacking"`)
# - `combiner_model` — the algorithm used by the combiner

# %%
from openstef_meta.presets import EnsembleForecastingWorkflowConfig, create_ensemble_forecasting_workflow

ensemble_config = EnsembleForecastingWorkflowConfig(
    model_id="ensemble_demo",
    # Ensemble architecture
    ensemble_type="learned_weights",
    base_models=["lgbm", "gblinear"],
    combiner_model="lgbm",
    # Forecast settings
    horizons=[LeadTime.from_string("PT36H")],
    quantiles=[Q(0.5), Q(0.1), Q(0.9)],
    # Data columns
    target_column="load",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    wind_speed_column="wind_speed_10m",
    radiation_column="shortwave_radiation",
    pressure_column="surface_pressure",
    # Disable MLFlow for tutorial
    mlflow_storage=None,
)

print(f"Base models:    {list(ensemble_config.base_models)}")
print(f"Ensemble type:  {ensemble_config.ensemble_type}")
print(f"Combiner:       {ensemble_config.combiner_model}")

# %% [markdown]
# ## Create and train the ensemble workflow

# %%
workflow = create_ensemble_forecasting_workflow(ensemble_config)
fit_result = workflow.fit(train_dataset)

print("Ensemble trained successfully")
print("\nPer-model validation R2:")
for name, child_result in fit_result.component_fit_results.items():
    r2 = child_result.metrics_val.get_metric(quantile=Q(0.5), metric_name="R2")
    print(f"  {name:12s}: {r2:.4f}")

# Get combiner (ensemble-level) R2
ensemble_r2 = fit_result.metrics_val.get_metric(quantile=Q(0.5), metric_name="R2")
print(f"  {'ensemble':12s}: {ensemble_r2:.4f}")

# %% tags=["remove-cell"]
assert ensemble_r2 is not None and ensemble_r2 > 0.0, f"Expected positive R2, got {ensemble_r2}"

# %% [markdown]
# ## Generate forecasts

# %%
forecast = workflow.predict(predict_dataset, forecast_start=train_end)

print(f"Forecast rows: {len(forecast.data):,}")
print(f"Quantiles: {[float(q) for q in forecast.quantiles]}")

# %% tags=["remove-cell"]
assert len(forecast.data) > 100, f"Expected >100 forecast rows, got {len(forecast.data)}"

# %% [markdown]
# ## Compare: ensemble vs individual base models
#
# To show the benefit of ensembling, let's also train each base model
# individually and compare their forecasts.

# %%
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

individual_forecasts = {}
for model_type in ["lgbm", "gblinear"]:
    config = ForecastingWorkflowConfig(
        model_id=f"{model_type}_solo",
        model=model_type,
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=[Q(0.5), Q(0.1), Q(0.9)],
        target_column="load",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        wind_speed_column="wind_speed_10m",
        radiation_column="shortwave_radiation",
        pressure_column="surface_pressure",
        mlflow_storage=None,
        verbosity=0,
    )
    wf = create_forecasting_workflow(config)
    wf.fit(train_dataset)
    individual_forecasts[model_type] = wf.predict(predict_dataset, forecast_start=train_end)

# %% [markdown]
# ## Visualize the comparison

# %% tags=["hide-input"]
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

plotter = ForecastTimeSeriesPlotter()
plotter.add_measurements(measurements=predict_dataset.data["load"].loc[train_end:])

# Add individual models
for name, fc in individual_forecasts.items():
    plotter.add_model(model_name=name, forecast=fc.median_series, quantiles=fc.quantiles_data)

# Add ensemble
plotter.add_model(model_name="Ensemble", forecast=forecast.median_series, quantiles=forecast.quantiles_data)

fig = plotter.plot()
fig.update_layout(
    title="Ensemble vs Individual Models",
    xaxis_title="Time",
    yaxis_title="MW",
    height=450,
)
fig.show()

# %% [markdown]
# ## Combiner insights
#
# The learned-weights combiner trains a classifier that predicts — for each
# timestep — which base model will be most accurate.  It then uses those
# predicted probabilities as mixing weights.
#
# We can inspect this behavior at two levels:
#
# 1. **Global feature importances** — which input signals the classifier
#    relies on most when deciding between models.
# 2. **Per-timestamp selection weights** — the actual mixing probabilities
#    assigned to each model during forecasting.

# %%
import plotly.graph_objects as go

ensemble_model = workflow.model
combiner = ensemble_model.combiner

# Global importances: shows how much the classifier attends to each base
# model's prediction value when deciding which model to trust.
importances = combiner.feature_importances
print("Combiner feature importances (per quantile):")
print(importances.to_string())

# %% [markdown]
# With two base models, the feature importance tells us how much the
# combiner's internal classifier *uses* each model's prediction to decide
# who should contribute more.  A higher importance means the classifier pays
# more attention to that model's output when making the selection decision.
#
# More informative is the **per-timestamp weight** — the actual probability
# the combiner assigns to each model at each point in time during forecasting.

# %% tags=["hide-input"]
# Reproduce the internal flow to extract per-timestamp weights
ensemble_dataset = ensemble_model._predict_forecasters(predict_dataset, forecast_start=train_end)
base_preds = ensemble_dataset.get_base_predictions_for_quantile(Q(0.5))
input_data = base_preds.input_data()
weights = combiner._predict_weights(input_data, Q(0.5))

fig = go.Figure()
for col in weights.columns:
    fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", name=col, stackgroup="one"))
fig.update_layout(
    title="Combiner Model Selection Weights Over Time (q50)",
    xaxis_title="Time",
    yaxis_title="Weight (probability)",
    yaxis_range=[0, 1],
    height=350,
    legend_title="Base model",
)
fig.show()

# %% [markdown]
# The stacked area chart reveals *when* the combiner trusts each model.
# Typical patterns:
#
# - **gblinear dominates at peaks/troughs** — its linear extrapolation
#   handles values near or beyond the training range better.
# - **lgbm dominates during stable periods** — its tree-based flexibility
#   captures non-linear patterns (time-of-day effects, weather interactions)
#   more accurately when extrapolation is not needed.
#
# This adaptive selection is the core advantage of ensembling: neither model
# alone achieves the accuracy of the dynamically-weighted combination.

# %% [markdown]
# ## Metrics comparison
#
# Let's quantify the ensemble advantage with relative MAE (rMAE) on the
# forecast period.  rMAE normalizes the MAE by the range of actuals, making it
# easier to compare across datasets with different scales.  We use the
# implementation from `openstef_beam.metrics`.

# %%

from openstef_beam.metrics import rmae

actuals = predict_dataset.data["load"].loc[train_end:forecast_end]

models = {"lgbm": individual_forecasts["lgbm"], "gblinear": individual_forecasts["gblinear"], "Ensemble": forecast}

print(f"{'Model':<12} {'rMAE':>8}")
print(f"{'':-<12} {'':-^8}")
for name, fc in models.items():
    common = actuals.index.intersection(fc.median_series.index)
    print(f"{name:<12} {rmae(actuals.loc[common].to_numpy(), fc.median_series.loc[common].to_numpy()):>8.4f}")

# %% [markdown]
# The ensemble consistently achieves the lowest rMAE by combining the
# strengths of both models.  In production with longer training windows and
# more diverse base models (e.g. adding XGBoost or a neural forecaster),
# the improvement typically grows larger.  To validate ensemble gains over
# longer periods, run a full {doc}`backtesting_quickstart`.

# %% [markdown]
# ## Next steps
#
# - {doc}`hyperparameter_tuning_with_optuna` — tune each base model's
#   parameters before combining them.
# - {doc}`quantile_calibration` — calibrate the ensemble's uncertainty
#   estimates for more reliable confidence intervals.
