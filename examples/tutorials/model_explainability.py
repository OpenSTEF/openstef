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


# %% tags=["remove-cell"]
import warnings
from typing import cast

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
# # Model Explainability
#
# Understand why a forecasting model makes the predictions it does, using
# feature importance scores and per-timestep SHAP contributions.
#
# **What you'll learn:**
#
# - Inspect global feature importance with an interactive treemap
# - Compute per-timestep feature contributions (SHAP values)
# - Visualize contributions with heatmaps, waterfall charts, and bar charts
#
# ```{note}
# This tutorial uses a small data slice for fast execution.
# See `examples/benchmarks/` for production-scale runs.
# ```
#
# **Key API references:**
# [`ExplainableForecaster`](https://openstef.github.io/openstef/api/generated/openstef_models.explainability.ExplainableForecaster.html)
# · [`ContributionsPlotter`](https://openstef.github.io/openstef/api/generated/openstef_models.explainability.ContributionsPlotter.html)
# · [`FeatureImportancePlotter`](https://openstef.github.io/openstef/api/generated/openstef_models.explainability.FeatureImportancePlotter.html)

# %% [markdown]
# ## Train a model
#
# We reuse the same setup as the {doc}`forecasting_quickstart` — train a GBLinear
# model on 45 days of Liander data.

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster

dataset = load_liander_dataset()

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
        model_id="explainability_gblinear",
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

result = workflow.fit(train_dataset)
print("Training complete.")
print(result.metrics_full.to_dataframe())

# %% tags=["remove-cell"]
assert result is not None, "Training should produce a result"

# %% [markdown]
# ## Feature importance
#
# Feature importance scores rank features by their overall impact on the model's
# predictions.  The [`FeatureImportancePlotter`](https://openstef.github.io/openstef/api/generated/openstef_models.explainability.FeatureImportancePlotter.html) treemap visualization groups features by magnitude — larger
# tiles represent more influential features.

# %% tags=["hide-input"]
from openstef_models.explainability import ExplainableForecaster
from openstef_models.models.forecasting_model import ForecastingModel

forecaster = cast(ForecastingModel, workflow.model).forecaster
explainable_model = cast(ExplainableForecaster, forecaster)

fig = explainable_model.plot_feature_importances()
fig.update_layout(title="Feature importance (treemap)", height=500)
fig.show()

# %% [markdown]
# ## Feature contributions
#
# While feature importance is a global summary, **feature contributions** explain
# individual predictions.  For each timestep, they decompose the prediction into
# additive terms: one per feature plus a bias.
#
# GBLinear models provide exact SHAP values, making this decomposition faithful
# to the model's internal logic.  Use [`ContributionsPlotter`](https://openstef.github.io/openstef/api/generated/openstef_models.explainability.ContributionsPlotter.html)
# to visualize contributions as heatmaps, bar charts, or waterfall charts.

# %%
from openstef_models.explainability import ContributionsPlotter

contributions = workflow.model.predict_contributions(predict_dataset, forecast_start=train_end)

print(f"Contributions shape: {contributions.data.shape}")
print(f"Features: {contributions.data.columns.tolist()[:5]} ... ({len(contributions.data.columns)} total)")

# %% tags=["remove-cell"]
assert contributions.data.shape[0] > 100, f"Expected >100 rows, got {contributions.data.shape[0]}"
assert "bias" in contributions.data.columns, "Contributions should include bias column"

# %% [markdown]
# ### Heatmap — contributions over time
#
# Each row is a feature, each column is a timestep.  Red cells indicate positive
# contributions (pushing the prediction up), blue cells indicate negative ones.
# The prediction line overlays the total.

# %% tags=["hide-input"]
fig = ContributionsPlotter.plot_heatmap(contributions, top_n=10, show_prediction=True)
fig.update_layout(height=500)
fig.show()

# %% [markdown]
# ### Bar chart — average feature impact
#
# Mean absolute contribution per feature, ranked from most to least impactful.
# This gives a complementary view to global importance — here you see which
# features actively moved predictions during the forecast window.  If certain
# features dominate unexpectedly, consider adjusting the pipeline via
# {doc}`custom_pipeline`.

# %% tags=["hide-input"]
fig = ContributionsPlotter.plot_bar(contributions, top_n=12)
fig.update_layout(title="Mean absolute contribution per feature", height=450)
fig.show()

# %% [markdown]
# ### Waterfall — single timestep decomposition
#
# The waterfall chart breaks down one specific prediction into its components.
# Starting from the bias (baseline prediction), each feature adds or subtracts
# from the final value.

# %% tags=["hide-input"]
fig = ContributionsPlotter.plot_waterfall(contributions, timestep=48, top_n=10)
fig.update_layout(title="Prediction decomposition (timestep 48)", height=500)
fig.show()

# %% [markdown]
# ## Next steps
#
# - {doc}`hyperparameter_tuning_with_optuna` — use explainability insights
#   to guide which parameters to tune.
# - {doc}`custom_pipeline` — fine-tune feature engineering based on what
#   the contributions reveal.
