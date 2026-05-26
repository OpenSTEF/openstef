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
    ),
)

# %% [markdown]
# # Building a Custom Pipeline
#
# The [`create_forecasting_workflow`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.create_forecasting_workflow.html) preset handles pipeline assembly
# automatically.  When you need full control — custom transforms, different
# feature engineering, or non-standard postprocessing — you can build a
# [`ForecastingModel`](https://openstef.github.io/openstef/api/generated/openstef_models.models.ForecastingModel.html) from individual components.
#
# **What you'll learn:**
#
# - Assemble preprocessing, forecaster, and postprocessing into a pipeline
# - Select and configure individual transforms
# - Train and predict with a hand-built pipeline
# - Compare the custom pipeline against a preset
#
# ```{note}
# This tutorial is for advanced users who need to go beyond presets.
# Start with {doc}`forecasting_quickstart` for the standard approach.
# ```
#
# **Key API references:**
# [`ForecastingModel`](https://openstef.github.io/openstef/api/generated/openstef_models.models.ForecastingModel.html)
# · [`TransformPipeline`](https://openstef.github.io/openstef/api/generated/openstef_core.mixins.TransformPipeline.html)
# · [`GBLinearForecaster`](https://openstef.github.io/openstef/api/generated/openstef_models.models.forecasting.html)

# %% [markdown]
# ## Load the dataset

# %%
from datetime import timedelta

from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q

dataset = load_liander_dataset()

from datetime import datetime

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
# ## Define pipeline components
#
# A [`ForecastingModel`](https://openstef.github.io/openstef/api/generated/openstef_models.models.ForecastingModel.html) has three stages:
#
# 1. **Preprocessing** — feature engineering and data cleaning transforms
# 2. **Forecaster** — the model that produces predictions
# 3. **Postprocessing** — transforms applied to the forecast output
#
# Below we build each stage explicitly.

# %% [markdown]
# ### Preprocessing
#
# We select transforms from the available modules:
#
# | Module | Transforms |
# |--------|-----------|
# | `transforms.general` | Scaler, Imputer, NaNDropper, OutlierHandler, EmptyFeatureRemover |
# | `transforms.time_domain` | HolidayFeatureAdder, DatetimeFeaturesAdder, CyclicFeaturesAdder, LagsAdder |
# | `transforms.weather_domain` | AtmosphereDerivedFeaturesAdder, DaylightFeatureAdder, RadiationDerivedFeaturesAdder |
# | `transforms.energy_domain` | WindPowerFeatureAdder |
# | `transforms.validation` | CompletenessChecker, FlatlineChecker |

# %%
from openstef_core.mixins import TransformPipeline
from openstef_models.transforms.general import EmptyFeatureRemover, Imputer, NaNDropper, Scaler
from openstef_models.transforms.time_domain import CyclicFeaturesAdder, HolidayFeatureAdder
from openstef_models.transforms.time_domain.lags_adder import LagsAdder
from openstef_models.utils.feature_selection import Exclude

quantiles = [Q(0.1), Q(0.5), Q(0.9)]
horizons = [LeadTime.from_string("PT36H")]

preprocessing = TransformPipeline(
    transforms=[
        # Feature engineering
        LagsAdder(
            history_available=timedelta(days=14),
            horizons=horizons,
            add_trivial_lags=False,
            target_column="load",
            custom_lags=[timedelta(days=7)],
            lag_fallback_offset=timedelta(days=7),
        ),
        CyclicFeaturesAdder(),
        HolidayFeatureAdder(country_code="NL"),
        # Standardization
        Scaler(selection=Exclude("load"), method="standard"),
        EmptyFeatureRemover(),
        # Missing value handling
        Imputer(selection=Exclude("load"), imputation_strategy="mean"),
        NaNDropper(selection=Exclude("load")),
    ]
)

print(f"Preprocessing steps: {len(preprocessing.transforms)}")
for t in preprocessing.transforms:
    print(f"  - {type(t).__name__}")

# %% [markdown]
# ### Forecaster
#
# We use `GBLinearForecaster` — a gradient-boosted linear model that works well
# with the Imputer + NaNDropper preprocessing pattern above.

# %%
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearHyperParams,
)

forecaster = GBLinearForecaster(
    quantiles=quantiles,
    horizons=horizons,
    hyperparams=GBLinearHyperParams(
        n_steps=100,
        learning_rate=0.3,
    ),
    verbosity=0,
)

# %% [markdown]
# ### Postprocessing
#
# We add a [`QuantileSorter`](https://openstef.github.io/openstef/api/generated/openstef_models.transforms.postprocessing.QuantileSorter.html) (ensures quantile ordering) and a
# [`ConfidenceIntervalApplicator`](https://openstef.github.io/openstef/api/generated/openstef_models.transforms.postprocessing.ConfidenceIntervalApplicator.html) (adds confidence interval columns).

# %%
from openstef_models.transforms.postprocessing import (
    ConfidenceIntervalApplicator,
    QuantileSorter,
)

postprocessing = TransformPipeline(
    transforms=[
        QuantileSorter(),
        ConfidenceIntervalApplicator(
            quantiles=quantiles,
            add_quantiles_from_std=False,
        ),
    ]
)

# %% [markdown]
# ## Assemble the model
#
# [`ForecastingModel`](https://openstef.github.io/openstef/api/generated/openstef_models.models.ForecastingModel.html) combines all three stages.  We wrap it in a
# [`CustomForecastingWorkflow`](https://openstef.github.io/openstef/api/generated/openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow.html) which adds train/predict orchestration.

# %%
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.workflows import CustomForecastingWorkflow

model = ForecastingModel(
    preprocessing=preprocessing,
    forecaster=forecaster,
    postprocessing=postprocessing,
    target_column="load",
)

workflow = CustomForecastingWorkflow(
    model_id="custom_pipeline_demo",
    model=model,
    callbacks=[],
)

# %% [markdown]
# ## Train and predict

# %%
result = workflow.fit(train_dataset)
forecast = workflow.predict(predict_dataset, forecast_start=train_end)

print(f"Forecast rows:  {len(forecast.data)}")
print(f"Columns:        {list(forecast.data.columns)}")

# %% tags=["remove-cell"]
assert len(forecast.data) > 100, f"Expected >100 forecast rows, got {len(forecast.data)}"

# %% [markdown]
# ## Visualize the result

# %% tags=["hide-input"]
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=predict_dataset.data["load"].loc[train_end:])
    .add_model(
        model_name="Custom GBLinear",
        forecast=forecast.median_series,
        quantiles=forecast.quantiles_data,
    )
    .plot()
)
fig.update_layout(
    title="Custom pipeline — forecast vs actuals",
    yaxis_title="Load (MW)",
    xaxis_title="Time",
    height=450,
)
fig.show()

# %% [markdown]
# ## Using components individually
#
# `ForecastingModel` is convenient, but every component also works on its
# own.  You can run the preprocessing pipeline, inspect intermediate data,
# and call the forecaster directly.

# %% [markdown]
# ### Run preprocessing on raw data

# %%
preprocessed = model.preprocessing.transform(train_dataset)

print(f"Before preprocessing: {train_dataset.data.shape[1]} columns")
print(f"After preprocessing:  {preprocessed.data.shape[1]} columns")
print(f"\nAdded features: {sorted(set(preprocessed.data.columns) - set(train_dataset.data.columns))[:8]}...")

# %% [markdown]
# ### Run a single transform

# %%
single_transform = CyclicFeaturesAdder()
single_transform.fit(train_dataset)
result_single = single_transform.transform(train_dataset)

print(
    f"CyclicFeaturesAdder added {len(single_transform.features_added())} columns: {single_transform.features_added()}"
)

# %% [markdown]
# ### Call the forecaster directly
#
# After preprocessing, you can pass the data to a [`ForecastInputDataset`](https://openstef.github.io/openstef/api/generated/openstef_core.datasets.ForecastInputDataset.html)
# and call the forecaster directly.
# This is useful for debugging or integrating into custom workflows.

# %%
from openstef_core.datasets import ForecastInputDataset

# Preprocess the prediction data
preprocessed_predict = model.preprocessing.transform(predict_dataset)

# Convert to ForecastInputDataset (what the forecaster expects)
forecast_input = ForecastInputDataset(
    data=preprocessed_predict.data,
    sample_interval=preprocessed_predict.sample_interval,
    target_column="load",
    forecast_start=train_end,
)

# Call the forecaster directly
raw_forecast = model.forecaster.predict(forecast_input)
print(f"Raw forecast shape: {raw_forecast.data.shape}")
print(f"Raw forecast columns: {list(raw_forecast.data.columns)}")

# %% [markdown]
# ## Next steps
#
# - {doc}`ensemble_forecasting` — combine your custom pipeline with other
#   models into an ensemble for better accuracy.
# - {doc}`quantile_calibration` — append isotonic calibration to your
#   postprocessing for more reliable confidence intervals.
