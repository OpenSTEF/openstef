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

# pyright: basic

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
# # Feature Engineering
#
# OpenSTEF provides a library of **transforms** that add derived features to your
# time series data before model training. This tutorial shows how to explore,
# apply, and visualize individual transforms, then compose them into a
# preprocessing pipeline.
#
# **What you'll learn:**
#
# - The four transform domains (time, weather, energy, general)
# - How to apply individual transforms and inspect their output
# - How to compose transforms into a `TransformPipeline`
# - How transforms interact with horizons and lag features
#
# ```{seealso}
# - {doc}`custom_pipeline` — full end-to-end custom model assembly
# - {doc}`forecasting_quickstart` — standard approach using presets
# ```

# %% [markdown]
# ## Load sample data

# %%
from datetime import datetime, timedelta

from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q

dataset = load_liander_dataset()

# Use 30 days for demonstration
start = datetime.fromisoformat("2024-03-01T00:00:00Z")
end = start + timedelta(days=30)
sample = dataset.filter_by_range(start=start, end=end)

print(f"Dataset: {sample.data.shape[0]:,} rows × {sample.data.shape[1]} columns")
print(f"Columns: {list(sample.data.columns[:10])}...")
sample.data.head()

# %% [markdown]
# ## Transform domains
#
# Transforms are organized into four domains:
#
# | Domain | Purpose | Examples |
# |--------|---------|----------|
# | **Time** | Temporal patterns | Lags, cyclic features, holidays, datetime |
# | **Weather** | Meteorological derived features | Atmosphere, radiation, daylight |
# | **Energy** | Domain-specific physics | Wind power curve |
# | **General** | Data cleaning & preparation | Scaling, imputation, outlier handling |
#
# Each transform implements the `TimeSeriesTransform` protocol: it takes a
# `TimeSeriesDataset` and returns a `TimeSeriesDataset` with additional or
# modified columns.

# %% [markdown]
# ## Time domain transforms
#
# ### Cyclic features
#
# Encodes hour-of-day and day-of-week as sine/cosine pairs, capturing
# periodicity without the discontinuity of raw integer encodings.

# %%
from openstef_models.transforms.time_domain import CyclicFeaturesAdder

cyclic = CyclicFeaturesAdder()
result = cyclic.transform(sample)

# Show the new cyclic columns
new_cols = [c for c in result.data.columns if c not in sample.data.columns]
print(f"Added {len(new_cols)} columns: {new_cols}")
result.data[new_cols].head(8)

# %% [markdown]
# ### Holiday features
#
# Adds binary indicators for national holidays, school holidays, and
# bridge days for a given country.

# %%
from openstef_models.transforms.time_domain import HolidayFeatureAdder

holidays = HolidayFeatureAdder(country_code="NL")
result = holidays.transform(sample)

holiday_cols = [c for c in result.data.columns if c not in sample.data.columns]
print(f"Added {len(holiday_cols)} columns: {holiday_cols}")
# Show rows where a holiday is detected
result.data[result.data[holiday_cols].any(axis=1)][holiday_cols].head()

# %% [markdown]
# ### Lag features
#
# Creates lagged copies of the target column. The lag offsets depend on the
# forecast horizon — features must use only data available at prediction time.

# %%
from openstef_models.transforms.time_domain.lags_adder import LagsAdder

horizons = [LeadTime.from_string("PT36H")]

lags = LagsAdder(
    history_available=timedelta(days=14),
    horizons=horizons,
    add_trivial_lags=False,
    target_column="load",
    custom_lags=[timedelta(days=7)],
    lag_fallback_offset=timedelta(days=7),
)
result = lags.transform(sample)

lag_cols = [c for c in result.data.columns if "lag" in c.lower() or c not in sample.data.columns]
print(f"Added lag columns: {lag_cols[:5]}...")
result.data[lag_cols[:3]].describe()

# %% [markdown]
# ### Datetime features
#
# Extracts numeric components (hour, day of week, month, etc.) from the
# timestamp index.

# %%
from openstef_models.transforms.time_domain import DatetimeFeaturesAdder

dt_features = DatetimeFeaturesAdder()
result = dt_features.transform(sample)

dt_cols = [c for c in result.data.columns if c not in sample.data.columns]
print(f"Added {len(dt_cols)} columns: {dt_cols}")
result.data[dt_cols].head()

# %% [markdown]
# ## Weather domain transforms
#
# ### Daylight features
#
# Computes sunrise/sunset times and daylight duration based on geographic
# coordinates (derived from the dataset's location metadata).

# %%
from openstef_models.transforms.weather_domain import DaylightFeatureAdder

daylight = DaylightFeatureAdder()
result = daylight.transform(sample)

daylight_cols = [c for c in result.data.columns if c not in sample.data.columns]
print(f"Added: {daylight_cols}")
result.data[daylight_cols].describe()

# %% [markdown]
# ## General transforms (data cleaning)
#
# These transforms don't add features — they clean and prepare data.
#
# ### Imputer
#
# Fills missing values using a configurable strategy (mean, median, zero, etc.).

# %%
from openstef_models.transforms.general import Imputer
from openstef_models.utils.feature_selection import Exclude

imputer = Imputer(selection=Exclude("load"), imputation_strategy="mean")

# Introduce some NaNs to demonstrate
import numpy as np

noisy = sample.data.copy()
rng = np.random.default_rng(42)
mask = rng.random(noisy.shape) < 0.02  # 2% missing
noisy[mask] = np.nan

from openstef_core.datasets import TimeSeriesDataset

noisy_ds = TimeSeriesDataset(data=noisy, target_column="load")
print(f"Before: {noisy_ds.data.isna().sum().sum()} NaN values")

result = imputer.transform(noisy_ds)
print(f"After:  {result.data.isna().sum().sum()} NaN values")

# %% [markdown]
# ### Scaler
#
# Standardizes or normalizes feature columns (excluding the target).

# %%
from openstef_models.transforms.general import Scaler

scaler = Scaler(selection=Exclude("load"), method="standard")
result = scaler.fit_transform(sample)

# Verify zero mean, unit variance on non-target columns
non_target = [c for c in result.data.columns if c != "load"]
stats = result.data[non_target].describe().loc[["mean", "std"]]
print("After scaling (non-target columns):")
stats.iloc[:, :5]

# %% [markdown]
# ## Composing a pipeline
#
# Individual transforms are composed into a `TransformPipeline` that applies
# them sequentially. This is what the forecasting model uses internally.

# %%
from openstef_core.mixins import TransformPipeline

pipeline = TransformPipeline(
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
        DatetimeFeaturesAdder(),
        # Cleaning
        Scaler(selection=Exclude("load"), method="standard"),
        Imputer(selection=Exclude("load"), imputation_strategy="mean"),
    ]
)

result = pipeline.fit_transform(sample)
print(f"Input:  {sample.data.shape[1]} columns")
print(f"Output: {result.data.shape[1]} columns")
print(f"\nNew features added: {result.data.shape[1] - sample.data.shape[1]}")

# %% [markdown]
# ## Summary
#
# OpenSTEF's transform system provides:
#
# - **Modular** — mix and match transforms from any domain
# - **Horizon-aware** — lag features respect prediction-time data availability
# - **Composable** — chain transforms in a `TransformPipeline`
# - **Configurable** — each transform is a Pydantic model with typed parameters
#
# ```{seealso}
# - {doc}`custom_pipeline` — use these transforms in a full forecasting model
# - [`TransformPipeline` API](../api/generated/openstef_core.mixins.TransformPipeline.rst)
# - [`TimeSeriesTransform` API](../api/generated/openstef_core.transforms.TimeSeriesTransform.rst)
# ```
