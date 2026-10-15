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
# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
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
# # Writing a Custom Forecaster
#
# OpenSTEF ships several forecasters (GBLinear, LightGBM, XGBoost, and more).
# Sometimes you need a model that is not in the list: a model from another
# library, a statistical method, or a simple baseline that your team trusts.
# You can add it by writing a subclass of
# [`Forecaster`](https://openstef.github.io/openstef/api/generated/openstef_models.models.forecasting.Forecaster.html).
# After that, your model works in every place where a built-in forecaster works.
#
# **What you'll learn:**
#
# - The contract of a `Forecaster`: what you must implement
# - How to write a small model, step by step: a *weekly profile* forecaster
# - How to put the model in a `ForecastingModel` and a workflow
# - How to compare your model with a built-in baseline
#
# ```{note}
# This tutorial is about the model itself (the `Forecaster` interface).
# To benchmark your own model in BEAM, see the
# [custom benchmark template](https://github.com/OpenSTEF/openstef/blob/main/examples/benchmarks/custom/custom_forecaster.py).
# To change the transforms around a model, see {doc}`custom_pipeline`.
# ```

# %% [markdown]
# ## The contract of a `Forecaster`
#
# A `Forecaster` is a configuration object ([Pydantic](https://docs.pydantic.dev/)) that also holds a trained model.
# It already has three fields:
#
# | Field | Meaning |
# |-------|---------|
# | `quantiles` | The quantiles that the model must predict, for example `[Q(0.1), Q(0.5), Q(0.9)]`. |
# | `horizons` | The lead times of the forecast, for example `PT36H`. |
# | `supports_batching` | Whether the model can handle more than one horizon in one call. |
#
# You must implement four members:
#
# | Member | What it does |
# |--------|--------------|
# | `hparams` | A property. Returns the hyperparameters of the model. |
# | `is_fitted` | A property. Returns `True` after `fit()` has run. |
# | `fit(data, data_val=None)` | Learns from a [`ForecastInputDataset`](https://openstef.github.io/openstef/api/generated/openstef_core.datasets.ForecastInputDataset.html). |
# | `predict(data)` | Returns a [`ForecastDataset`](https://openstef.github.io/openstef/api/generated/openstef_core.datasets.ForecastDataset.html) with one column for each quantile. |
#
# Two rules help you avoid problems:
#
# 1. Keep the learned state in *private attributes* (`PrivateAttr`). They are not part of the configuration.
# 2. Raise `NotFittedError` if `predict()` runs before `fit()`.

# %% [markdown]
# ## Load the dataset
#
# We use the same data as in {doc}`custom_pipeline`.

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
# ## The idea: a weekly profile
#
# Electricity load often repeats each week. Monday at 08:00 looks like the
# previous Mondays at 08:00. Our model uses this directly:
#
# - **`fit`**: for each time slot of the week (day, hour, minute), store the requested quantiles of the past values in that slot.
# - **`predict`**: for each future timestamp, look up its slot.
#
# The model uses only the target column. It needs no weather data and no lags.

# %% [markdown]
# ## Step 1: the hyperparameters
#
# Hyperparameters are a Pydantic class that extends
# [`HyperParams`](https://openstef.github.io/openstef/api/generated/openstef_core.mixins.predictor.HyperParams.html).
# Our model has one setting: the fewest observations that a time slot needs.
# A slot with fewer observations uses the overall quantile of the training data.

# %%
from pydantic import Field

from openstef_core.mixins.predictor import HyperParams


class WeeklyProfileHyperParams(HyperParams):
    """Settings of the weekly profile forecaster."""

    min_samples: int = Field(
        default=3,
        ge=1,
        description="Fewest observations that a time slot of the week needs. Other slots use the overall quantile.",
    )


# %% [markdown]
# ## Step 2: the forecaster
#
# The code below follows the contract from the table above.
#
# - `_profile` and `_fallback` are private attributes. They hold what `fit` learns.
# - `fit` reads the target with `data.target_series`.
# - `predict` calls `data.create_forecast_range(...)`. This gives the timestamps that must be forecast, starting at `forecast_start`.
# - The result has one column for each quantile. `Quantile.format()` gives the column name, for example `quantile_P10`.

# %%
from typing import override

import pandas as pd
from pydantic import PrivateAttr

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import InputValidationError, NotFittedError
from openstef_models.models.forecasting.forecaster import Forecaster


class WeeklyProfileForecaster(Forecaster):
    """Forecast each time slot of the week with the quantiles of the same slot in the past."""

    hyperparams: WeeklyProfileHyperParams = Field(default_factory=WeeklyProfileHyperParams)

    # Learned state: private, so it is not part of the configuration.
    _profile: pd.DataFrame | None = PrivateAttr(default=None)
    _fallback: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    @override
    def hparams(self) -> WeeklyProfileHyperParams:
        return self.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self._profile is not None

    @staticmethod
    def _slots(index: pd.DatetimeIndex) -> pd.MultiIndex:
        """Map each timestamp to its slot in the week."""
        return pd.MultiIndex.from_arrays([index.dayofweek, index.hour, index.minute], names=["day", "hour", "minute"])

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        target = data.target_series.dropna()
        if target.empty:
            raise InputValidationError("Training data must contain at least one non-NaN target value.")

        grouped = target.groupby(self._slots(pd.DatetimeIndex(target.index)))
        enough = grouped.count() >= self.hyperparams.min_samples
        names = {q: q.format() for q in self.quantiles}

        profile = pd.DataFrame({name: grouped.quantile(float(q)) for q, name in names.items()})
        self._profile = profile[enough]
        self._fallback = {name: float(target.quantile(float(q))) for q, name in names.items()}

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if self._profile is None:
            raise NotFittedError(self.__class__.__name__)

        index = data.create_forecast_range(horizon=self.max_horizon)
        values = self._profile.reindex(self._slots(index)).fillna(self._fallback)
        values.index = index

        return ForecastDataset(
            data=values,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
        )


# %% [markdown]
# ## Step 3: try the forecaster alone
#
# A `Forecaster` takes a `ForecastInputDataset`. We build one from the raw
# data. The model needs no features, so no preprocessing is required here.

# %%
quantiles = [Q(0.1), Q(0.5), Q(0.9)]
horizons = [LeadTime.from_string("PT36H")]

forecaster = WeeklyProfileForecaster(
    quantiles=quantiles,
    horizons=horizons,
    hyperparams=WeeklyProfileHyperParams(min_samples=3),
)
print(f"Fitted before fit(): {forecaster.is_fitted}")

train_input = ForecastInputDataset(
    data=train_dataset.data[["load"]],
    sample_interval=train_dataset.sample_interval,
    target_column="load",
)
forecaster.fit(train_input)
print(f"Fitted after fit():  {forecaster.is_fitted}")

# %% tags=["remove-cell"]
assert forecaster.is_fitted

# %% [markdown]
# ## Step 4: put it in a `ForecastingModel`
#
# [`ForecastingModel`](https://openstef.github.io/openstef/api/generated/openstef_models.models.ForecastingModel.html)
# wraps preprocessing, forecaster, and postprocessing.
# Our forecaster needs no features, so the preprocessing is empty.
# For the postprocessing we add a
# [`QuantileSorter`](https://openstef.github.io/openstef/api/generated/openstef_models.transforms.postprocessing.QuantileSorter.html),
# which makes sure that the quantiles never cross.
#
# The
# [`CustomForecastingWorkflow`](https://openstef.github.io/openstef/api/generated/openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow.html)
# adds the train and predict steps on top.
#
# ```{note}
# By default, `ForecastingModel` holds back part of the training data for
# validation and testing. The weekly profile needs no validation data, and it
# needs every week it can get. So we set both fractions to zero with a
# `DataSplitter`.
# ```

# %%
from openstef_core.mixins import TransformPipeline
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms.postprocessing import QuantileSorter
from openstef_models.utils.data_split import DataSplitter
from openstef_models.workflows import CustomForecastingWorkflow


def build_workflow(model_id: str, forecaster: Forecaster) -> CustomForecastingWorkflow:
    """Put a forecaster in a workflow without feature engineering."""
    model = ForecastingModel(
        preprocessing=TransformPipeline(transforms=[]),
        forecaster=forecaster,
        postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
        target_column="load",
        # Our models need no validation or test data, so they train on all rows.
        data_splitter=DataSplitter(val_fraction=0.0, test_fraction=0.0),
    )
    return CustomForecastingWorkflow(model_id=model_id, model=model, callbacks=[])


profile_workflow = build_workflow(
    "weekly_profile",
    WeeklyProfileForecaster(quantiles=quantiles, horizons=horizons),
)
profile_workflow.fit(train_dataset)
profile_forecast = profile_workflow.predict(predict_dataset, forecast_start=train_end)

print(f"Forecast rows: {len(profile_forecast.data)}")
print(f"Columns:       {list(profile_forecast.data.columns)}")

# %% tags=["remove-cell"]
assert len(profile_forecast.data) > 100, f"Expected >100 forecast rows, got {len(profile_forecast.data)}"

# %% [markdown]
# ## Step 5: compare with a built-in baseline
#
# A new model must beat a simple baseline. We use the built-in
# [`ConstantQuantileForecaster`](https://openstef.github.io/openstef/api/generated/openstef_models.models.forecasting.html).
# It predicts the same quantiles for every timestamp.
# Because both forecasters share the same interface, we change only one line.

# %%
from openstef_models.models.forecasting.constant_quantile_forecaster import (
    ConstantQuantileForecaster,
    ConstantQuantileForecasterHyperParams,
)

baseline_workflow = build_workflow(
    "constant_quantile",
    ConstantQuantileForecaster(
        quantiles=quantiles,
        horizons=horizons,
        hyperparams=ConstantQuantileForecasterHyperParams(),
    ),
)
baseline_workflow.fit(train_dataset)
baseline_forecast = baseline_workflow.predict(predict_dataset, forecast_start=train_end)

actual = predict_dataset.data["load"].loc[train_end:]


def mae(forecast: ForecastDataset) -> float:
    """Mean absolute error of the median forecast."""
    aligned = actual.reindex(forecast.median_series.index)
    return float((forecast.median_series - aligned).abs().mean())


print(f"MAE constant quantile: {mae(baseline_forecast):.3f}")
print(f"MAE weekly profile:    {mae(profile_forecast):.3f}")

# %% [markdown]
# ## Visualize the result

# %% tags=["hide-input"]
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=actual)
    .add_model(
        model_name="Constant quantile",
        forecast=baseline_forecast.median_series,
        quantiles=baseline_forecast.quantiles_data,
    )
    .add_model(
        model_name="Weekly profile",
        forecast=profile_forecast.median_series,
        quantiles=profile_forecast.quantiles_data,
    )
    .plot()
)
fig.update_layout(
    title="Custom forecaster — weekly profile vs constant baseline",
    yaxis_title="Load",
    xaxis_title="Time",
    height=450,
)
fig.show()

# %% [markdown]
# ## Checklist for your own forecaster
#
# - Extend `Forecaster`. Add a `HyperParams` class if the model has settings.
# - Implement `hparams`, `is_fitted`, `fit`, and `predict`.
# - Keep learned state in `PrivateAttr` fields.
# - Return one column for each quantile, named with `Quantile.format()`.
# - Raise `NotFittedError` when `predict()` runs before `fit()`.
# - If your model needs features, add transforms to the preprocessing pipeline, as shown in {doc}`custom_pipeline`.
#
# ## Next steps
#
# - {doc}`custom_pipeline` — add feature engineering around your forecaster.
# - {doc}`ensemble_forecasting` — combine your forecaster with other models.
# - {doc}`hyperparameter_tuning_with_optuna` — tune the hyperparameters of your model.
