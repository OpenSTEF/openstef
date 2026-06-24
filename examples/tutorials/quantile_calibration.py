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
# # Quantile Calibration
#
# Improve the reliability of probabilistic forecasts using isotonic quantile
# calibration.  A well-calibrated P10 quantile should exceed actual values
# roughly 10 % of the time — this tutorial shows how to measure and correct
# deviations.
#
# **What you'll learn:**
#
# - Measure quantile calibration with observed coverage
# - Add isotonic calibration as a postprocessing step
# - Compare before/after calibration on real data
#
# ```{note}
# This tutorial uses a small data slice for fast execution.
# See `examples/benchmarks/` for production-scale runs.
# ```
#
# **Key API references:**
# [`IsotonicQuantileCalibrator`](https://openstef.github.io/openstef/api/generated/openstef_models.transforms.postprocessing.IsotonicQuantileCalibrator.html)
# · [`ForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.ForecastingWorkflowConfig.html)

# %% [markdown]
# ## Load data and train an uncalibrated model
#
# We start with the same GBLinear setup as the {doc}`forecasting_quickstart` and
# measure how well its predicted quantiles match observed coverage.
# The [`ForecastingWorkflowConfig`](https://openstef.github.io/openstef/api/generated/openstef_models.presets.ForecastingWorkflowConfig.html)
# defines the model architecture and quantile levels.

# %%
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

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

quantiles = [Q(0.1), Q(0.5), Q(0.9)]

config = ForecastingWorkflowConfig(
    model_id="uncalibrated_gblinear",
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

workflow_uncal = create_forecasting_workflow(config=config)
workflow_uncal.fit(train_dataset)
forecast_uncal = workflow_uncal.predict(predict_dataset, forecast_start=train_end)

print(f"Forecast rows: {len(forecast_uncal.data)}")

# %% tags=["remove-cell"]
assert len(forecast_uncal.data) > 100, f"Expected >100 forecast rows, got {len(forecast_uncal.data)}"

# %% [markdown]
# ## Measure calibration quality
#
# For a perfectly calibrated forecast at quantile $p$, the fraction of
# observations falling below the predicted value should equal $p$.  We compute
# the **observed coverage** for each quantile and compare it to the expected
# level.

# %%
actuals = predict_dataset.data["load"].loc[train_end:].reindex(forecast_uncal.data.index).dropna()
forecast_aligned = forecast_uncal.data.loc[actuals.index]

expected = [float(q) for q in quantiles]
observed_uncal = [float((actuals <= forecast_aligned[f"quantile_P{int(float(q) * 100)}"]).mean()) for q in quantiles]

calibration_df = pd.DataFrame(
    {
        "quantile": [f"P{int(float(q) * 100)}" for q in quantiles],
        "expected": expected,
        "observed": observed_uncal,
        "error": [o - e for o, e in zip(observed_uncal, expected, strict=True)],
    }
)
print("Calibration before isotonic correction:")
print(calibration_df.to_string(index=False))

# %% [markdown]
# ## Add isotonic calibration
#
# [`IsotonicQuantileCalibrator`](https://openstef.github.io/openstef/api/generated/openstef_models.transforms.postprocessing.IsotonicQuantileCalibrator.html) is a postprocessing transform that learns a
# monotonic mapping from predicted quantiles to observed quantile levels.
# During training it fits on the validation split; during prediction it
# corrects each quantile value.
#
# We create a second workflow identical to the first, but with the calibrator
# appended to its postprocessing pipeline.

# %%
from openstef_models.transforms.postprocessing import IsotonicQuantileCalibrator

config_cal = config.model_copy(update={"model_id": "calibrated_gblinear"})
workflow_cal = create_forecasting_workflow(config=config_cal)

# Append isotonic calibration to the existing postprocessing pipeline
workflow_cal.model.postprocessing.transforms.append(
    IsotonicQuantileCalibrator(
        quantiles=quantiles,
        use_local_quantile_estimation=True,
    )
)

workflow_cal.fit(train_dataset)
forecast_cal = workflow_cal.predict(predict_dataset, forecast_start=train_end)

# %% tags=["remove-cell"]
assert len(forecast_cal.data) > 100, f"Expected >100 calibrated forecast rows, got {len(forecast_cal.data)}"

# %% [markdown]
# ## Compare calibration before and after

# %%
forecast_cal_aligned = forecast_cal.data.loc[actuals.index]

observed_cal = [float((actuals <= forecast_cal_aligned[f"quantile_P{int(float(q) * 100)}"]).mean()) for q in quantiles]

comparison_df = pd.DataFrame(
    {
        "quantile": [f"P{int(float(q) * 100)}" for q in quantiles],
        "expected": expected,
        "observed (before)": observed_uncal,
        "observed (after)": observed_cal,
        "error (before)": [o - e for o, e in zip(observed_uncal, expected, strict=True)],
        "error (after)": [o - e for o, e in zip(observed_cal, expected, strict=True)],
    }
)
print(comparison_df.to_string(index=False))

# %% tags=["hide-input"]
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line={"color": "gray", "dash": "dash", "width": 2},
    )
)

fig.add_trace(
    go.Scatter(
        x=expected,
        y=observed_uncal,
        mode="markers+lines",
        name="Before calibration",
        marker={"size": 12, "color": "red", "symbol": "x"},
        line={"color": "red", "width": 2, "dash": "dot"},
    )
)

fig.add_trace(
    go.Scatter(
        x=expected,
        y=observed_cal,
        mode="markers+lines",
        name="After calibration",
        marker={"size": 12, "color": "blue"},
        line={"color": "blue", "width": 2},
    )
)

fig.update_layout(
    title="Quantile calibration: expected vs observed coverage",
    xaxis_title="Expected quantile level",
    yaxis_title="Observed coverage",
    xaxis={"range": [0, 1], "tickvals": [0, 0.1, 0.5, 0.9, 1]},
    yaxis={"range": [0, 1], "tickvals": [0, 0.1, 0.5, 0.9, 1]},
    height=500,
    width=600,
)
fig.show()

# %% [markdown]
# Points closer to the diagonal indicate better calibration.  The isotonic
# correction pulls the observed coverage towards the expected level, improving
# the reliability of uncertainty estimates.  To measure calibration stability
# over longer time horizons, combine this with a {doc}`backtesting_quickstart`.

# %% [markdown]
# ## Next steps
#
# - {doc}`backtesting_quickstart` — measure calibration consistency over
#   realistic operational periods.
# - {doc}`ensemble_forecasting` — apply calibration to ensemble models
#   for combined accuracy and reliable uncertainty.
