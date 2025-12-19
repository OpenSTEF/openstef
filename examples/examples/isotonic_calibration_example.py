"""Isotonic Quantile Calibration Example.

=======================================

This example demonstrates how to use isotonic quantile calibration to improve
the reliability of probabilistic forecasts. It shows:

1. Training a forecasting model with isotonic calibration as postprocessing
2. Visualizing calibration quality (expected vs observed coverage)

Isotonic calibration ensures that predicted quantiles match observed quantile
levels, improving the reliability of uncertainty estimates.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearForecasterConfig,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms.postprocessing import IsotonicQuantileCalibrator
from openstef_models.workflows import CustomForecastingWorkflow

# Step 1: Create synthetic time series data
n_samples = 24 * 31 * 3  # 3 months of hourly data
rng = np.random.default_rng(42)
timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="h")

dataset = TimeSeriesDataset(
    data=pd.DataFrame(
        {
            "load": rng.standard_normal(size=n_samples) * 10 + 50,
            "feature": rng.standard_normal(size=n_samples),
        },
        index=timestamps,
    ),
    sample_interval=timedelta(hours=1),
)

# Step 2: Configure model without calibration (for comparison)
model_uncalibrated = ForecastingModel(
    forecaster=GBLinearForecaster(
        config=GBLinearForecasterConfig(
            horizons=[LeadTime.from_string("PT1H")],
            quantiles=[Q(0.1), Q(0.5), Q(0.9)],
            hyperparams=GBLinearHyperParams(n_steps=100),
            verbosity=0,
        )
    ),
    target_column="load",
)

pipeline_uncalibrated = CustomForecastingWorkflow(model_id="uncalibrated_forecaster", model=model_uncalibrated)
pipeline_uncalibrated.fit(dataset)
forecast_uncalibrated = pipeline_uncalibrated.predict(dataset)

# Step 3: Configure model with windowed isotonic quantile calibration
model_calibrated = ForecastingModel(
    forecaster=GBLinearForecaster(
        config=GBLinearForecasterConfig(
            horizons=[LeadTime.from_string("PT1H")],
            quantiles=[Q(0.1), Q(0.5), Q(0.9)],
            hyperparams=GBLinearHyperParams(n_steps=100),
            verbosity=0,
        )
    ),
    postprocessing=TransformPipeline(
        transforms=[
            IsotonicQuantileCalibrator(
                quantiles=[Q(0.1), Q(0.5), Q(0.9)],
                use_local_quantile_estimation=True,  # Enable windowed approach
                window_size=100,  # Window size for local estimation
            ),
        ],
    ),
    target_column="load",
)

# Step 4: Train and predict with calibration
pipeline_calibrated = CustomForecastingWorkflow(model_id="calibrated_forecaster", model=model_calibrated)
pipeline_calibrated.fit(dataset)
forecast_calibrated = pipeline_calibrated.predict(dataset)


# Step 5: Visualize calibration quality (before and after)
def plot_calibration_comparison(
    forecast_before: ForecastDataset, forecast_after: ForecastDataset, actuals: pd.Series
) -> go.Figure:
    """Plot expected vs observed quantile coverage before and after calibration.

    Returns:
        A Plotly figure showing the calibration comparison.
    """

    def calculate_coverage(forecast: ForecastDataset) -> list[float]:
        common_index = forecast.data.index.intersection(actuals.index)
        forecast_aligned = forecast.data.loc[common_index]
        actuals_aligned = actuals.loc[common_index]
        return [
            (actuals_aligned <= forecast_aligned["quantile_P10"]).mean(),
            (actuals_aligned <= forecast_aligned["quantile_P50"]).mean(),
            (actuals_aligned <= forecast_aligned["quantile_P90"]).mean(),
        ]

    expected = [0.1, 0.5, 0.9]
    observed_before = calculate_coverage(forecast_before)
    observed_after = calculate_coverage(forecast_after)

    fig = go.Figure()

    # expected == observed line
    fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="expected equals observed",
            line={"color": "gray", "dash": "dash", "width": 2},
        )
    )

    # Before calibration
    fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
        go.Scatter(
            x=expected,
            y=observed_before,
            mode="markers+lines",
            name="before isotonic calibration",
            marker={"size": 12, "color": "red", "symbol": "x"},
            line={"color": "red", "width": 2, "dash": "dot"},
        )
    )

    # After calibration
    fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
        go.Scatter(
            x=expected,
            y=observed_after,
            mode="markers+lines",
            name="after isotonic calibration",
            marker={"size": 12, "color": "blue"},
            line={"color": "blue", "width": 2},
        )
    )

    fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
        title="Isontonic quantile calibration",
        xaxis_title="expected quantile",
        yaxis_title="observed quantile",
        xaxis={"range": [0, 1], "tickvals": [0, 0.1, 0.5, 0.9, 1]},
        yaxis={"range": [0, 1], "tickvals": [0, 0.1, 0.5, 0.9, 1]},
        width=600,
        height=600,
    )
    return fig


calibration_fig = plot_calibration_comparison(
    forecast_uncalibrated, forecast_calibrated, dataset.select_version().data["load"]
)
calibration_fig.write_html("calibration_plot.html")  # pyright: ignore[reportUnknownMemberType]
print("Calibration plot saved to calibration_plot.html")
