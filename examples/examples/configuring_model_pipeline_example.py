"""Configuring Model Pipeline Example.

====================================

This example demonstrates how to configure and use a complete forecasting pipeline
in OpenSTEF. It shows how to:

1. Create synthetic time series data for demonstration
2. Configure a full forecasting model with preprocessing and postprocessing
3. Set up model storage for persistence
4. Use the workflow pattern for training and prediction

The example uses a ConstantMedianForecaster with feature engineering including
holiday features, lag transforms, and data scaling. This represents a typical
OpenSTEF forecasting setup that can be adapted for real-world use cases.

Key Components:
    - VersionedTimeSeriesDataset: Time series data structure
    - ForecastingModel: Complete forecasting pipeline
    - FeaturePipeline: Preprocessing with holidays and lags
    - LocalModelStorage: File-based model persistence
    - CustomForecastingWorkflow: High-level orchestration

This example is useful for understanding how to integrate all OpenSTEF components
into a working forecasting system.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic_extra_types.country import CountryAlpha2

from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow import MLFlowStorageCallback
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearForecasterConfig,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeatureEngineeringPipeline, PostprocessingPipeline
from openstef_models.transforms.general import ScalerTransform
from openstef_models.transforms.time_domain import HolidayFeaturesTransform
from openstef_models.transforms.time_domain.lag_transform import VersionedLagTransform
from openstef_models.workflows import CustomForecastingWorkflow

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

workspace_dir = Path(__file__).parent.resolve()

# Create synthetic time series data
n_samples = 24 * 31 * 3  # 3 months of hourly data
rng = np.random.default_rng(42)
temp = rng.standard_normal(size=n_samples)
wind = rng.standard_normal(size=n_samples)
radiation = rng.standard_normal(size=n_samples)
timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="h")

load_dataset = VersionedTimeSeriesDataset.from_dataframe(
    data=pd.DataFrame({
        "load": wind * -10 + temp * -3 + radiation * -5 + rng.standard_normal(size=n_samples) * 2,
        "timestamp": timestamps,
        "available_at": timestamps,
    }),
    sample_interval=timedelta(hours=1),
)
predictor_dataset = VersionedTimeSeriesDataset.from_dataframe(
    data=pd.DataFrame({
        "temp": temp,
        "wind": wind,
        "radiation": radiation,
        "timestamp": timestamps,
        "available_at": timestamps - timedelta(days=7),
    }),
    sample_interval=timedelta(hours=1),
)

dataset = VersionedTimeSeriesDataset.concat([load_dataset, predictor_dataset], mode="inner")

model = ForecastingModel(
    preprocessing=FeatureEngineeringPipeline.create(
        horizons=[LeadTime.from_string("PT36H")],
        horizon_transforms=[
            ScalerTransform(method="standard", columns=["temp", "wind", "radiation"]),
            HolidayFeaturesTransform(country_code=CountryAlpha2("NL")),
        ],
        versioned_transforms=[VersionedLagTransform(column="load", lags=[timedelta(days=-7)])],
    ),
    forecaster=GBLinearForecaster(
        config=GBLinearForecasterConfig(
            horizons=[LeadTime.from_string("PT36H")],
            quantiles=[Q(0.5), Q(0.1), Q(0.9)],
            hyperparams=GBLinearHyperParams(
                n_estimators=1000,
                learning_rate=0.3,
            ),
            verbosity=True,
        )
    ),
    postprocessing=PostprocessingPipeline(transforms=[]),
    target_column="load",
    tags={
        "model": "gblinear",
        "version": "1.0.0",
    },
)

pipeline = CustomForecastingWorkflow(
    model_id="gblinear_forecaster_v1",
    model=model,
    callbacks=[
        MLFlowStorageCallback(
            storage=MLFlowStorage(
                tracking_uri=str(workspace_dir / "mlflow_tracking"),
                local_artifacts_path=workspace_dir / "mlflow_tracking_artifacts",
            ),
            model_reuse_enable=False,
        )
    ],
)

logger.info("Starting model training")
result = pipeline.fit(dataset)
if result is not None:
    logger.info("Full eval result:\n%s", result.metrics_full.to_dataframe())

    if result.metrics_test is not None:
        logger.info("Test result:\n%s", result.metrics_test.to_dataframe())

logger.info("Starting forecasting")
forecast: ForecastDataset = pipeline.predict(dataset)

print(forecast.data.tail())


logger.info("Storing forecast plot to forecast_plot.html")
fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(
        measurements=TimeSeriesDataset(
            data=dataset.select_version().data[["load"]],
            sample_interval=dataset.sample_interval,
        )
    )
    .add_model(
        model_name="gblinear",
        forecast=TimeSeriesDataset(
            data=forecast.median_series().to_frame(name="load"),
            sample_interval=dataset.sample_interval,
        ),
    )
    .plot()
)

fig.write_html("forecast_plot.html")  # pyright: ignore[reportUnknownMemberType]
