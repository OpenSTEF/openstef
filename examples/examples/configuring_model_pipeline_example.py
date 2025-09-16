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
    - ForecastingWorkflow: High-level orchestration

This example is useful for understanding how to integrate all OpenSTEF components
into a working forecasting system.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.joblib import LocalModelStorage
from openstef_models.models.forecasting.constant_median_forecaster import (
    ConstantMedianForecaster,
    ConstantMedianForecasterConfig,
)
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeaturePipeline, ForecastTransformPipeline
from openstef_models.transforms.general import ScalerTransform
from openstef_models.transforms.time_domain import HolidayFeaturesTransform
from openstef_models.transforms.time_domain.lag_transform import VersionedLagTransform
from openstef_models.workflows import ForecastingWorkflow

dataset = VersionedTimeSeriesDataset.from_dataframe(
    data=pd.DataFrame({
        "load": np.random.default_rng().standard_normal(size=24 * 14),
        "timestamp": pd.date_range("2025-01-01", periods=24 * 14, freq="h"),
        "available_at": pd.date_range("2025-01-01", periods=24 * 14, freq="h"),
    }),
    sample_interval=timedelta(hours=1),
)

model = ForecastingModel(
    preprocessing=FeaturePipeline(
        horizons=[LeadTime.from_string("PT36H")],
        horizon_transforms=[
            ScalerTransform(method="standard"),
            HolidayFeaturesTransform(country_code=CountryAlpha2("NL")),
        ],
        versioned_transforms=[VersionedLagTransform(column="load", lags=[timedelta(days=-7)])],
    ),
    forecaster=ConstantMedianForecaster(
        config=ConstantMedianForecasterConfig(
            horizons=[LeadTime.from_string("PT36H")],
            quantiles=[Q(0.5), Q(0.1), Q(0.9)],
        )
    ),
    postprocessing=ForecastTransformPipeline(transforms=[]),
    target_column="load",
)

storage = LocalModelStorage(storage_dir=Path("./model_storage"))

pipeline = ForecastingWorkflow.from_storage(
    model_id="constant_median_forecaster_v1",
    storage=storage,
    default_model_factory=lambda: model,
)

pipeline.fit(dataset)

forecast = pipeline.predict(dataset)

print(forecast.quantiles)
