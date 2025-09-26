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
from tabnanny import verbose

import numpy as np
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster, XGBoostForecasterConfig
import pandas as pd
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import (
    ForecastDataset,
    TimeSeriesDataset,
)
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster, GBLinearForecasterConfig
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeatureEngineeringPipeline
from openstef_models.transforms.general import DimensionalityReduction, ScalerTransform
from openstef_models.transforms.time_domain import HolidayFeaturesTransform
from openstef_models.workflows import ForecastingWorkflow

num_samples = 24 * 140


temperature = np.random.default_rng().standard_normal(size=num_samples) * 10 + 20
wind_speed = np.random.default_rng().standard_normal(size=num_samples) * 3 + 5
solar_radiation = np.random.default_rng().uniform(low=0, high=800, size=num_samples)

dataset = TimeSeriesDataset(
    data=pd.DataFrame({
        "load": temperature + wind_speed + np.random.default_rng().standard_normal(size=num_samples) * 0.5,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "solar_radiation": solar_radiation,
    }, index=pd.date_range("2025-01-01", periods=num_samples, freq="h")),
    sample_interval=timedelta(hours=1),
)

with_dim_reduction = True
model_name = "gblinear"
prefix = ("with_dim_red" if with_dim_reduction else "no_dim_red") + " " + model_name

model = ForecastingModel(
    preprocessing=FeatureEngineeringPipeline.create(
        horizons=[LeadTime.from_string("PT36H")],
        horizon_transforms=[
            # ScalerTransform(method="standard"),
            # HolidayFeaturesTransform(country_code=CountryAlpha2("NL")),
            DimensionalityReduction(columns=["temperature", "wind_speed", "solar_radiation"], method="pca"),
        ] if with_dim_reduction else []
    ),
    forecaster=GBLinearForecaster(
        config=GBLinearForecasterConfig(
            horizons=[LeadTime.from_string("PT36H")],
            quantiles=[Q(0.5), Q(0.1), Q(0.9)],
            verbosity=1
        )
    ),
    # forecaster=XGBoostForecaster(
    #     config=XGBoostForecasterConfig(
    #         horizons=[LeadTime.from_string("PT36H")],
    #         quantiles=[Q(0.5), Q(0.1), Q(0.9)],
    #         verbosity=1
    #     )
    # ),
    postprocessing=TransformPipeline[ForecastDataset](transforms=[]),
    target_column="load",
)

pipeline = ForecastingWorkflow(
    model_id="constant_median_forecaster_v1",
    model=model,
)

pipeline.fit(dataset)

forecast = pipeline.predict(dataset)

print(forecast.data)


preview = forecast.data.copy()
preview["load"] = dataset.data["load"]

pd.options.plotting.backend = "plotly"

fig = preview.plot(title="Forecast vs Actual Load")

output_dir = Path(__file__).resolve().parent


fig.write_html(output_dir / f"forecast_vs_actual_{prefix}.html")


list(model.preprocessing.transform(dataset).values())[0].to_parquet(output_dir / f"preprocessed_data_{prefix}.parquet")