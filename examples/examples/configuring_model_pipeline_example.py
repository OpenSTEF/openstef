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
from openstef_models.transforms import FeaturePipeline, PostprocessingPipeline
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
    postprocessing=PostprocessingPipeline(transforms=[]),
    target_column="load",
)

storage = LocalModelStorage(storage_dir=Path("./model_storage"))

pipeline = ForecastingWorkflow.from_storage(
    model_id="constant_median_forecaster_v1",
    storage=storage,
    default_model=model,
)

pipeline.fit(dataset)

forecast = pipeline.predict(dataset)

print(forecast.quantiles)
