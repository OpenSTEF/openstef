# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import LeadTime, Q
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
    XGBoostHyperParams,
)


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    """Create sample input dataset for forecaster training and prediction."""
    rng = np.random.default_rng(42)
    num_samples = 14
    start_date = datetime.fromisoformat("2025-01-01T00:00:00")

    feature_1 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_2 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_3 = rng.uniform(low=-1, high=1, size=num_samples)

    return ForecastInputDataset(
        data=pd.DataFrame(
            {
                "load": (feature_1 + feature_2 + feature_3) / 3,
                "feature1": feature_1,
                "feature2": feature_2,
                "feature3": feature_3,
            },
            index=pd.date_range(start=start_date, periods=num_samples, freq="1d"),
        ),
        sample_interval=timedelta(days=1),
        target_column="load",
        forecast_start=start_date + timedelta(days=num_samples // 2),
    )


def test_xgboost_forecaster__fit_predict(
    sample_forecast_input_dataset: ForecastInputDataset,
):
    # Arrange
    forecaster = XGBoostForecaster(
        config=XGBoostForecasterConfig(
            horizons=[LeadTime(timedelta(days=1))],
            quantiles=[Q(0.1), Q(0.5), Q(0.9)],
            verbosity=3,
        ),
    )

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)


def test_xgboost_forecaster__state_roundtrip(
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster state can be serialized and restored with preserved functionality."""
    # Arrange
    config = XGBoostForecasterConfig(
        horizons=[LeadTime(timedelta(days=1))],
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        hyperparams=XGBoostHyperParams(
            n_estimators=10,
        ),
    )

    original_forecaster = XGBoostForecaster(config=config)
    original_forecaster.fit(sample_forecast_input_dataset)

    # Act
    # Serialize state and create new forecaster from state
    state = original_forecaster.to_state()

    restored_forecaster = XGBoostForecaster(config=config)
    restored_forecaster = restored_forecaster.from_state(state)

    # Assert
    # Check that restored forecaster produces identical predictions
    assert restored_forecaster.is_fitted
    original_result = original_forecaster.predict(sample_forecast_input_dataset)
    restored_result = restored_forecaster.predict(sample_forecast_input_dataset)

    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval
