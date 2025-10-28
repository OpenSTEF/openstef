# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.constant_median_forecaster import (
    ConstantMedianForecaster,
    ConstantMedianForecasterConfig,
    ConstantMedianForecasterHyperParams,
)


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    """Create sample input dataset for forecaster training and prediction.

    Returns:
        ForecastInputDataset with load values [90, 100, 110, 120, 130] spanning 5 hours,
        designed for predictable median calculation.
    """
    data = pd.DataFrame(
        {"load": [90.0, 100.0, 110.0, 120.0, 130.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1h"),
    )
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T02:00:00"),
    )


@pytest.fixture
def sample_forecaster_config() -> ConstantMedianForecasterConfig:
    """Create sample forecaster configuration with standard quantiles."""
    return ConstantMedianForecasterConfig(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime(timedelta(hours=6))],
        hyperparams=ConstantMedianForecasterHyperParams(constant=5.0),
    )


def test_constant_median_forecaster__fit_predict(
    sample_forecaster_config: ConstantMedianForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster trains on data and produces constant predictions with added hyperparameter."""
    # Arrange
    forecaster = ConstantMedianForecaster(config=sample_forecaster_config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) == 7  # Only forecasts after forecast_start (2025-01-01T02:00:00)

    # Check quantile values: quantiles of [90, 100, 110, 120, 130] plus constant 5.0
    expected_p10 = 99.0  # 94 + 5
    expected_median = 115.0  # 110 + 5
    expected_p90 = 131.0  # 126 + 5

    actual_values = result.data.iloc[0]  # First forecast row
    assert actual_values["quantile_P10"] == expected_p10
    assert actual_values["quantile_P50"] == expected_median
    assert actual_values["quantile_P90"] == expected_p90


def test_constant_median_forecaster__predict_not_fitted_raises_error(
    sample_forecaster_config: ConstantMedianForecasterConfig,
):
    """Test that predicting without fitting raises ModelNotFittedError."""
    # Arrange
    forecaster = ConstantMedianForecaster(config=sample_forecaster_config)
    dummy_data = pd.DataFrame(
        {"load": [100.0]}, index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=1, freq="1h")
    )
    input_dataset = ForecastInputDataset(data=dummy_data, sample_interval=timedelta(hours=1), target_column="load")

    # Act & Assert
    with pytest.raises(NotFittedError, match="ConstantMedianForecaster"):
        forecaster.predict(input_dataset)
