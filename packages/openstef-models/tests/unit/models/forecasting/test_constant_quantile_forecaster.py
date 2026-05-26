# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.constant_quantile_forecaster import (
    ConstantQuantileForecaster,
    ConstantQuantileForecasterHyperParams,
)


def create_forecast_input_dataset(values: list[float]) -> ForecastInputDataset:
    data = pd.DataFrame(
        {"load": values},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1h"),
    )
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T02:00:00"),
    )


@pytest.fixture
def sample_forecaster() -> ConstantQuantileForecaster:
    """Create sample forecaster configuration with standard quantiles."""
    return ConstantQuantileForecaster(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime(timedelta(hours=6))],
        hyperparams=ConstantQuantileForecasterHyperParams(constant=5.0),
    )


@pytest.mark.parametrize(
    ("values", "expected_quantile_values"),
    [
        pytest.param([90.0, 100.0, 110.0, 120.0, 130.0], [99, 115, 131], id="simple"),
        pytest.param([100.0, 100.0, 100.0, 100.0, 100.0], [105, 105, 105], id="constant"),
        pytest.param([90.0, 100.0, np.nan, 120.0, 130.0], [98, 115, 132], id="single_nan"),
    ],
)
def test_constant_quantile_forecaster__fit_predict(
    values: list[float],
    expected_quantile_values: list[float],
    sample_forecaster: ConstantQuantileForecaster,
):
    """Test that forecaster trains on data and produces constant predictions with added hyperparameter."""
    # Arrange
    forecaster = sample_forecaster.model_copy(deep=True)
    sample_forecast_input_dataset = create_forecast_input_dataset(values)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) == 7  # Only forecasts after forecast_start (2025-01-01T02:00:00)

    expected_p10, expected_median, expected_p90 = expected_quantile_values
    actual_values = result.data.iloc[0]  # First forecast row
    assert np.isfinite(actual_values).all()
    assert math.isclose(actual_values["quantile_P10"], expected_p10)
    assert math.isclose(actual_values["quantile_P50"], expected_median)
    assert math.isclose(actual_values["quantile_P90"], expected_p90)


def test_constant_quantile_forecaster__fit_with_all_nan_target_raises_error(
    sample_forecaster: ConstantQuantileForecaster,
):
    """Test that fitting with all NaN target values raises InputValidationError."""
    # Arrange
    forecaster = sample_forecaster.model_copy(deep=True)
    nan_values = [float("nan")] * 5
    sample_forecast_input_dataset = create_forecast_input_dataset(nan_values)

    # Act & Assert
    with pytest.raises(
        ValueError, match=r"Training data must contain at least one non-NaN value in the target column."
    ):
        forecaster.fit(sample_forecast_input_dataset)


def test_constant_quantile_forecaster__predict_not_fitted_raises_error(
    sample_forecaster: ConstantQuantileForecaster,
):
    """Test that predicting without fitting raises ModelNotFittedError."""
    # Arrange
    forecaster = sample_forecaster.model_copy(deep=True)
    dummy_data = pd.DataFrame(
        {"load": [100.0]}, index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=1, freq="1h")
    )
    input_dataset = ForecastInputDataset(data=dummy_data, sample_interval=timedelta(hours=1), target_column="load")

    # Act & Assert
    with pytest.raises(NotFittedError, match="ConstantQuantileForecaster"):
        forecaster.predict(input_dataset)
