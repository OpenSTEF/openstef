# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelNotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.constant_median_forecaster import (
    ConstantMedianForecaster,
    ConstantMedianForecasterConfig,
    ConstantMedianForecasterHyperParams,
    ConstantMedianHorizonForecaster,
)
from openstef_models.models.forecasting.multi_horizon_adapter import MultiHorizonForecasterConfig


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
    forecaster.fit_horizon(sample_forecast_input_dataset)
    result = forecaster.predict_horizon(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) == 2  # Only forecasts after forecast_start (2025-01-01T02:00:00)

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
    with pytest.raises(ModelNotFittedError, match="ConstantMedianForecaster"):
        forecaster.predict_horizon(input_dataset)


def test_constant_median_forecaster__state_serialize_restore(
    sample_forecaster_config: ConstantMedianForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster state can be serialized and restored with preserved functionality."""
    # Arrange
    original_forecaster = ConstantMedianForecaster(config=sample_forecaster_config)
    original_forecaster.fit_horizon(sample_forecast_input_dataset)

    # Act
    # Serialize state and create new forecaster from state
    state = original_forecaster.get_state()
    restored_forecaster = ConstantMedianForecaster.from_state(state)

    # Assert
    # Check that restored forecaster produces identical predictions
    assert restored_forecaster.is_fitted
    original_result = original_forecaster.predict_horizon(sample_forecast_input_dataset)
    restored_result = restored_forecaster.predict_horizon(sample_forecast_input_dataset)

    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval


@pytest.fixture
def multi_horizon_config() -> MultiHorizonForecasterConfig[ConstantMedianForecasterConfig]:
    """Create multi-horizon forecaster configuration for testing."""
    return MultiHorizonForecasterConfig[ConstantMedianForecasterConfig](
        horizons=[LeadTime(timedelta(hours=3)), LeadTime(timedelta(hours=6))],
        quantiles=[Quantile(0.5)],
        forecaster_config=ConstantMedianForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=[LeadTime(timedelta(hours=1))],  # Will be overridden
            hyperparams=ConstantMedianForecasterHyperParams(constant=0.0),  # No constant for clearer results
        ),
    )


@pytest.fixture
def multi_horizon_input_data() -> dict[LeadTime, ForecastInputDataset]:
    """Create horizon-specific input data with different load patterns for testing."""
    # Create horizon-specific data with different medians for distinguishable results
    horizon_3h_data = pd.DataFrame(
        {"load": [100.0, 200.0, 300.0]},  # Median = 200
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    horizon_6h_data = pd.DataFrame(
        {"load": [500.0, 600.0, 700.0]},  # Median = 600
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )

    return {
        LeadTime(timedelta(hours=3)): ForecastInputDataset(
            data=horizon_3h_data, sample_interval=timedelta(hours=1), target_column="load"
        ),
        LeadTime(timedelta(hours=6)): ForecastInputDataset(
            data=horizon_6h_data, sample_interval=timedelta(hours=1), target_column="load"
        ),
    }


def test_constant_median_horizon_forecaster__fit_predict(
    multi_horizon_config: MultiHorizonForecasterConfig[ConstantMedianForecasterConfig],
    multi_horizon_input_data: dict[LeadTime, ForecastInputDataset],
):
    """Test ConstantMedianHorizonForecaster fit and predict workflow."""
    # Arrange
    multi_forecaster = ConstantMedianHorizonForecaster.create(multi_horizon_config)

    # Act
    multi_forecaster.fit(multi_horizon_input_data)
    result = multi_forecaster.predict(multi_horizon_input_data)

    # Assert
    assert multi_forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) > 0

    # Verify predictions contain expected median values from horizon-specific data
    result_values = result.data["quantile_P50"].tolist()
    assert all(value in {200.0, 600.0} for value in result_values), f"Unexpected values: {result_values}"


def test_constant_median_horizon_forecaster__state_serialize_restore(
    multi_horizon_config: MultiHorizonForecasterConfig[ConstantMedianForecasterConfig],
    multi_horizon_input_data: dict[LeadTime, ForecastInputDataset],
):
    """Test ConstantMedianHorizonForecaster state serialization and restoration."""
    # Arrange
    multi_forecaster = ConstantMedianHorizonForecaster.create(multi_horizon_config)
    multi_forecaster.fit(multi_horizon_input_data)
    original_result = multi_forecaster.predict(multi_horizon_input_data)

    # Act
    state = multi_forecaster.get_state()
    restored_forecaster = ConstantMedianHorizonForecaster.from_state(state)
    restored_result = restored_forecaster.predict(multi_horizon_input_data)

    # Assert
    assert restored_forecaster.is_fitted
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval
