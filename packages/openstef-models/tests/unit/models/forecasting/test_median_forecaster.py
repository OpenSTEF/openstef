# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.median_forecaster import (
    MedianForecaster,
    MedianForecasterConfig,
    MedianForecasterHyperParams,
)


@pytest.fixture
def sample_forecast_input_dataset() -> ForecastInputDataset:
    """Create sample input dataset with lag features for median forecaster training and prediction.

    Returns:
        ForecastInputDataset with load values and lag features T-1min, T-2min, T-3min.
    """
    data = pd.DataFrame(
        {
            "load": [90.0, 100.0, 110.0, 120.0, 130.0],
            "T-1min": [1.0, np.nan, np.nan, np.nan, np.nan],
            "T-2min": [4.0, 1.0, np.nan, np.nan, np.nan],
            "T-3min": [7.0, 4.0, 1.0, np.nan, np.nan],
            "unrelated_feature": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1min"),
    )
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(minutes=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T00:02:00"),
    )


@pytest.fixture
def sample_forecaster_config() -> MedianForecasterConfig:
    """Create sample forecaster configuration for median forecaster."""
    return MedianForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        hyperparams=MedianForecasterHyperParams(),
    )


def test_median_forecaster__fit_predict(
    sample_forecaster_config: MedianForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster fits on data and produces median predictions."""
    # Arrange
    forecaster = MedianForecaster(config=sample_forecaster_config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Check that model is fitted and produces forecast
    assert forecaster.is_fitted
    assert isinstance(result, ForecastDataset)
    assert len(result.data) == 2  # Only forecasts after forecast_start (2025-01-01T00:02:00), so 00:03:00 and 00:04:00

    # Check median predictions
    # For the data pattern with lag features:
    # At time 00:00: lags are [1, 4, 7] -> median = 4.0
    # At time 00:01: lags are [nan, 1, 4] -> median = 2.5  
    # At time 00:02: lags are [nan, nan, 1] -> median = 1.0
    # At time 00:03: lags are [nan, nan, nan] -> median = nan (but autoregression should fill this)
    # At time 00:04: lags are [nan, nan, nan] -> median = nan (but autoregression should fill this)
    
    # The results are filtered to only include times after forecast_start (00:02:00)
    # So we expect predictions for 00:03:00 and 00:04:00
    # Due to autoregressive behavior, these should be influenced by earlier medians
    
    actual_values = result.data.iloc[0]  # First forecast row (00:03:00)
    # This should not be NaN due to autoregressive updates
    assert not np.isnan(actual_values["quantile_P50"])


def test_median_forecaster__predict_not_fitted_raises_error(
    sample_forecaster_config: MedianForecasterConfig,
):
    """Test that predicting without fitting raises NotFittedError."""
    # Arrange
    forecaster = MedianForecaster(config=sample_forecaster_config)
    dummy_data = pd.DataFrame(
        {
            "load": [100.0],
            "T-1min": [50.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=1, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=dummy_data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act & Assert
    with pytest.raises(NotFittedError, match="MedianForecaster"):
        forecaster.predict(input_dataset)


def test_median_forecaster__state_serialize_restore(
    sample_forecaster_config: MedianForecasterConfig,
    sample_forecast_input_dataset: ForecastInputDataset,
):
    """Test that forecaster state can be serialized and restored with preserved functionality."""
    # Arrange
    original_forecaster = MedianForecaster(config=sample_forecaster_config)
    original_forecaster.fit(sample_forecast_input_dataset)

    # Act
    # Serialize state and create new forecaster from state
    state = original_forecaster.to_state()

    restored_forecaster = MedianForecaster(config=sample_forecaster_config)
    restored_forecaster = restored_forecaster.from_state(state)

    # Assert
    # Check that restored forecaster produces identical predictions
    assert restored_forecaster.is_fitted
    original_result = original_forecaster.predict(sample_forecast_input_dataset)
    restored_result = restored_forecaster.predict(sample_forecast_input_dataset)

    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert original_result.sample_interval == restored_result.sample_interval


def test_median_forecaster__extract_and_validate_lags():
    """Test lag feature extraction and validation."""
    # Arrange
    data = pd.DataFrame(
        {
            "T-1min": [1.0, 2.0, 3.0],
            "T-2min": [4.0, 5.0, 6.0],
            "T-3min": [7.0, 8.0, 9.0],
            "other_feature": [10.0, 11.0, 12.0],
        }
    )

    # Act
    feature_names, frequency, feature_to_lags = MedianForecaster._extract_and_validate_lags(data)

    # Assert
    assert feature_names == ["T-1min", "T-2min", "T-3min"]
    assert frequency == 1  # 1 minute spacing
    assert feature_to_lags == {"T-1min": 1, "T-2min": 2, "T-3min": 3}


def test_median_forecaster__extract_and_validate_lags_days():
    """Test lag feature extraction with day-based lags."""
    # Arrange
    data = pd.DataFrame(
        {
            "T-1d": [1.0, 2.0, 3.0],
            "T-2d": [4.0, 5.0, 6.0],
            "other_feature": [10.0, 11.0, 12.0],
        }
    )

    # Act
    feature_names, frequency, feature_to_lags = MedianForecaster._extract_and_validate_lags(data)

    # Assert
    assert feature_names == ["T-1d", "T-2d"]
    assert frequency == 1440  # 1 day = 1440 minutes spacing
    assert feature_to_lags == {"T-1d": 1440, "T-2d": 2880}


def test_median_forecaster__extract_and_validate_lags_no_lag_features():
    """Test that error is raised when no lag features are found."""
    # Arrange
    data = pd.DataFrame({"other_feature": [1.0, 2.0, 3.0]})

    # Act & Assert
    with pytest.raises(ValueError, match="No lag features found"):
        MedianForecaster._extract_and_validate_lags(data)


def test_median_forecaster__extract_and_validate_lags_uneven_spacing():
    """Test that error is raised when lag features are not evenly spaced."""
    # Arrange
    data = pd.DataFrame(
        {
            "T-1min": [1.0, 2.0, 3.0],
            "T-3min": [4.0, 5.0, 6.0],  # Gap: 2min spacing instead of 1min
            "T-4min": [7.0, 8.0, 9.0],
        }
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Lag features are not evenly spaced"):
        MedianForecaster._extract_and_validate_lags(data)


def test_median_forecaster__extract_and_validate_lags_invalid_format():
    """Test that error is raised for invalid lag feature names."""
    # Arrange
    data = pd.DataFrame({"T-invalid": [1.0, 2.0, 3.0]})

    # Act & Assert
    with pytest.raises(ValueError, match="does not follow the expected format"):
        MedianForecaster._extract_and_validate_lags(data)


def test_median_forecaster__median_calculation():
    """Test median calculation with various lag patterns."""
    # Arrange
    forecaster = MedianForecaster()
    data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0],  # Add required target column
            "T-1min": [1.0, 2.0, 3.0],
            "T-2min": [4.0, 5.0, 6.0],
            "T-3min": [7.0, 8.0, 9.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act
    forecaster.fit(input_dataset)
    result = forecaster.predict(input_dataset)

    # Assert
    # Expected medians: [4, 5, 6] (median of each row)
    expected_medians = [4.0, 5.0, 6.0]
    actual_medians = result.data["quantile_P50"].values
    np.testing.assert_allclose(actual_medians, expected_medians)


def test_median_forecaster__handles_missing_data():
    """Test that forecaster handles missing data (NaN values) correctly."""
    # Arrange
    forecaster = MedianForecaster()
    data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0],  # Add required target column
            "T-1min": [1.0, np.nan, np.nan],
            "T-2min": [np.nan, 1.0, np.nan],
            "T-3min": [3.0, np.nan, 1.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act
    forecaster.fit(input_dataset)
    result = forecaster.predict(input_dataset)

    # Assert
    # Expected medians with autoregressive behavior:
    # Row 0: median([1, nan, 3]) = 2.0
    # Row 1: median([nan, 1, nan]) = 1.0, but autoregression may have filled some lag values
    # Row 2: median([nan, nan, 1]) = 1.0, but autoregression may have filled some lag values
    # Due to autoregressive updates, later predictions are influenced by earlier ones
    expected_medians = [2.0, 1.5, 1.5]  # Adjusted for autoregressive behavior
    actual_medians = result.data["quantile_P50"].values
    np.testing.assert_allclose(actual_medians, expected_medians)


def test_median_forecaster__handles_all_nan():
    """Test that forecaster handles cases where all lag features are NaN."""
    # Arrange
    forecaster = MedianForecaster()
    data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0],  # Add required target column
            "T-1min": [1.0, np.nan, np.nan],
            "T-2min": [2.0, np.nan, np.nan],
            "T-3min": [3.0, np.nan, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act
    forecaster.fit(input_dataset)
    result = forecaster.predict(input_dataset)

    # Assert
    # First row: median([1, 2, 3]) = 2.0
    # Subsequent rows have all NaN initially, but autoregressive behavior fills them
    # with previous prediction values, so they won't be NaN
    expected_first = 2.0
    actual_values = result.data["quantile_P50"].values
    assert actual_values[0] == expected_first
    # Due to autoregressive behavior, subsequent values should not be NaN
    assert not np.isnan(actual_values[1])
    assert not np.isnan(actual_values[2])


def test_median_forecaster__frequency_mismatch_error():
    """Test that error is raised when input data frequency doesn't match model frequency."""
    # Arrange
    forecaster = MedianForecaster()
    
    # Fit with 1-minute frequency
    fit_data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0],  # Add required target column
            "T-1min": [1.0, 2.0, 3.0],
            "T-2min": [4.0, 5.0, 6.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1min"),
    )
    fit_dataset = ForecastInputDataset(
        data=fit_data, sample_interval=timedelta(minutes=1), target_column="load"
    )
    forecaster.fit(fit_dataset)
    
    # Try to predict with 5-minute frequency
    predict_data = pd.DataFrame(
        {
            "load": [100.0, 200.0],  # Add required target column
            "T-1min": [1.0, 2.0],
            "T-2min": [4.0, 5.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T01:00:00"), periods=2, freq="5min"),
    )
    predict_dataset = ForecastInputDataset(
        data=predict_data, sample_interval=timedelta(minutes=5), target_column="load"
    )

    # Act & Assert
    with pytest.raises(ValueError, match="input data frequency does not match"):
        forecaster.predict(predict_dataset)


def test_median_forecaster__missing_lag_features_error():
    """Test that error is raised when required lag features are missing from prediction data."""
    # Arrange
    forecaster = MedianForecaster()
    
    # Fit with multiple lag features
    fit_data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0],  # Add required target column
            "T-1min": [1.0, 2.0, 3.0],
            "T-2min": [4.0, 5.0, 6.0],
            "T-3min": [7.0, 8.0, 9.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1min"),
    )
    fit_dataset = ForecastInputDataset(
        data=fit_data, sample_interval=timedelta(minutes=1), target_column="load"
    )
    forecaster.fit(fit_dataset)
    
    # Try to predict with missing lag feature
    predict_data = pd.DataFrame(
        {
            "load": [100.0, 200.0],  # Add required target column
            "T-1min": [1.0, 2.0],
            "T-2min": [4.0, 5.0],
            # Missing T-3min
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T01:00:00"), periods=2, freq="1min"),
    )
    predict_dataset = ForecastInputDataset(
        data=predict_data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act & Assert
    with pytest.raises(ValueError, match="missing the following lag features"):
        forecaster.predict(predict_dataset)


def test_median_forecaster__autoregressive_behavior():
    """Test that the forecaster exhibits autoregressive behavior."""
    # Arrange
    forecaster = MedianForecaster()
    
    # Create data where autoregressive behavior would be observable
    # Set up scenario where later predictions depend on earlier ones
    data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0, 400.0, 500.0],  # Add required target column
            "T-1min": [10.0, np.nan, np.nan, np.nan, np.nan],
            "T-2min": [20.0, 10.0, np.nan, np.nan, np.nan],
            "T-3min": [30.0, 20.0, 10.0, np.nan, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=data, sample_interval=timedelta(minutes=1), target_column="load"
    )

    # Act
    forecaster.fit(input_dataset)
    result = forecaster.predict(input_dataset)

    # Assert
    # The autoregressive nature means that predictions should propagate forward
    # First prediction: median([10, 20, 30]) = 20
    # Subsequent predictions should use the previous predictions as inputs
    predictions = result.data["quantile_P50"].values
    assert predictions[0] == 20.0  # median of [10, 20, 30]
    
    # Later predictions should reflect the autoregressive updates
    # We expect them to be influenced by the initial median value
    assert not np.isnan(predictions[1])
    assert not np.isnan(predictions[2])


def test_median_forecaster__config_only_supports_median():
    """Test that configuration only supports median quantile."""
    # Arrange & Act
    config = MedianForecasterConfig()
    
    # Assert
    assert len(config.quantiles) == 1
    assert config.quantiles[0] == Quantile(0.5)


def test_median_forecaster__infer_frequency():
    """Test frequency inference from datetime index."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=5, freq="15min")
    
    # Act
    inferred_freq = MedianForecaster._infer_frequency(index)
    
    # Assert
    assert inferred_freq == pd.Timedelta(minutes=15)


def test_median_forecaster__infer_frequency_insufficient_data():
    """Test that error is raised when trying to infer frequency from insufficient data."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=1, freq="15min")
    
    # Act & Assert
    with pytest.raises(ValueError, match="Cannot infer frequency from an index with fewer than 2 timestamps"):
        MedianForecaster._infer_frequency(index)


def test_median_forecaster__forecast_start_filtering():
    """Test that predictions are filtered based on forecast_start."""
    # Arrange
    forecaster = MedianForecaster()
    data = pd.DataFrame(
        {
            "load": [100.0, 200.0, 300.0, 400.0, 500.0],  # Add required target column
            "T-1min": [1.0, 2.0, 3.0, 4.0, 5.0],
            "T-2min": [2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1min"),
    )
    input_dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(minutes=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T00:02:00"),  # Start from 3rd timestamp
    )

    # Act
    forecaster.fit(input_dataset)
    result = forecaster.predict(input_dataset)

    # Assert
    # Should only return predictions after forecast_start (3 predictions)
    assert len(result.data) == 2  # 00:03:00 and 00:04:00 (after 00:02:00)
    # Index should start from 00:03:00
    assert result.data.index[0] == pd.Timestamp("2025-01-01T00:03:00")