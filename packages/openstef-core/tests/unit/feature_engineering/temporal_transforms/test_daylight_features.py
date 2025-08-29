# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the DaylightFeatures transform."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.temporal.daylight_features import DaylightFeatures

pvlib = pytest.importorskip("pvlib")


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with daily frequency and timezone-aware index.
    """
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0, 130.0, 140.0]},
        index=pd.date_range("2025-06-01", periods=5, freq="D", tz="Europe/Amsterdam"),
    )
    return TimeSeriesDataset(data, timedelta(days=1))


@pytest.fixture
def sample_dataset_no_tz() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset without timezone for testing error cases.

    Returns:
        TimeSeriesDataset: A dataset with daily frequency but no timezone.
    """
    data = pd.DataFrame({"load": [100.0, 110.0, 120.0]}, index=pd.date_range("2025-06-01", periods=3, freq="D"))
    return TimeSeriesDataset(data, timedelta(days=1))


@pytest.fixture
def mock_clearsky_data() -> pd.DataFrame:
    """Mock clearsky data for testing.

    Returns:
        A DataFrame with GHI values for testing.
    """
    return pd.DataFrame(
        {"ghi": [0.0, 500.0, 800.0, 600.0, 200.0]},
        index=pd.date_range("2025-06-01", periods=5, freq="D", tz="Europe/Amsterdam"),
    )


def test_daylight_features_initialization() -> None:
    """Test DaylightFeatures can be initialized properly."""
    # Arrange
    latitude = 52.0
    longitude = 5.0

    # Act
    transform = DaylightFeatures(latitude=latitude, longitude=longitude)

    # Assert
    assert transform.latitude == latitude
    assert transform.longitude == longitude
    assert transform._daylight_continuous.empty


def test_daylight_features_initialization_parameters() -> None:
    """Test DaylightFeatures initialization with different coordinates."""
    # Arrange
    latitude = 52.0
    longitude = 5.0

    # Act
    transform = DaylightFeatures(latitude=latitude, longitude=longitude)

    # Assert
    assert transform.latitude == latitude
    assert transform.longitude == longitude


def test_fit_creates_daylight_features(sample_dataset: TimeSeriesDataset, mock_clearsky_data: pd.DataFrame) -> None:
    """Test that fit creates expected daylight features."""
    # Arrange
    with patch("openstef_core.feature_engineering.temporal_transforms.daylight_features.pvlib") as mock_pvlib:
        mock_location = Mock()
        mock_location.get_clearsky.return_value = mock_clearsky_data
        mock_pvlib.location.Location.return_value = mock_location

        transform = DaylightFeatures(latitude=52.0, longitude=5.0)

        # Act
        transform.fit(sample_dataset)

    # Assert
    assert not transform._daylight_continuous.empty
    assert "daylight_continuous" in transform._daylight_continuous.columns
    assert len(transform._daylight_continuous) == len(sample_dataset.index)

    # Verify pvlib was called correctly
    mock_pvlib.location.Location.assert_called_once_with(52.0, 5.0, tz=str(sample_dataset.index.tz))
    mock_location.get_clearsky.assert_called_once_with(sample_dataset.index)


def test_fit_timezone_error(sample_dataset_no_tz: TimeSeriesDataset) -> None:
    """Test that fit raises ValueError for non-timezone-aware index."""
    # Arrange
    transform = DaylightFeatures(latitude=52.0, longitude=5.0)

    # Act & Assert
    with pytest.raises(ValueError, match="The datetime index must be timezone-aware"):
        transform.fit(sample_dataset_no_tz)


def test_transform_adds_features(sample_dataset: TimeSeriesDataset, mock_clearsky_data: pd.DataFrame) -> None:
    """Test that transform adds daylight features to the dataset."""
    # Arrange
    with patch("openstef_core.feature_engineering.temporal_transforms.daylight_features.pvlib") as mock_pvlib:
        mock_location = Mock()
        mock_location.get_clearsky.return_value = mock_clearsky_data
        mock_pvlib.location.Location.return_value = mock_location

        transform = DaylightFeatures(latitude=52.0, longitude=5.0)
        transform.fit(sample_dataset)

    # Act
    result = transform.transform(sample_dataset)

    # Assert
    # Check structure
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval

    # Check that original features are preserved
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])

    # Check that daylight features are added
    assert "daylight_continuous" in result.data.columns
    assert len(result.data.columns) == len(sample_dataset.data.columns) + 1


def test_fit_transform_integration(sample_dataset: TimeSeriesDataset) -> None:
    """Test the complete fit_transform workflow with mocked pvlib."""
    # Arrange
    mock_clearsky_data = pd.DataFrame({"ghi": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=sample_dataset.index)

    with patch("openstef_core.feature_engineering.temporal_transforms.daylight_features.pvlib") as mock_pvlib:
        mock_location = Mock()
        mock_location.get_clearsky.return_value = mock_clearsky_data
        mock_pvlib.location.Location.return_value = mock_location

        transform = DaylightFeatures(latitude=52.0, longitude=5.0)

        # Act
        result = transform.fit_transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "daylight_continuous" in result.data.columns

    # Check that daylight values are correctly mapped
    expected_values = [100.0, 200.0, 300.0, 400.0, 500.0]
    actual_values = result.data["daylight_continuous"].tolist()
    assert actual_values == expected_values


@pytest.mark.parametrize(
    "timezone",
    [
        pytest.param("Europe/Amsterdam", id="amsterdam"),
        pytest.param("US/Eastern", id="eastern"),
        pytest.param("Asia/Tokyo", id="tokyo"),
        pytest.param("UTC", id="utc"),
    ],
)
def test_different_timezones(timezone: str) -> None:
    """Test DaylightFeatures with different timezones."""
    # Arrange
    data = pd.DataFrame({"load": [100.0, 110.0]}, index=pd.date_range("2025-06-01", periods=2, freq="D", tz=timezone))
    dataset = TimeSeriesDataset(data, timedelta(days=1))

    mock_clearsky_data = pd.DataFrame({"ghi": [200.0, 300.0]}, index=data.index)

    with patch("openstef_core.feature_engineering.temporal_transforms.daylight_features.pvlib") as mock_pvlib:
        mock_location = Mock()
        mock_location.get_clearsky.return_value = mock_clearsky_data
        mock_pvlib.location.Location.return_value = mock_location

        transform = DaylightFeatures(latitude=52.0, longitude=5.0)

        # Act
        result = transform.fit_transform(dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "daylight_continuous" in result.data.columns

    # Verify timezone was passed correctly to pvlib
    mock_pvlib.location.Location.assert_called_once_with(52.0, 5.0, tz=timezone)


def test_empty_dataset() -> None:
    """Test handling of empty dataset."""
    # Arrange
    empty_data = pd.DataFrame({"load": []}, index=pd.DatetimeIndex([], tz="Europe/Amsterdam"))
    empty_dataset = TimeSeriesDataset(empty_data, timedelta(days=1))

    mock_clearsky_data = pd.DataFrame({"ghi": []}, index=empty_data.index)

    with patch("openstef_core.feature_engineering.temporal_transforms.daylight_features.pvlib") as mock_pvlib:
        mock_location = Mock()
        mock_location.get_clearsky.return_value = mock_clearsky_data
        mock_pvlib.location.Location.return_value = mock_location

        transform = DaylightFeatures(latitude=52.0, longitude=5.0)

        # Act
        result = transform.fit_transform(empty_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert len(result.data) == 0


# Integration tests with real pvlib calls (no mocking)


@pytest.mark.parametrize(
    ("latitude", "longitude", "timezone"),
    [
        pytest.param(52.0, 5.0, "Europe/Amsterdam", id="netherlands"),
        pytest.param(40.7, -74.0, "US/Eastern", id="new_york"),
        pytest.param(-33.9, 18.4, "Africa/Johannesburg", id="cape_town"),
    ],
)
def test_pvlib_location_creation_with_timezone(latitude: float, longitude: float, timezone: str) -> None:
    """Test that pvlib Location object can be correctly created with timezone objects."""
    # Act
    location = pvlib.location.Location(latitude, longitude, tz=timezone)

    # Assert
    assert location.latitude == latitude
    assert location.longitude == longitude
    assert str(location.tz) == timezone


def test_pvlib_integration_with_real_data() -> None:
    """Test complete DaylightFeatures workflow with real pvlib calls."""
    # Arrange
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range("2025-06-01 10:00", periods=3, freq="2h", tz="Europe/Amsterdam"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=2))
    transform = DaylightFeatures(latitude=52.0, longitude=5.0)

    # Act
    result = transform.fit_transform(dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "daylight_continuous" in result.data.columns
    assert len(result.data) == len(dataset.data)

    # Check that daylight values are realistic (should be positive during daytime)
    daylight_values = result.data["daylight_continuous"]
    assert daylight_values.dtype.kind in "fi"  # float or int
    assert (daylight_values >= 0).all()

    # For June in Netherlands at midday, we expect some positive GHI values
    assert daylight_values.max() > 0


@pytest.mark.parametrize(
    "timezone",
    [
        pytest.param("Europe/Amsterdam", id="amsterdam"),
        pytest.param("US/Pacific", id="pacific"),
        pytest.param("Asia/Tokyo", id="tokyo"),
    ],
)
def test_pvlib_different_timezones_integration(timezone: str) -> None:
    """Test DaylightFeatures with real pvlib calls across different timezones."""
    # Arrange
    data = pd.DataFrame(
        {"load": [100.0, 110.0]}, index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz=timezone)
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = DaylightFeatures(latitude=52.0, longitude=5.0)

    # Act
    result = transform.fit_transform(dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "daylight_continuous" in result.data.columns
    assert len(result.data) == len(dataset.data)

    # Verify that the timezone is properly handled
    assert result.data.index.tz is not None
    assert str(result.data.index.tz) == timezone
