# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the RadiationDerivedFeatures transform."""

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.weather_transforms.radiation_derived_features import (
    RadiationDerivedFeatures,
)

pvlib = pytest.importorskip("pvlib")


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset with radiation data for testing.

    Returns:
        TimeSeriesDataset: A dataset with hourly frequency, timezone-aware index, and radiation data in J/m².
    """
    data = pd.DataFrame(
        {"radiation": [3600000, 7200000, 5400000, 1800000, 0]},  # J/m² values
        index=pd.date_range("2025-06-01 08:00", periods=5, freq="h", tz="Europe/Amsterdam"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def sample_dataset_no_tz() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset without timezone for testing error cases.

    Returns:
        TimeSeriesDataset: A dataset with hourly frequency but no timezone.
    """
    data = pd.DataFrame(
        {"radiation": [3600000, 7200000, 5400000]}, index=pd.date_range("2025-06-01", periods=3, freq="h")
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def sample_dataset_no_radiation() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset without radiation feature for testing error cases.

    Returns:
        TimeSeriesDataset: A dataset with load data but no radiation feature.
    """
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range("2025-06-01", periods=3, freq="h", tz="Europe/Amsterdam"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def basic_transform() -> RadiationDerivedFeatures:
    """
    Create a basic RadiationDerivedFeatures.

    Returns:
        RadiationDerivedFeatures: An instance of the transform with default parameters.
    """
    return RadiationDerivedFeatures(
        latitude=52.3676,
        longitude=4.9041,
        included_features=["dni", "gti"],
        surface_tilt=34.0,
        surface_azimuth=180.0,
    )


@pytest.fixture
def mock_solar_position() -> pd.DataFrame:
    """Mock solar position data for testing.

    Returns:
        A DataFrame with solar position data.
    """
    return pd.DataFrame(
        {
            "apparent_zenith": [85.0, 45.0, 30.0, 60.0, 90.0],
            "azimuth": [90.0, 180.0, 180.0, 270.0, 0.0],
        },
        index=pd.date_range("2025-06-01 08:00", periods=5, freq="h", tz="Europe/Amsterdam"),
    )


@pytest.fixture
def mock_clearsky_radiation() -> pd.DataFrame:
    """Mock clearsky radiation data for testing.

    Returns:
        A DataFrame with clearsky radiation values.
    """
    return pd.DataFrame(
        {
            "ghi": [100.0, 800.0, 1000.0, 600.0, 50.0],
            "dni": [200.0, 900.0, 1100.0, 700.0, 100.0],
            "dhi": [50.0, 100.0, 80.0, 150.0, 30.0],
        },
        index=pd.date_range("2025-06-01 08:00", periods=5, freq="h", tz="Europe/Amsterdam"),
    )


def test_check_feature_exists_method(sample_dataset: TimeSeriesDataset):
    """Test the _check_feature_exists static method."""
    # Test existing feature
    assert RadiationDerivedFeatures._check_feature_exists(sample_dataset, "radiation") is True

    # Test non-existing feature
    assert RadiationDerivedFeatures._check_feature_exists(sample_dataset, "nonexistent") is False


def test_fit_with_timezone_aware_dataset(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeatures,
    mock_solar_position: pd.DataFrame,
    mock_clearsky_radiation: pd.DataFrame,
):
    """Test that fit works correctly with timezone-aware dataset."""
    with (
        patch("pvlib.solarposition.get_solarposition", return_value=mock_solar_position),
        patch("pvlib.location.Location.get_clearsky", return_value=mock_clearsky_radiation),
        patch("pvlib.irradiance.dni", return_value=pd.Series([0.0, 0.5, 1.0, 0.3, 0.0], name="dni")),
        patch("pvlib.irradiance.get_total_irradiance") as mock_gti,
    ):
        # Arrange
        mock_gti.return_value = {"poa_global": pd.Series([0.0, 0.6, 1.2, 0.4, 0.0], name="gti")}

        # Act
        basic_transform.fit(sample_dataset)

        # Assert
        assert not basic_transform._derived_features.empty
        assert "dni" in basic_transform._derived_features.columns
        assert "gti" in basic_transform._derived_features.columns
        assert len(basic_transform._derived_features) == len(sample_dataset.data)


def test_fit_raises_error_for_non_timezone_aware_dataset(
    sample_dataset_no_tz: TimeSeriesDataset, basic_transform: RadiationDerivedFeatures
):
    """Test that fit raises ValueError for non-timezone-aware dataset."""
    # Act & Assert
    with pytest.raises(ValueError, match="The datetime index must be timezone-aware"):
        basic_transform.fit(sample_dataset_no_tz)


def test_fit_handles_missing_radiation_feature(
    sample_dataset_no_radiation: TimeSeriesDataset, basic_transform: RadiationDerivedFeatures
):
    """Test that fit handles missing radiation feature gracefully."""
    # Act
    with patch("openstef_core.feature_engineering.weather_transforms.radiation_derived_features.logger") as mock_logger:
        basic_transform.fit(sample_dataset_no_radiation)

    # Assert
    assert basic_transform._derived_features.empty
    mock_logger.warning.assert_called_with(
        "Skipping calculation of radiation derived features because 'radiation' feature is missing."
    )


def test_fit_with_no_included_features(sample_dataset: TimeSeriesDataset):
    """Test that fit handles configuration with no included features."""
    # Arrange
    transform = RadiationDerivedFeatures(latitude=52.0, longitude=5.0, included_features=[])

    # Act
    with patch("openstef_core.feature_engineering.weather_transforms.radiation_derived_features.logger") as mock_logger:
        transform.fit(sample_dataset)

    # Assert
    mock_logger.warning.assert_called_with("No radiation derived features selected to include.")


@pytest.mark.parametrize(
    ("included_features", "expected_columns"),
    [
        (["dni"], ["dni"]),
        (["gti"], ["gti"]),
        (["dni", "gti"], ["dni", "gti"]),
    ],
)
def test_fit_with_different_feature_combinations(
    sample_dataset: TimeSeriesDataset,
    included_features: list,
    expected_columns: list,
    mock_solar_position: pd.DataFrame,
    mock_clearsky_radiation: pd.DataFrame,
):
    """Test fit with different combinations of included features."""
    with (
        patch("pvlib.solarposition.get_solarposition", return_value=mock_solar_position),
        patch("pvlib.location.Location.get_clearsky", return_value=mock_clearsky_radiation),
        patch("pvlib.irradiance.dni", return_value=pd.Series([0.0, 0.5, 1.0, 0.3, 0.0], name="dni")),
        patch("pvlib.irradiance.get_total_irradiance") as mock_gti,
    ):
        mock_gti.return_value = {"poa_global": pd.Series([0.0, 0.6, 1.2, 0.4, 0.0], name="gti")}

        # Arrange
        transform = RadiationDerivedFeatures(
            latitude=52.0,
            longitude=5.0,
            included_features=included_features,
        )

        # Act
        transform.fit(sample_dataset)

        # Assert
        assert list(transform._derived_features.columns) == expected_columns


def test_calculate_dni_method(basic_transform: RadiationDerivedFeatures):
    """Test the _calculate_dni private method."""
    # Arrange
    solar_position = pd.DataFrame({
        "apparent_zenith": [30.0, 45.0, 60.0],
    })

    clearsky_radiation = pd.DataFrame({
        "dhi": [100.0, 150.0, 200.0],
        "dni": [800.0, 600.0, 400.0],
    })

    ghi = pd.Series([500.0, 400.0, 300.0])

    with patch("pvlib.irradiance.dni") as mock_dni:
        mock_dni.return_value = pd.Series([400.0, 250.0, 100.0])

        # Act
        result = basic_transform._calculate_dni(solar_position, clearsky_radiation, ghi)

        # Assert
        mock_dni.assert_called_once()
        assert len(result) == 3
        pd.testing.assert_series_equal(result, pd.Series([400.0, 250.0, 100.0]))


def test_calculate_gti_method(basic_transform: RadiationDerivedFeatures):
    """Test the _calculate_gti private method."""
    # Arrange
    solar_position = pd.DataFrame({
        "apparent_zenith": [30.0, 45.0, 60.0],
        "azimuth": [180.0, 180.0, 180.0],
    })

    clearsky_radiation = pd.DataFrame({
        "dhi": [100.0, 150.0, 200.0],
    })

    dni = pd.Series([400.0, 250.0, 100.0])
    ghi = pd.Series([500.0, 400.0, 300.0])

    with patch("pvlib.irradiance.get_total_irradiance") as mock_gti:
        mock_gti.return_value = {"poa_global": pd.Series([450.0, 350.0, 250.0])}

        # Act
        result = basic_transform._calculate_gti(
            solar_position, dni, ghi, clearsky_radiation, surface_tilt=34.0, surface_azimuth=180.0
        )

        # Assert
        mock_gti.assert_called_once()
        assert len(result) == 3
        pd.testing.assert_series_equal(result, pd.Series([450.0, 350.0, 250.0]))


def test_transform_adds_derived_features(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeatures,
):
    """Test that transform adds derived features to the dataset."""
    # Create mock derived features
    mock_derived_features = pd.DataFrame(
        {
            "dni": [0.0, 0.5, 1.0, 0.3, 0.0],
            "gti": [0.0, 0.6, 1.2, 0.4, 0.0],
        },
        index=sample_dataset.index,
    )
    basic_transform._derived_features = mock_derived_features

    # Act
    result = basic_transform.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert len(result.feature_names) == len(sample_dataset.feature_names) + 2
    assert result.sample_interval == sample_dataset.sample_interval

    # Check that original features are preserved
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])

    # Check that derived features are added
    assert "dni" in result.feature_names
    assert "gti" in result.feature_names
    pd.testing.assert_series_equal(result.data["dni"], mock_derived_features["dni"])
    pd.testing.assert_series_equal(result.data["gti"], mock_derived_features["gti"])


def test_transform_with_empty_derived_features(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeatures,
):
    """Test that transform works correctly when no derived features are computed."""
    # Leave _derived_features as empty DataFrame

    # Act
    result = basic_transform.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert len(result.feature_names) == len(sample_dataset.feature_names)
    assert result.sample_interval == sample_dataset.sample_interval
    pd.testing.assert_frame_equal(result.data, sample_dataset.data)


def test_fit_transform_integration(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeatures,
    mock_solar_position: pd.DataFrame,
    mock_clearsky_radiation: pd.DataFrame,
):
    """Test the complete fit_transform workflow."""
    with (
        patch("pvlib.solarposition.get_solarposition", return_value=mock_solar_position),
        patch("pvlib.location.Location.get_clearsky", return_value=mock_clearsky_radiation),
        patch("pvlib.irradiance.dni", return_value=pd.Series([0.0, 0.5, 1.0, 0.3, 0.0], name="dni")),
        patch("pvlib.irradiance.get_total_irradiance") as mock_gti,
    ):
        mock_gti.return_value = {"poa_global": pd.Series([0.0, 0.6, 1.2, 0.4, 0.0], name="gti")}

        # Act
        result = basic_transform.fit_transform(sample_dataset)

        # Assert
        assert isinstance(result, TimeSeriesDataset)
        assert "radiation" in result.feature_names
        assert "dni" in result.feature_names
        assert "gti" in result.feature_names
        assert len(result.data) == len(sample_dataset.data)


def test_radiation_unit_conversion(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeatures,
):
    """Test that radiation is correctly converted from J/m² to kWh/m²."""
    with (
        patch("pvlib.solarposition.get_solarposition"),
        patch("pvlib.location.Location.get_clearsky"),
        patch("pvlib.irradiance.dni") as mock_dni,
        patch("pvlib.irradiance.get_total_irradiance"),
    ):
        mock_dni.return_value = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

        # Act
        basic_transform.fit(sample_dataset)

        # Assert - Check that GHI was called with converted values (J/m² / 3600 = kWh/m²)
        # Original values: [3600000, 7200000, 5400000, 1800000, 0]
        # Expected GHI: [1000.0, 2000.0, 1500.0, 500.0, 0.0]
        call_args = mock_dni.call_args
        ghi_values = call_args.kwargs["ghi"]
        expected_ghi = pd.Series([1000.0, 2000.0, 1500.0, 500.0, 0.0], index=sample_dataset.index)
        pd.testing.assert_series_equal(ghi_values, expected_ghi)


def test_empty_dataset(basic_transform: RadiationDerivedFeatures):
    """Test handling of empty dataset."""
    # Arrange
    data = pd.DataFrame(columns=["radiation"]).astype(float)
    data.index = pd.DatetimeIndex([], tz="Europe/Amsterdam")
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    # Act
    basic_transform.fit(dataset)
    result = basic_transform.transform(dataset)

    # Assert
    assert len(result.data) == 0
    assert len(result.feature_names) == 1  # Only radiation column


@pytest.mark.parametrize(
    ("surface_tilt", "surface_azimuth"),
    [
        (0.0, 180.0),  # Horizontal surface
        (90.0, 180.0),  # Vertical south-facing
        (34.0, 90.0),  # East-facing
        (34.0, 270.0),  # West-facing
    ],
)
def test_different_surface_orientations(
    sample_dataset: TimeSeriesDataset,
    surface_tilt: float,
    surface_azimuth: float,
    mock_solar_position: pd.DataFrame,
    mock_clearsky_radiation: pd.DataFrame,
):
    """Test RadiationDerivedFeatures with different surface orientations."""
    with (
        patch("pvlib.solarposition.get_solarposition", return_value=mock_solar_position),
        patch("pvlib.location.Location.get_clearsky", return_value=mock_clearsky_radiation),
        patch("pvlib.irradiance.dni", return_value=pd.Series([0.0, 0.5, 1.0, 0.3, 0.0], name="dni")),
        patch("pvlib.irradiance.get_total_irradiance") as mock_gti,
    ):
        mock_gti.return_value = {"poa_global": pd.Series([0.0, 0.6, 1.2, 0.4, 0.0], name="gti")}

        # Arrange
        transform = RadiationDerivedFeatures(
            latitude=52.0,
            longitude=5.0,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
        )

        # Act
        result = transform.fit_transform(sample_dataset)

        # Assert
        assert "dni" in result.feature_names
        assert "gti" in result.feature_names

        # Verify that the correct surface parameters were passed to pvlib
        mock_gti.assert_called()
        call_kwargs = mock_gti.call_args.kwargs
        assert call_kwargs["surface_tilt"] == surface_tilt
        assert call_kwargs["surface_azimuth"] == surface_azimuth


def test_dni_fillna_behavior(basic_transform: RadiationDerivedFeatures):
    """Test that DNI calculation properly handles NaN values by filling them with 0.0."""
    # Arrange
    solar_position = pd.DataFrame({
        "apparent_zenith": [30.0, 45.0, 60.0],
    })

    clearsky_radiation = pd.DataFrame({
        "dhi": [100.0, 150.0, 200.0],
        "dni": [800.0, 600.0, 400.0],
    })

    ghi = pd.Series([500.0, 400.0, 300.0])

    with patch("pvlib.irradiance.dni") as mock_dni:
        # Mock DNI to return NaN values
        mock_dni.return_value = pd.Series([400.0, np.nan, 100.0])

        # Act
        result = basic_transform._calculate_dni(solar_position, clearsky_radiation, ghi)

        # Assert
        expected = pd.Series([400.0, 0.0, 100.0])  # NaN should be filled with 0.0
        pd.testing.assert_series_equal(result, expected)
