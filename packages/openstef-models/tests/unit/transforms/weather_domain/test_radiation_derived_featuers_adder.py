# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the RadiationDerivedFeaturesAdder."""

from datetime import timedelta
from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError, TimeSeriesValidationError
from openstef_core.testing import assert_timeseries_equal
from openstef_models.transforms.weather_domain import RadiationDerivedFeaturesAdder

pvlib = pytest.importorskip("pvlib")


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Create a sample TimeSeriesDataset with radiation data for testing."""
    data = pd.DataFrame(
        {"radiation": [3600000, 7200000, 5400000, 1800000, 0]},  # J/m² values
        index=pd.date_range("2025-06-01 08:00", periods=5, freq="h", tz="Europe/Amsterdam"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def sample_dataset_no_tz() -> TimeSeriesDataset:
    """Create a sample TimeSeriesDataset without timezone for testing error cases."""
    data = pd.DataFrame(
        {"radiation": [3600000, 7200000, 5400000]}, index=pd.date_range("2025-06-01", periods=3, freq="h")
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def sample_dataset_no_radiation() -> TimeSeriesDataset:
    """Create a sample TimeSeriesDataset without radiation feature for testing error cases."""
    data = pd.DataFrame(
        {"load": [100.0, 110.0, 120.0]},
        index=pd.date_range("2025-06-01", periods=3, freq="h", tz="Europe/Amsterdam"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def basic_transform() -> RadiationDerivedFeaturesAdder:
    """Create a basic RadiationDerivedFeaturesAdder instance."""
    return RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.3676), longitude=Longitude(4.9041)),
        included_features=["dni", "gti"],
        surface_tilt=34.0,
        surface_azimuth=180.0,
    )


def test_transform_raises_error_for_non_timezone_aware_dataset(
    sample_dataset_no_tz: TimeSeriesDataset, basic_transform: RadiationDerivedFeaturesAdder
):
    """Test that transform raises TimeSeriesValidationError for non-timezone-aware dataset."""
    # Act & Assert
    with pytest.raises(TimeSeriesValidationError, match="The datetime index must be timezone-aware"):
        basic_transform.transform(sample_dataset_no_tz)


def test_transform_missing_column(
    sample_dataset_no_radiation: TimeSeriesDataset, basic_transform: RadiationDerivedFeaturesAdder
):
    """Test that transform raises MissingColumnsError when radiation column is missing."""
    # Act
    result = basic_transform.transform(sample_dataset_no_radiation)

    # Assert
    assert_timeseries_equal(actual=result, expected=sample_dataset_no_radiation)


def test_transform_raises_error_when_pvlib_not_available(sample_dataset: TimeSeriesDataset):
    """Test that transform raises MissingExtraError when pvlib is not available."""
    # Arrange
    transform = RadiationDerivedFeaturesAdder(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))

    # Act & Assert
    with (
        patch.dict("sys.modules", {"pvlib": None}),
        pytest.raises(MissingExtraError, match="pvlib"),
    ):
        transform.transform(sample_dataset)


@pytest.mark.parametrize(
    ("included_features", "expected_additional_features"),
    [
        pytest.param(["dni"], ["dni"], id="dni_only"),
        pytest.param(["gti"], ["gti"], id="gti_only"),
        pytest.param(["dni", "gti"], ["dni", "gti"], id="both_features"),
    ],
)
def test_transform_with_different_feature_combinations(
    sample_dataset: TimeSeriesDataset,
    included_features: list[Literal["dni", "gti"]],
    expected_additional_features: list[str],
):
    """Test transform with different combinations of included features."""
    # Arrange
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        included_features=included_features,
    )

    # Act
    result = transform.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    # Check that all original features are preserved
    assert "radiation" in result.feature_names
    # Check that expected features are added
    for feature in expected_additional_features:
        assert feature in result.feature_names
    # Check total number of features
    expected_total = len(sample_dataset.feature_names) + len(expected_additional_features)
    assert len(result.feature_names) == expected_total


def test_transform_preserves_original_data_and_metadata(
    sample_dataset: TimeSeriesDataset,
    basic_transform: RadiationDerivedFeaturesAdder,
):
    """Test that transform preserves original data and metadata."""
    # Act
    result = basic_transform.transform(sample_dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == sample_dataset.sample_interval
    assert len(result.data) == len(sample_dataset.data)

    # Check that original features are preserved unchanged
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])


def test_transform_radiation_unit_conversion():
    """Test that radiation is correctly converted from J/m² to kWh/m² for pvlib."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [3600000, 7200000]},  # 1000 and 2000 kWh/m² when divided by 3.6e6
        index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz="Europe/Amsterdam"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = RadiationDerivedFeaturesAdder(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))

    # Act
    result = transform.transform(dataset)

    # Assert
    # The exact values depend on solar calculations, but we can verify the result is reasonable
    assert "dni" in result.data.columns
    assert "gti" in result.data.columns
    assert (result.data["dni"] >= 0).all()
    assert (result.data["gti"] >= 0).all()


def test_transform_with_empty_dataset():
    """Test handling of empty dataset."""
    # Arrange
    data = pd.DataFrame(columns=["radiation"]).astype(float)
    data.index = pd.DatetimeIndex([], tz="Europe/Amsterdam")
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = RadiationDerivedFeaturesAdder(coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)))

    # Act
    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == 0
    assert "radiation" in result.feature_names
    assert "dni" in result.feature_names
    assert "gti" in result.feature_names


@pytest.mark.parametrize(
    ("surface_tilt", "surface_azimuth"),
    [
        pytest.param(0.0, 180.0, id="horizontal_surface"),
        pytest.param(90.0, 180.0, id="vertical_south_facing"),
        pytest.param(34.0, 90.0, id="east_facing"),
        pytest.param(34.0, 270.0, id="west_facing"),
    ],
)
def test_transform_with_different_surface_orientations(
    sample_dataset: TimeSeriesDataset,
    surface_tilt: float,
    surface_azimuth: float,
):
    """Test RadiationDerivedFeaturesAdder with different surface orientations."""
    # Arrange
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )

    # Act
    result = transform.transform(sample_dataset)

    # Assert
    assert "dni" in result.feature_names
    assert "gti" in result.feature_names
    assert (result.data["dni"] >= 0).all()
    assert (result.data["gti"] >= 0).all()


def test_transform_custom_radiation_column():
    """Test transform with custom radiation column name."""
    # Arrange
    data = pd.DataFrame(
        {"solar_irradiance": [3600000, 7200000]},
        index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz="Europe/Amsterdam"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)), radiation_column="solar_irradiance"
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    assert "solar_irradiance" in result.feature_names
    assert "dni" in result.feature_names
    assert "gti" in result.feature_names


def test_transform_handles_missing_columns():
    # Arrange
    dataset = TimeSeriesDataset(
        data=pd.DataFrame(
            {"radiation": [3600000, 7200000]},
            index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz="Europe/Amsterdam"),
        ),
        sample_interval=timedelta(hours=1),
    )
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)), radiation_column="solar_irradiance"
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    assert_timeseries_equal(actual=result, expected=dataset)


@pytest.mark.parametrize(
    ("latitude", "longitude", "timezone"),
    [
        pytest.param(52.0, 5.0, "Europe/Amsterdam", id="netherlands"),
        pytest.param(40.7, -74.0, "US/Eastern", id="new_york"),
        pytest.param(-33.9, 18.4, "Africa/Johannesburg", id="cape_town"),
    ],
)
def test_pvlib_integration_different_locations(latitude: float, longitude: float, timezone: str):
    """Test RadiationDerivedFeaturesAdder with real pvlib calls across different locations."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [7200000, 5400000]},  # J/m² values
        index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz=timezone),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(latitude), longitude=Longitude(longitude))
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert "dni" in result.data.columns
    assert "gti" in result.data.columns
    assert len(result.data) == len(dataset.data)

    # Check that derived values are realistic (should be positive or zero)
    assert (result.data["dni"] >= 0).all()
    assert (result.data["gti"] >= 0).all()

    # Verify that the timezone is properly handled
    assert hasattr(result.data.index, "tz")
    assert result.data.index.tz is not None  # type: ignore[attr-defined]
    assert str(result.data.index.tz) == timezone  # type: ignore[attr-defined]


def test_pvlib_integration_summer_midday():
    """Test that solar calculations produce reasonable results during summer midday."""
    # Arrange - Use summer midday for better solar radiation
    data = pd.DataFrame(
        {"radiation": [7200000, 10800000, 14400000]},  # High radiation values for summer
        index=pd.date_range("2025-06-21 11:00", periods=3, freq="1h", tz="Europe/Amsterdam"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        surface_tilt=30.0,
        surface_azimuth=180.0,
    )

    # Act
    result = transform.transform(dataset)

    # Assert
    dni_values = result.data["dni"]
    gti_values = result.data["gti"]

    # Check that we have realistic solar radiation values
    assert len(dni_values) == 3
    assert len(gti_values) == 3

    # During summer midday in Netherlands, we should get some solar radiation
    assert (dni_values >= 0).all()
    assert (gti_values >= 0).all()


def test_pvlib_integration_surface_orientations():
    """Test different surface orientations with real pvlib calculations."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [7200000, 7200000]},
        index=pd.date_range("2025-06-01 12:00", periods=2, freq="1h", tz="Europe/Amsterdam"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    # Test different orientations
    south_facing = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        surface_tilt=30.0,
        surface_azimuth=180.0,
    )
    east_facing = RadiationDerivedFeaturesAdder(
        coordinate=Coordinate(latitude=Latitude(52.0), longitude=Longitude(5.0)),
        surface_tilt=30.0,
        surface_azimuth=90.0,
    )

    # Act
    south_result = south_facing.transform(dataset)
    east_result = east_facing.transform(dataset)

    # Assert
    assert "gti" in south_result.data.columns
    assert "gti" in east_result.data.columns

    # Values should be non-negative
    assert (south_result.data["gti"] >= 0).all()
    assert (east_result.data["gti"] >= 0).all()

    # Both should produce valid results (exact values depend on sun position)
    assert len(south_result.data["gti"]) == 2
    assert len(east_result.data["gti"]) == 2
