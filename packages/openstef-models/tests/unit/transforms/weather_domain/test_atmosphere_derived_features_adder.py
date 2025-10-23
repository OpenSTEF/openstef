# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.weather_domain.atmosphere_derived_features_adder import (
    AtmosphereDerivedFeaturesAdder,
)


@pytest.fixture
def sample_data() -> TimeSeriesDataset:
    data = pd.DataFrame(
        {
            "temperature": [10, 15, 20, 25, 30],  # Celsius
            "pressure": [1013, 1010, 1007, 1005, 1002],  # hPa
            "relative_humidity": [50, 60, 70, 80, 90],  # %
        },
        index=pd.date_range("2025-01-01", periods=5, freq="h"),
    )
    return TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))


def test_transform_adds_all_features(sample_data: TimeSeriesDataset):
    transform = AtmosphereDerivedFeaturesAdder()
    result = transform.transform(sample_data)
    for feature in ["saturation_vapour_pressure", "vapour_pressure", "dewpoint", "air_density"]:
        assert feature in result.data.columns
        assert not result.data[feature].isna().any()


def test_transform_included_features_subset(sample_data: TimeSeriesDataset):
    transform = AtmosphereDerivedFeaturesAdder(included_features=["dewpoint", "air_density"])
    result = transform.transform(sample_data)
    assert "dewpoint" in result.data.columns
    assert "air_density" in result.data.columns
    assert "saturation_vapour_pressure" not in result.data.columns
    assert "vapour_pressure" not in result.data.columns


def test_transform_missing_columns():
    data = pd.DataFrame(
        {
            "temperature": [10, 15, 20, 25, 30],  # Celsius
            "relative_humidity": [50, 60, 70, 80, 90],  # %
        },
        index=pd.date_range("2025-01-01", periods=5, freq="h"),
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(hours=1))
    transform = AtmosphereDerivedFeaturesAdder()

    # Act
    result = transform.transform(dataset)

    # Assert
    assert "saturation_vapour_pressure" in result.feature_names
    assert "vapour_pressure" in result.feature_names
    assert "dewpoint" in result.feature_names


def test_saturation_vapour_pressure_calculation():
    temperature = pd.Series([-10, 20, 30])
    result = AtmosphereDerivedFeaturesAdder._calculate_saturation_vapour_pressure(temperature)
    assert isinstance(result, pd.Series)
    assert np.all(result > 0)
    # Saturation vapour pressure should increase with temperature
    assert np.all(result.astype(dtype=float).diff().dropna() > 0)
    expected_values = pd.Series([286.56, 2338.34, 4245.126])
    pd.testing.assert_series_equal(result, expected_values, atol=0.001)


def test_vapour_pressure_calculation():
    transform = AtmosphereDerivedFeaturesAdder()
    temperature = pd.Series([-10, 10, 20])
    relative_humidity = pd.Series([50, 50, 80])
    result = transform._calculate_vapour_pressure(temperature, relative_humidity)
    assert isinstance(result, pd.Series)
    assert np.all(result > 0)
    # Vapour pressure should increase with temperature
    assert np.all(result.astype(dtype=float).diff().dropna() > 0)
    expected_values = pd.Series([14328.017, 61393.008, 187067.198])
    pd.testing.assert_series_equal(result, expected_values, atol=0.001)


def test_dewpoint_calculation():
    temperature = pd.Series([-10, 10, 20])
    relative_humidity = pd.Series([50, 50, 80])
    result = AtmosphereDerivedFeaturesAdder._calculate_dewpoint(temperature, relative_humidity)
    assert isinstance(result, pd.Series)
    # Dewpoint should not exceed temperature
    assert np.all(result < temperature + 1)
    expected_values = pd.Series([-18.468, 0.047, 16.444])
    pd.testing.assert_series_equal(result, expected_values, atol=0.001)


def test_air_density_calculation():
    transform = AtmosphereDerivedFeaturesAdder()
    temperature = pd.Series([20, 10, -10])
    relative_humidity = pd.Series([80, 50, 50])
    pressure = pd.Series([1000, 1000, 1020])
    result = transform._calculate_air_density(temperature, relative_humidity, pressure)
    assert isinstance(result, pd.Series)
    assert np.all(result > 0)
    # Air density increases as temperature decreases, relative humidity decreases and pressure increases
    assert np.all(result.astype(dtype=float).diff().dropna() > 0)
    expected_values = pd.Series([0.3480, 0.9448, 1.278])
    pd.testing.assert_series_equal(result, expected_values, atol=0.001)
