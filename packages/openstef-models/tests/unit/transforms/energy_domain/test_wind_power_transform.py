# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.energy_domain.wind_power_transform import WindPowerTransform


@pytest.mark.parametrize(
    ("wind_speed", "reference_height", "hub_height", "expected"),
    [
        pytest.param(10.0, 10.0, 100.0, 13.90, id="standard_case"),
        pytest.param(5.0, 10.0, 100.0, 6.95, id="lower_wind_speed"),
        pytest.param(10.0, 50.0, 100.0, 11.042, id="different_reference_height"),
        pytest.param(5.0, 50.0, 50.0, 5.0, id="same_height"),
    ],
)
def test_calculate_wind_speed_at_hub_height(
    wind_speed: float, reference_height: float, hub_height: float, expected: float
) -> None:
    """Test wind speed calculation at hub height using power law."""
    # Arrange
    transform = WindPowerTransform(reference_height=reference_height, hub_height=hub_height)
    wind_speed_series = pd.Series([wind_speed], index=[datetime.fromisoformat("2025-01-01T00:00:00")])

    # Act
    result = transform._calculate_wind_speed_at_hub_height(wind_speed_series)

    # Assert
    assert abs(result.iloc[0] - expected) < 1e-3
    assert len(result) == 1
    assert result.index.equals(wind_speed_series.index)


@pytest.mark.parametrize(
    ("wind_speed_hub", "rated_power", "steepness", "slope_center", "expected"),
    [
        pytest.param(0.0, 1.0, 0.664, 8.07, 0.00, id="zero_wind_speed"),
        pytest.param(8.07, 1.0, 0.664, 8.07, 0.50, id="slope_center_wind_speed"),
        pytest.param(20.0, 1.0, 0.664, 8.07, 1.0, id="high_wind_speed"),
        pytest.param(8.07, 2.0, 0.664, 8.07, 1.0, id="double_rated_power"),
        pytest.param(8.07, 1.0, 1.5, 8.07, 0.50, id="steeper_sigmoid"),
        pytest.param(8.07, 1.0, 0.2, 8.07, 0.50, id="flatter_sigmoid"),
        pytest.param(10.0, 1.0, 0.664, 10.0, 0.50, id="custom_slope_center"),
    ],
)
def test_calculate_wind_power(
    wind_speed_hub: float, rated_power: float, steepness: float, slope_center: float, expected: float
) -> None:
    """Test wind power calculation using sigmoid power curve."""
    # Arrange
    transform = WindPowerTransform(rated_power=rated_power, steepness=steepness, slope_center=slope_center)
    wind_speed_series = pd.Series([wind_speed_hub], index=[datetime.fromisoformat("2025-01-01T00:00:00")])

    # Act
    result = transform._calculate_wind_power(wind_speed_series)

    # Assert
    assert abs(result.iloc[0] - expected) < 1e-2
    assert len(result) == 1
    assert result.index.equals(wind_speed_series.index)


def test_transform_preserves_original_data() -> None:
    """Test that transform preserves all original data columns."""
    # Arrange
    data = pd.DataFrame(
        {"windspeed": [8.0, 12.0], "windspeed_100m": [10.0, 15.0], "load": [150, 250], "temperature": [20.0, 22.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    transform = WindPowerTransform(windspeed_hub_height_column="windspeed_100m")
    dataset = TimeSeriesDataset(data=data, sample_interval="1h")

    # Act
    result = transform.transform(dataset)

    # Assert
    # All original columns should be preserved
    for col in data.columns:
        assert col in result.data.columns
        pd.testing.assert_series_equal(result.data[col], data[col])

    # New wind_power column should be added
    assert "wind_power" in result.data.columns
    assert len(result.data.columns) == len(data.columns) + 1


def test_wind_power_sigmoid_properties() -> None:
    """Test that wind power calculation has expected sigmoid properties."""
    # Arrange
    transform = WindPowerTransform()
    wind_speeds = np.linspace(0, 20, 100)
    index = pd.date_range("2025-01-01", periods=100, freq="h")
    wind_speed_series = pd.Series(wind_speeds, index=index)

    # Act
    wind_power = transform._calculate_wind_power(wind_speed_series)

    # Assert
    # Power should be monotonically increasing
    assert all(wind_power.iloc[i] <= wind_power.iloc[i + 1] for i in range(len(wind_power) - 1))

    # Power should approach 0 for very low wind speeds
    assert wind_power.iloc[0] < 0.1

    # Power should approach rated power for very high wind speeds
    assert wind_power.iloc[-1] > 0.9 * transform.rated_power

    # Power at slope center should be approximately half of rated power
    center_idx = np.argmin(np.abs(wind_speeds - transform.slope_center))
    assert abs(wind_power.iloc[center_idx] - transform.rated_power / 2) < 0.1
