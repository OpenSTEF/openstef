# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.multi_horizon_adapter import combine_horizon_forecasts


def _create_test_forecast_dataset(
    start_time: datetime,
    num_points: int,
    sample_interval: timedelta,
    forecast_start: datetime | None = None,
    quantile_value: float = 0.5,
    constant_value: float = 100.0,
) -> ForecastDataset:
    return ForecastDataset(
        data=pd.DataFrame(
            data={Quantile(quantile_value).format(): [constant_value] * num_points},
            index=pd.date_range(start_time, periods=num_points, freq=sample_interval),
        ),
        sample_interval=sample_interval,
        forecast_start=forecast_start or start_time,
    )


@pytest.mark.parametrize(
    ("horizon_configs", "expected_values"),
    [
        pytest.param([(timedelta(hours=6), 100.0)], [100.0] * 6, id="single_forecast"),
        pytest.param(
            [(timedelta(hours=6), 100.0), (timedelta(hours=12), 200.0)], [100.0] * 6 + [200.0] * 6, id="two_horizons"
        ),
        pytest.param(
            [(timedelta(hours=3), 300.0), (timedelta(hours=6), 400.0), (timedelta(hours=9), 500.0)],
            [300.0] * 3 + [400.0] * 3 + [500.0] * 3,
            id="three_progressive_horizons",
        ),
        pytest.param(
            [(timedelta(days=1), 150.0), (timedelta(days=2), 250.0)], [150.0] * 24 + [250.0] * 24, id="day_horizons"
        ),
    ],
)
def test_combine_multiple_forecasts(horizon_configs: list[tuple[timedelta, float]], expected_values: list[float]):
    """Test combining forecasts with predictable constant values per horizon."""
    # Arrange
    forecast_start = datetime.fromisoformat("2025-01-01T10:00:00")
    sample_interval = timedelta(hours=1)
    total_points = len(expected_values)
    forecasts: dict[LeadTime, ForecastDataset] = {
        LeadTime(lead_time_delta): _create_test_forecast_dataset(
            start_time=forecast_start,
            num_points=total_points,
            sample_interval=sample_interval,
            forecast_start=forecast_start,
            constant_value=constant_value,
        )
        for lead_time_delta, constant_value in horizon_configs
    }

    # Act
    result = combine_horizon_forecasts(forecasts)

    # Assert
    assert len(result.data) == total_points
    assert result.sample_interval == sample_interval
    assert result.forecast_start == forecast_start

    # Check that the combined data matches expected values
    actual_values = result.data.iloc[:, 0].tolist()
    assert actual_values == expected_values


def test_combine_empty_forecasts_raises_error():
    """Test combining empty forecasts raises ValueError."""
    # Arrange
    forecasts: dict[LeadTime, ForecastDataset] = {}

    # Act & Assert
    with pytest.raises(ValueError, match="No forecasts to combine"):
        combine_horizon_forecasts(forecasts)
