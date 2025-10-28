# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import override

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.mixins import Transform, TransformPipeline


class SimpleAddTransform(Transform[TimeSeriesDataset, TimeSeriesDataset]):
    """Simple transform that adds a constant to all time series values."""

    def __init__(self, add_value: float = 1.0):
        self.add_value = add_value

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Stateless transform, always considered fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            data=data.data + self.add_value,
            sample_interval=data.sample_interval,
        )


class SimpleMultiplyTransform(Transform[TimeSeriesDataset, TimeSeriesDataset]):
    """Simple transform that multiplies all time series values by a constant."""

    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Stateless transform, always considered fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            data=data.data * self.multiplier,
            sample_interval=data.sample_interval,
        )


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create a sample TimeSeriesDataset for testing."""
    data = pd.DataFrame(
        {"load": [100.0, 105.0]},
        index=pd.date_range("2025-01-01 10:00", periods=2, freq="h"),
    )

    return TimeSeriesDataset(
        data=data,
        sample_interval=timedelta(hours=1),
    )


@pytest.mark.parametrize(
    ("transforms", "expected_values"),
    [
        pytest.param([], [100.0, 105.0], id="no_transforms"),
        pytest.param([SimpleAddTransform(5.0)], [105.0, 110.0], id="single_transform"),
        pytest.param([SimpleAddTransform(10.0), SimpleMultiplyTransform(2.0)], [220.0, 230.0], id="chained_transforms"),
    ],
)
def test_transform_pipeline__fit_transform_functionality(
    sample_timeseries_dataset: TimeSeriesDataset,
    transforms: list[Transform[TimeSeriesDataset, TimeSeriesDataset]],
    expected_values: list[float],
):
    """Test TransformPipeline fit_transform with essential scenarios."""
    # Arrange
    pipeline = TransformPipeline[TimeSeriesDataset](transforms=transforms)

    # Act - fit is required before transform
    result = pipeline.fit_transform(sample_timeseries_dataset)

    # Assert
    assert result.data["load"].tolist() == expected_values
    assert result.sample_interval == sample_timeseries_dataset.sample_interval
