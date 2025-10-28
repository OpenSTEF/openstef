# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pickle  # noqa: S403 - controlled test
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


def test_transform_pipeline__pickle_roundtrip(
    sample_timeseries_dataset: TimeSeriesDataset,
):
    """Test that TransformPipeline with generic types can be pickled and unpickled.

    This verifies that the __reduce__ implementation correctly handles the
    pickling of generic TransformPipeline[T] instances, which would otherwise
    fail with: "Can't pickle <class 'openstef_core.mixins.transform.TransformPipeline[T]'>".
    """
    # Arrange - create a pipeline with generic type parameter and multiple transforms
    pipeline = TransformPipeline[TimeSeriesDataset](
        transforms=[
            SimpleAddTransform(add_value=10.0),
            SimpleMultiplyTransform(multiplier=2.0),
        ]
    )

    # Fit the pipeline
    pipeline.fit(sample_timeseries_dataset)

    # Get expected result before pickling
    expected_result = pipeline.transform(sample_timeseries_dataset)

    # Act - pickle and unpickle
    pickled = pickle.dumps(pipeline)
    restored_pipeline = pickle.loads(pickled)  # noqa: S301 - Controlled test

    # Assert - verify the restored pipeline works correctly
    assert restored_pipeline.is_fitted
    actual_result = restored_pipeline.transform(sample_timeseries_dataset)
    assert actual_result.data["load"].tolist() == expected_result.data["load"].tolist()
    assert actual_result.sample_interval == expected_result.sample_interval
