# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_beam.datasets.base import TimeseriesDataset
from openstef_beam.datasets.pandas_dataset import PandasTimeseriesDataset
from openstef_beam.datasets.transforms import (
    ConcatFeaturewise,
    HorizonTransform,
    ResampleTransform,
    TimeseriesDatasetTransforms,
)


@pytest.fixture
def simple_dataset() -> PandasTimeseriesDataset:
    return PandasTimeseriesDataset(
        data=pd.DataFrame({
            "timestamp": [
                datetime.fromisoformat("2024-01-01T10:00:00"),
                datetime.fromisoformat("2024-01-01T11:00:00"),
                datetime.fromisoformat("2024-01-01T12:00:00"),
            ],
            "available_at": [
                datetime.fromisoformat("2024-01-01T10:05:00"),
                datetime.fromisoformat("2024-01-01T11:05:00"),
                datetime.fromisoformat("2024-01-01T12:05:00"),
            ],
            "value": [10.0, 20.0, 30.0],
        }),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def partial_dataset() -> PandasTimeseriesDataset:
    """Create a dataset with some missing timestamps"""
    data = pd.DataFrame({
        "timestamp": [
            datetime.fromisoformat("2024-01-01T10:00:00"),
            # Gap at 11:00
            datetime.fromisoformat("2024-01-01T12:00:00"),
        ],
        "available_at": [
            datetime.fromisoformat("2024-01-01T10:05:00"),
            datetime.fromisoformat("2024-01-01T12:05:00"),
        ],
        "third_value": [1000.0, 3000.0],
    })

    return PandasTimeseriesDataset(data=data, sample_interval=timedelta(hours=1))


def test_horizon_transform_basic(simple_dataset: PandasTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")

    expected = pd.DataFrame(
        {"value": [10.0, 20.0, np.nan]},
        index=pd.DatetimeIndex(["2024-01-01T10:00:00", "2024-01-01T11:00:00", "2024-01-01T12:00:00"]),
    )

    # Act
    transform = HorizonTransform(simple_dataset, horizon)
    window = transform.get_window(start, end)

    # Assert
    assert transform.feature_names == simple_dataset.feature_names
    assert transform.sample_interval == simple_dataset.sample_interval
    pd.testing.assert_frame_equal(window, expected, check_freq=False)


def test_horizon_transform_validation(simple_dataset: PandasTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")
    invalid_available = datetime.fromisoformat("2024-01-01T13:00:00")  # After horizon

    # Act & Assert
    transform = HorizonTransform(simple_dataset, horizon)
    with pytest.raises(ValueError, match=r"Available before .* is greater than the horizon"):
        transform.get_window(start, end, invalid_available)


# ConcatFeaturewise Tests
@pytest.mark.parametrize(
    ("mode", "expected_length"),
    [
        pytest.param("left", 3, id="left_join"),
        pytest.param("inner", 3, id="inner_join"),
        pytest.param("outer", 3, id="outer_join"),
    ],
)
def test_concat_featurewise_basic(simple_dataset: PandasTimeseriesDataset, mode: str, expected_length: int):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")
    second_dataset = PandasTimeseriesDataset(
        data=simple_dataset.data.rename(columns={"value": "other_value"}),
        sample_interval=simple_dataset.sample_interval,
    )

    # Act
    transform = ConcatFeaturewise([simple_dataset, second_dataset], mode=mode)
    window = transform.get_window(start, end)

    # Assert
    assert len(window) == expected_length
    assert set(transform.feature_names) == {"value", "other_value"}
    assert transform.sample_interval == simple_dataset.sample_interval
    assert all(col in window.columns for col in ["value", "other_value"])


def test_concat_featurewise_with_partial(
    simple_dataset: PandasTimeseriesDataset, partial_dataset: PandasTimeseriesDataset
):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")

    # Act
    transform = ConcatFeaturewise([simple_dataset, partial_dataset], mode="outer")
    window = transform.get_window(start, end)

    # Assert
    assert len(window) == 3  # Should include all timestamps from both datasets
    assert set(transform.feature_names) == {"value", "third_value"}
    assert pd.isna(window.loc[pd.Timestamp.fromisoformat("2024-01-01T11:00:00"), "third_value"])
    assert window.loc[pd.Timestamp.fromisoformat("2024-01-01T11:00:00"), "value"] == 20.0


def test_concat_featurewise_validation_error():
    # Arrange
    invalid_datasets: list[TimeseriesDataset] = []

    # Act & Assert
    with pytest.raises(ValueError, match="At least two datasets are required"):
        ConcatFeaturewise(invalid_datasets, mode="left")


def test_concat_featurewise_overlapping_features(simple_dataset: PandasTimeseriesDataset):
    # Arrange - both datasets have "value" column

    # Act & Assert
    with pytest.raises(ValueError, match=r"Datasets have overlapping feature names: value"):
        ConcatFeaturewise([simple_dataset, simple_dataset], mode="left")


# ResampleTransform Tests
def test_resample_transform_upsampling(simple_dataset: PandasTimeseriesDataset):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T12:30:00")
    new_interval = timedelta(minutes=30)

    expected = pd.DataFrame(
        {"value": [10.0, 15.0, 20.0, 25.0, 30.0]},
        index=pd.DatetimeIndex([
            "2024-01-01T10:00:00",
            "2024-01-01T10:30:00",
            "2024-01-01T11:00:00",
            "2024-01-01T11:30:00",
            "2024-01-01T12:00:00",
        ]),
    )

    # Act
    transform = ResampleTransform(simple_dataset, new_interval)
    window = transform.get_window(start, end)

    # Assert
    assert transform.sample_interval == new_interval
    pd.testing.assert_frame_equal(window, expected, check_freq=False)


def test_resample_transform_downsampling(simple_dataset: PandasTimeseriesDataset):
    # Arrange
    start = datetime.fromisoformat("2024-01-01T10:00:00")
    end = datetime.fromisoformat("2024-01-01T13:00:00")
    new_interval = timedelta(hours=2)

    # Act
    transform = ResampleTransform(simple_dataset, new_interval)
    window = transform.get_window(start, end)
    expected = pd.DataFrame(
        {"value": [10.0, 30.0]}, index=pd.DatetimeIndex(["2024-01-01T10:00:00", "2024-01-01T12:00:00"])
    )

    # Assert
    assert transform.sample_interval == new_interval
    pd.testing.assert_frame_equal(window, expected, check_freq=False)


# DatasetTransforms Factory Tests
def test_dataset_transforms_factory(simple_dataset: PandasTimeseriesDataset):
    # Arrange
    horizon = datetime.fromisoformat("2024-01-01T12:00:00")
    new_interval = timedelta(minutes=30)
    second_dataset = PandasTimeseriesDataset(
        data=simple_dataset.data.rename(columns={"value": "other_value"}),
        sample_interval=simple_dataset.sample_interval,
    )

    # Act
    horizon_transform = TimeseriesDatasetTransforms.horizon(simple_dataset, horizon)
    concat_transform = TimeseriesDatasetTransforms.concat_featurewise([simple_dataset, second_dataset], mode="left")
    resample_transform = TimeseriesDatasetTransforms.resample(simple_dataset, new_interval)

    # Assert
    assert isinstance(horizon_transform, HorizonTransform)
    assert isinstance(concat_transform, ConcatFeaturewise)
    assert isinstance(resample_transform, ResampleTransform)
    assert horizon_transform.horizon == horizon
    assert resample_transform.sample_interval == new_interval
