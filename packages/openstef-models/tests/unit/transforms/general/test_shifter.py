# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.general import Shifter
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Sample dataset on a 15-minute grid."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {
                "load": list(range(8)),
                "radiation": [200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0],
            },
            index=pd.date_range("2025-01-01", periods=8, freq="15min"),
        ),
        sample_interval=timedelta(minutes=15),
    )


@pytest.mark.parametrize(
    ("source_aggregation_period", "target_aggregation_period", "expected_radiation"),
    [
        pytest.param(
            timedelta(minutes=60),
            timedelta(minutes=15),
            [230.0, 250.0, 270.0, 290.0, 310.0, 330.0, np.nan, np.nan],
            id="60min_to_15min",
        ),
        pytest.param(
            timedelta(minutes=30),
            timedelta(minutes=15),
            [210.0, 230.0, 250.0, 270.0, 290.0, 310.0, 330.0, np.nan],
            id="30min_to_15min",
        ),
        pytest.param(
            timedelta(minutes=60),
            timedelta(0),
            [240.0, 260.0, 280.0, 300.0, 320.0, 340.0, np.nan, np.nan],
            id="60min_to_instantaneous",
        ),
    ],
)
def test_shifter__shift_and_interpolate(
    sample_dataset: TimeSeriesDataset,
    source_aggregation_period: timedelta,
    target_aggregation_period: timedelta,
    expected_radiation: list[float],
):
    """Test that features are shifted and interpolated correctly for different intervals."""
    # Arrange
    shifter = Shifter(
        selection=FeatureSelection(include={"radiation"}),
        source_aggregation_period=source_aggregation_period,
        target_aggregation_period=target_aggregation_period,
    )

    # Act
    result = shifter.transform(sample_dataset)

    # Assert
    expected = pd.Series(expected_radiation, index=sample_dataset.index, name="radiation")
    pd.testing.assert_series_equal(result.data["radiation"], expected)
    # Unselected feature should be unchanged
    assert result.data["load"].tolist() == list(range(8))


def test_shifter__no_shift_when_intervals_equal(sample_dataset: TimeSeriesDataset):
    """Test that the same dataset object is returned when no shift is needed."""
    # Arrange
    shifter = Shifter(
        selection=FeatureSelection(include={"radiation"}),
        source_aggregation_period=timedelta(minutes=15),
        target_aggregation_period=timedelta(minutes=15),
    )

    # Act
    result = shifter.transform(sample_dataset)

    # Assert
    assert result is sample_dataset


@pytest.mark.parametrize(
    ("fill_edges", "expected_trailing"),
    [
        pytest.param(False, [np.nan, np.nan], id="no_fill_leaves_nan"),
        pytest.param(True, [340.0, 340.0], id="fill_uses_last_original_value"),
    ],
)
def test_shifter__fill_edges(
    sample_dataset: TimeSeriesDataset,
    fill_edges: bool,
    expected_trailing: list[float],
):
    """Test edge handling with fill_edges."""
    # Arrange
    shifter = Shifter(
        selection=FeatureSelection(include={"radiation"}),
        fill_edges=fill_edges,
    )

    # Act
    result = shifter.transform(sample_dataset)

    # Assert — first 6 values are always the shifted+interpolated result
    assert result.data["radiation"].iloc[:6].tolist() == [230.0, 250.0, 270.0, 290.0, 310.0, 330.0]
    # Trailing 2 values depend on fill_edges
    expected = pd.Series(expected_trailing, name="radiation", index=sample_dataset.index[-2:])
    pd.testing.assert_series_equal(result.data["radiation"].iloc[-2:], expected)


def test_shifter__fill_edges_leading_nan(sample_dataset: TimeSeriesDataset):
    """Test fill_edges handles leading NaN when source interval is smaller than target."""
    # Arrange — negative shift (source < target) produces leading NaN
    shifter = Shifter(
        selection=FeatureSelection(include={"radiation"}),
        source_aggregation_period=timedelta(minutes=15),
        target_aggregation_period=timedelta(minutes=60),
        fill_edges=True,
    )

    # Act
    result = shifter.transform(sample_dataset)

    # Assert — leading NaN should be filled with the first original value (200.0)
    expected = pd.Series(
        [200.0, 200.0, 210.0, 230.0, 250.0, 270.0, 290.0, 310.0],
        index=sample_dataset.index,
        name="radiation",
    )
    pd.testing.assert_series_equal(result.data["radiation"], expected)


def test_shifter__preserves_preexisting_nan():
    """Test that pre-existing NaN values are shifted rather than imputed."""
    # Arrange
    data = pd.DataFrame(
        {
            "load": list(range(8)),
            "radiation": [200.0, 220.0, np.nan, 260.0, 280.0, np.nan, np.nan, 340.0],
        },
        index=pd.date_range("2025-01-01", periods=8, freq="15min"),
    )
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    shifter = Shifter(
        selection=FeatureSelection(include={"radiation"}),
        source_aggregation_period=timedelta(minutes=60),
        target_aggregation_period=timedelta(minutes=15),
        fill_edges=True,
    )

    # Act
    result = shifter.transform(dataset)

    # Assert
    # Each grid point inherits the NaN status of its nearest shifted data point.
    # For equidistant cases, pandas nearest picks the later (right) neighbor.
    expected = pd.Series(
        [np.nan, 250.0, 270.0, np.nan, np.nan, 330.0, 340.0, 340.0],
        index=dataset.index,
        name="radiation",
    )
    pd.testing.assert_series_equal(result.data["radiation"], expected)
