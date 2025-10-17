# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import Clipper
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.fixture
def train_dataset() -> TimeSeriesDataset:
    """Training dataset with three features A, B, C."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def test_dataset() -> TimeSeriesDataset:
    """Test dataset with values outside training ranges."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [0.5, 4.0], "B": [5.0, 35.0], "C": [150.0, 350.0]},
            index=pd.date_range("2025-01-06", periods=2, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.mark.parametrize(
    ("mode", "expected_a", "expected_b"),
    [
        pytest.param("minmax", [1.0, 3.0], [10.0, 30.0], id="minmax"),
        pytest.param("standard", [0.5, 4.0], [5.0, 35.0], id="standard"),
    ],
)
def test_clipper__fit_transform(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
    expected_a: list[float],
    expected_b: list[float],
):
    """Test fit and transform clips correctly for both modes and leaves other columns unchanged."""
    # Arrange
    clipper = Clipper(selection=FeatureSelection(include={"A", "B"}), mode=mode, n_std=2.0)  # type: ignore[arg-type]

    # Act
    clipper.fit(train_dataset)
    transformed_dataset = clipper.transform(test_dataset)

    # Assert
    # Columns A and B should be clipped according to mode
    # Column C should remain unchanged
    expected_df = pd.DataFrame(
        {
            "A": expected_a,
            "B": expected_b,
            "C": [150.0, 350.0],  # Unchanged
        },
        index=test_dataset.index,
    )
    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)
    assert transformed_dataset.sample_interval == test_dataset.sample_interval


def test_clipper__handles_missing_features():
    """Test clipper handles missing features gracefully in both training and test data."""
    # Arrange
    train_data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    # Test data only has feature A and C, missing B
    test_data = pd.DataFrame(
        {"A": [0.5, 4.0], "C": [150.0, 350.0]},
        index=pd.date_range("2025-01-06", periods=2, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    # Clipper configured for A and B, but B is missing in test data
    clipper = Clipper(selection=FeatureSelection(include={"A", "B"}), mode="minmax")

    # Act
    clipper.fit(train_dataset)
    transformed_dataset = clipper.transform(test_dataset)

    # Assert
    # A should be clipped, B is missing in test data so not clipped, C is not selected so unchanged
    expected_df = pd.DataFrame(
        {"A": [1.0, 3.0], "C": [150.0, 350.0]},  # A clipped, C unchanged
        index=test_dataset.index,
    )
    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)


def test_clipper__transform_without_fit(test_dataset: TimeSeriesDataset):
    """Test that transform raises error when called without fitting."""
    # Arrange
    clipper = Clipper(selection=FeatureSelection(include={"A", "B"}))

    # Act & Assert
    with pytest.raises(NotFittedError, match=r"Clipper.*has not been fitted"):
        clipper.transform(test_dataset)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("minmax", id="minmax"),
        pytest.param("standard", id="standard"),
    ],
)
def test_clipper__state_roundtrip(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
):
    """Test clipper state serialization and restoration for both modes."""
    # Arrange
    original_transform = Clipper(selection=FeatureSelection(include={"A", "B"}), mode=mode, n_std=3.0)  # type: ignore[arg-type]
    original_transform.fit(train_dataset)

    # Act
    state = original_transform.to_state()
    restored_transform = Clipper(selection=FeatureSelection(include={"A", "B"}))
    restored_transform = restored_transform.from_state(state)

    original_result = original_transform.transform(test_dataset)
    restored_result = restored_transform.transform(test_dataset)

    # Assert
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
    assert restored_transform.mode == mode
    assert restored_transform.n_std == 3.0  # Non-default value should be restored
