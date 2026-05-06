# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import OUTLIER_NAN_MASK_PREFIX, OutlierHandler
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
        pytest.param("minmax", [1.0, 3.0], [10.0, 30.0], id="minmax_clip"),
        pytest.param("standard", [0.5, 4.0], [5.0, 35.0], id="standard_clip"),
    ],
)
def test_outlier_handler__clip_behavior(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
    expected_a: list[float],
    expected_b: list[float],
):
    # Arrange
    # Initialize handler with clipping behavior
    outlier_handler = OutlierHandler(
        selection=FeatureSelection(include={"A", "B"}),
        mode=mode,
        n_std=2.0,
        outlier_action="clip",
    )

    # Act
    # Fit on training data and transform test data
    outlier_handler.fit(train_dataset)
    transformed_dataset = outlier_handler.transform(test_dataset)

    # Assert
    # Selected features are clipped, others remain unchanged
    expected_df = pd.DataFrame(
        {
            "A": expected_a,
            "B": expected_b,
            "C": [150.0, 350.0],
        },
        index=test_dataset.index,
    )

    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)


@pytest.mark.parametrize(
    ("mode", "expected_a", "expected_b"),
    [
        pytest.param("minmax", [np.nan, np.nan], [np.nan, np.nan], id="minmax_nan"),
        pytest.param("standard", [0.5, 4.0], [5.0, 35.0], id="standard_nan_no_clip"),
    ],
)
def test_outlier_handler__nan_behavior(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
    expected_a: list[float],
    expected_b: list[float],
):

    # Arrange
    # Initialize handler with NaN outlier handling
    outlier_handler = OutlierHandler(
        selection=FeatureSelection(include={"A", "B"}),
        mode=mode,
        n_std=2.0,
        outlier_action="nan",
    )

    # Act
    outlier_handler.fit(train_dataset)
    transformed_dataset = outlier_handler.transform(test_dataset)

    # Assert
    # Out-of-range values are replaced with NaN where applicable
    expected_df = pd.DataFrame(
        {
            "A": expected_a,
            "B": expected_b,
            "C": [150.0, 350.0],
        },
        index=test_dataset.index,
    )

    # Compare only the feature columns (sentinel columns are tested separately)
    feature_cols = [c for c in transformed_dataset.data.columns if not c.startswith(OUTLIER_NAN_MASK_PREFIX)]
    pd.testing.assert_frame_equal(transformed_dataset.data[feature_cols], expected_df)


def test_outlier_handler__handles_missing_features(
    train_dataset: TimeSeriesDataset,
):
    # Arrange
    test_data = pd.DataFrame(
        {"A": [0.5, 4.0], "C": [150.0, 350.0]},
        index=pd.date_range("2025-01-06", periods=2, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    outlier_handler = OutlierHandler(selection=FeatureSelection(include={"A", "B"}), mode="minmax")

    # Act
    outlier_handler.fit(train_dataset)
    transformed_dataset = outlier_handler.transform(test_dataset)

    # Assert
    # A is clipped, C unchanged, missing B is ignored
    expected_df = pd.DataFrame(
        {"A": [1.0, 3.0], "C": [150.0, 350.0]},  # A clipped, C unchanged
        index=test_dataset.index,
    )

    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)


def test_outlier_handler__transform_without_fit(test_dataset: TimeSeriesDataset):
    """Test that transform raises error when called without fitting."""

    # Arrange
    outlier_handler = OutlierHandler(selection=FeatureSelection(include={"A", "B"}))

    # Act & Assert
    with pytest.raises(NotFittedError):
        outlier_handler.transform(test_dataset)


def test_outlier_handler__nan_action_adds_sentinel_columns():
    """Test that NaN action adds sentinel mask columns for features with outliers."""

    # Arrange - train on narrow range so test data has outliers
    train_data = pd.DataFrame(
        {"load": [100.0, 110.0, 105.0, 108.0, 103.0], "temp": [20.0, 21.0, 20.5, 20.8, 20.3]},
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    # Test data with outliers in load (well outside mean±2*std) but not in temp
    test_data = pd.DataFrame(
        {"load": [105.0, 500.0, -200.0], "temp": [20.5, 20.8, 20.3]},
        index=pd.date_range("2025-01-06", periods=3, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    handler = OutlierHandler(
        selection=FeatureSelection(include={"load", "temp"}),
        mode="standard",
        outlier_action="nan",
    )

    # Act
    handler.fit(train_dataset)
    result = handler.transform(test_dataset)

    # Assert - sentinel column exists for load (has outliers)
    assert f"{OUTLIER_NAN_MASK_PREFIX}load__" in result.data.columns
    mask = result.data[f"{OUTLIER_NAN_MASK_PREFIX}load__"]
    assert mask.iloc[0] is np.bool_(False)  # 105.0 is within range
    assert mask.iloc[1] is np.bool_(True)  # 500.0 is outlier
    assert mask.iloc[2] is np.bool_(True)  # -200.0 is outlier

    # Sentinel column for temp should NOT exist (no outliers in temp)
    assert f"{OUTLIER_NAN_MASK_PREFIX}temp__" not in result.data.columns


def test_outlier_handler__clip_action_does_not_add_sentinel_columns():
    """Test that clip action does not add sentinel columns."""

    # Arrange
    train_data = pd.DataFrame(
        {"load": [100.0, 110.0, 105.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    test_data = pd.DataFrame(
        {"load": [500.0]},
        index=pd.date_range("2025-01-06", periods=1, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    handler = OutlierHandler(
        selection=FeatureSelection(include={"load"}),
        mode="standard",
        outlier_action="clip",
    )

    # Act
    handler.fit(train_dataset)
    result = handler.transform(test_dataset)

    # Assert - no sentinel columns for clip action
    sentinel_cols = [c for c in result.data.columns if c.startswith(OUTLIER_NAN_MASK_PREFIX)]
    assert sentinel_cols == []
