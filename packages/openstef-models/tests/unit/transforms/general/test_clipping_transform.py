# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_models.transforms.general import ClippingTransform


@pytest.fixture
def train_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def test_dataset() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [0.5, 4.0], "B": [5.0, 35.0], "C": [150.0, 350.0]},
            index=pd.date_range("2025-01-06", periods=2, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def clipper() -> ClippingTransform:
    return ClippingTransform(column_names=["A", "B", "D"])


def test_feature_clipper_fit(clipper: ClippingTransform, train_dataset: TimeSeriesDataset):
    """Test if the fit method correctly computes min and max values."""
    clipper.fit(train_dataset)
    pd.testing.assert_series_equal(clipper._feature_mins, pd.Series([1.0, 10.0, np.nan], index=["A", "B", "D"]))
    pd.testing.assert_series_equal(clipper._feature_maxs, pd.Series([3.0, 30.0, np.nan], index=["A", "B", "D"]))


def test_feature_clipper_transform(
    clipper: ClippingTransform, train_dataset: TimeSeriesDataset, test_dataset: TimeSeriesDataset
):
    """Test if the transform method correctly clips values."""
    clipper.fit(train_dataset)
    transformed_dataset = clipper.transform(test_dataset)
    expected_df = pd.DataFrame(
        {
            "A": [1.0, 3.0],  # Clipped to range [1.0, 3.0]
            "B": [10.0, 30.0],  # Clipped to range [10.0, 30.0]
            "C": [150.0, 350.0],  # Unchanged
        },
        index=test_dataset.index,
    )
    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)
    assert transformed_dataset.sample_interval == test_dataset.sample_interval


def test_feature_clipper_invalid_column(train_dataset: TimeSeriesDataset):
    """Test behavior when a column that doesn't exist is specified."""
    clipper_with_invalid_column = ClippingTransform(column_names=["E"])
    clipper_with_invalid_column.fit(train_dataset)
    assert np.isnan(clipper_with_invalid_column._feature_mins["E"])
    assert np.isnan(clipper_with_invalid_column._feature_maxs["E"])


def test_feature_clipper_transform_without_fit(clipper: ClippingTransform, test_dataset: TimeSeriesDataset):
    """Test behavior when transform is called without fitting."""
    clipper = ClippingTransform(column_names=["A", "B"])
    # The transform should raise TransformNotFittedError when not fitted
    with pytest.raises(TransformNotFittedError, match=r"ClippingTransform.*has not been fitted"):
        clipper.transform(test_dataset)
