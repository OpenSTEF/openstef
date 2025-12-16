# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.general import Flagger
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.fixture
def train_dataset() -> TimeSeriesDataset:
    """Training dataset with three features A, B, C."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0], "C": [1.0, 2.0, 3.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def test_dataset() -> TimeSeriesDataset:
    """Test dataset with values outside training ranges."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [2, 2], "B": [0.0, 2.0], "C": [1, 4]},
            index=pd.date_range("2025-01-06", periods=2, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


def test_flagger__fit_transform(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
):
    """Test fit and transform flags correctly leaves other columns unchanged."""
    # Arrange
    flagger = Flagger(selection=FeatureSelection(include={"A", "B", "C"}))

    # Act
    flagger.fit(train_dataset)
    transformed_dataset = flagger.transform(test_dataset)

    # Assert
    # Column C should remain unchanged
    expected_df = pd.DataFrame(
        {
            "A": [1, 1],
            "B": [0, 1],
            "C": [0, 0],  # Unchanged
        },
        index=test_dataset.index,
    )
    pd.testing.assert_frame_equal(transformed_dataset.data, expected_df)
    assert transformed_dataset.sample_interval == test_dataset.sample_interval
