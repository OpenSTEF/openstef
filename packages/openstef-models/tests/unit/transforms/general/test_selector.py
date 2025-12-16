# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_models.transforms.general import Selector
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.mark.parametrize(
    ("feature_selection", "expected_features"),
    [
        pytest.param(
            FeatureSelection(include={"load", "temperature"}),
            {"load", "temperature"},
            id="include_subset",
        ),
        pytest.param(
            FeatureSelection(exclude={"humidity"}),
            {"load", "temperature"},
            id="exclude_subset",
        ),
        pytest.param(
            FeatureSelection.ALL,
            {"load", "temperature", "humidity"},
            id="all_features",
        ),
        pytest.param(
            FeatureSelection.NONE,
            set(),
            id="no_features",
        ),
    ],
)
def test_selector__selects_specified_features(
    feature_selection: FeatureSelection,
    expected_features: set[str],
) -> None:
    """Test that Selector selects only the specified features."""
    # Arrange
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 120.0],
            "temperature": [20.0, 22.0, 23.0],
            "humidity": [60.0, 65.0, 70.0],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    selector = Selector(selection=feature_selection)

    # Act
    transformed = selector.fit_transform(dataset)

    # Assert
    assert set(transformed.feature_names) == expected_features
    for feature in expected_features:
        pd.testing.assert_series_equal(transformed.data[feature], dataset.data[feature])


def test_selector__fit_adds_target_to_include() -> None:
    """Test that fit adds target column to include set when using ForecastInputDataset."""
    # Arrange
    data = pd.DataFrame(
        {"load": [100.0, 110.0], "temperature": [20.0, 22.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="1h"),
    )
    dataset = ForecastInputDataset(data, timedelta(hours=1), target_column="load")
    selector = Selector(selection=FeatureSelection(include={"temperature"}))

    # Act
    transformed = selector.fit_transform(dataset)

    # Assert
    assert selector.selection.include == {"temperature", "load"}
    assert "load" in transformed.feature_names
    assert "temperature" in transformed.feature_names
