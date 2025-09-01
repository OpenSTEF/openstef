# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import InsufficientlyCompleteError
from openstef_core.feature_engineering.validation_transforms.completeness_check import (
    CompletenessCheckTransform,
)


@pytest.mark.parametrize(
    ("columns", "weights", "threshold", "expected_completeness", "expected_sufficiently_complete"),
    [
        pytest.param(
            None,
            {"radiation": 1.0, "temperature": 1.0, "wind_speed": 1.0},
            0.5,
            0.75,
            True,
            id="sufficient_equal_weights",
        ),
        pytest.param(None, None, 0.5, 0.75, True, id="sufficient_default_weights"),
        pytest.param(
            None,
            {"radiation": 1.0, "temperature": 1.0, "wind_speed": 1.0},
            0.8,
            0.75,
            False,
            id="insufficient_equal_weights",
        ),
        pytest.param(
            None,
            {"radiation": 1.0, "temperature": 3.0, "wind_speed": 1.0},
            0.5,
            0.65,
            True,
            id="sufficient_unequal_weights",
        ),
        pytest.param(
            ["radiation", "temperature"],
            {"radiation": 1.0, "temperature": 1.0},
            0.5,
            0.625,
            True,
            id="sufficient_partial_columns",
        ),
    ],
)
def test_calculate_completeness(
    columns: list[str] | None,
    weights: dict[str, float],
    threshold: float,
    expected_completeness: float,
    expected_sufficiently_complete: bool,
):
    data = pd.DataFrame(
        {
            "radiation": [100, 110, 110, np.nan],
            "temperature": [20, np.nan, np.nan, 21],
            "wind_speed": [5, 6, 6, 3],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    transform = CompletenessCheckTransform(
        columns=columns, weights=weights, completeness_threshold=threshold, error_on_insufficient_completeness=False
    )
    transform.fit(dataset)

    assert transform.completeness == expected_completeness
    assert transform.is_sufficiently_complete is expected_sufficiently_complete


def test_fit_raises_error_on_insufficient_completeness() -> None:
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100, 110, 110, np.nan],
            "temperature": [20, np.nan, np.nan, 21],
            "wind_speed": [5, 6, 6, 3],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    transform = CompletenessCheckTransform(completeness_threshold=0.8)

    # Act & Assert
    with pytest.raises(InsufficientlyCompleteError):
        transform.fit(dataset)


def test_transform_insufficiently_complete_data() -> None:
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100, np.nan, np.nan, np.nan],
            "temperature": [20, np.nan, 24, np.nan],
            "wind_speed": [np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="15min"),
    )
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    transform = CompletenessCheckTransform(completeness_threshold=0.5)

    # Act
    with pytest.raises(InsufficientlyCompleteError):
        transform.fit(dataset)

    # Assert
    assert transform.completeness == 0.25
    assert transform.is_sufficiently_complete is False


def test_transform_not_check_on_completeness() -> None:
    # Arrange
    idx = [datetime.fromisoformat("2025-01-01T00:00:00")]
    df = pd.DataFrame({"radiation": [1]}, index=idx)
    dataset = TimeSeriesDataset(df, timedelta(minutes=1))
    transform = CompletenessCheckTransform(check_on_transform=False)

    # Act
    result = transform.transform(dataset)

    # Assert
    assert result is dataset
