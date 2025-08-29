# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.validation_transforms.flatliner_check import FlatlinerCheckTransform


@pytest.mark.parametrize(
    ("load_values", "threshold_minutes", "detect_non_zero", "absolute_tolerance", "relative_tolerance", "expected"),
    [
        pytest.param([0, 0, 0, 0], 180, False, 0, 1e-5, True, id="flatliner_all_zeros"),
        pytest.param([2, 2, 2, 2], 180, True, 0, 1e-5, True, id="flatliner_all_nonzero"),
        pytest.param([1, 1, 2, 2], 120, True, 0, 1e-5, False, id="no_flatliner_within_threshold_minutes"),
        pytest.param([1, 2, 3, 4], 120, True, 0, 1e-5, False, id="no_flatliner_all_different"),
        pytest.param([1, 1, 1, 0.5], 180, True, 0, 0.5, True, id="flatliner_within_tolerance"),
        pytest.param([0, 0, 0, np.nan], 180, False, 0, 1e-5, True, id="flatliner_trailing_nan"),
        pytest.param([1, 0, 0, np.nan], 180, False, 0, 1e-5, False, id="no_flatliner_trailing_nan"),
        pytest.param([1, 1, 2, 2], 180, True, 1, 0, True, id="flatliner_within_threshold_minutes_absolute_tolerance"),
        pytest.param([0.5, 1, 2, 2], 180, True, 0.5, 0, False, id="no_flatliner_absolute_tolerance"),
        pytest.param([1, 2, 3, 2], 180, True, 0.5, 0.5, True, id="flatliner_within_combined_tolerance"),
        pytest.param([1, 2, 4, 2], 180, True, 0.5, 0.5, False, id="no_flatliner_combined_tolerance"),
    ],
)
def test_detect_ongoing_flatliner(
    load_values: list[float],
    threshold_minutes: int,
    detect_non_zero: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
    expected: bool,
) -> None:
    # Arrange
    data = TimeSeriesDataset(
        data=pd.DataFrame(
            data={"load": load_values},
            index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )
    transform = FlatlinerCheckTransform(
        flatliner_threshold_minutes=threshold_minutes,
        detect_non_zero_flatliner=detect_non_zero,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    # Act
    transform.fit(data)
    # Assert
    assert transform.flatliner_indicator is expected


def test_fit_raises_on_missing_load_column() -> None:
    # Arrange
    idx = [datetime.fromisoformat("2025-01-01T00:00:00")]
    df = pd.DataFrame({"not_load": [1]}, index=idx)
    dataset = TimeSeriesDataset(df, timedelta(minutes=1))
    transform = FlatlinerCheckTransform()
    # Act & Assert
    with pytest.raises(ValueError, match=r"The DataFrame must contain a 'load' column."):
        transform.fit(dataset)


def test_transform_returns_input() -> None:
    # Arrange
    idx = [datetime.fromisoformat("2025-01-01T00:00:00")]
    df = pd.DataFrame({"load": [1]}, index=idx)
    dataset = TimeSeriesDataset(df, timedelta(minutes=1))
    transform = FlatlinerCheckTransform()
    # Act
    result = transform.transform(dataset)
    # Assert
    assert result is dataset
