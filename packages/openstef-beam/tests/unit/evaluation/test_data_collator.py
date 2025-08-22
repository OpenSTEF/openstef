# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_beam.evaluation.data_collator import get_timeseries_coverage


@pytest.mark.parametrize(
    ("data", "sample_interval", "expected"),
    [
        pytest.param(
            pd.Series(
                [1, 2, 3],
                index=pd.to_datetime([
                    datetime.fromisoformat("2023-01-01"),
                    datetime.fromisoformat("2023-01-02"),
                    datetime.fromisoformat("2023-01-03"),
                ]),
            ),
            timedelta(days=1),
            timedelta(days=3),
            id="basic_case",
        ),
        pytest.param(
            pd.Series(
                [1, 2, 2],
                index=pd.to_datetime([
                    datetime.fromisoformat("2023-01-01"),
                    datetime.fromisoformat("2023-01-02"),
                    datetime.fromisoformat("2023-01-02"),
                ]),
            ),
            timedelta(days=1),
            timedelta(days=2),
            id="with_duplicates",
        ),
    ],
)
def test_get_timeseries_coverage(data: pd.DataFrame, sample_interval: timedelta, expected: timedelta):
    # Arrange - done in parametrize

    # Act
    result = get_timeseries_coverage(data, sample_interval)

    # Assert
    assert result == expected


def test_get_timeseries_coverage_raises_error():
    # Arrange
    data = pd.Series([1, 2, 3], index=[1, 2, 3])  # Not a DatetimeIndex
    sample_interval = timedelta(days=1)

    # Act / Assert
    with pytest.raises(TypeError, match="DataFrame index must be a DatetimeIndex"):
        get_timeseries_coverage(data, sample_interval)
