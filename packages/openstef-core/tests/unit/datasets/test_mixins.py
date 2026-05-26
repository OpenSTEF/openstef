# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset


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
def test_get_timeseries_coverage(data: pd.Series, sample_interval: timedelta, expected: timedelta):
    # Arrange - done in parametrize
    dataset = TimeSeriesDataset(
        data=data.to_frame(name="value"),
        sample_interval=sample_interval,
    )

    # Act
    result = dataset.calculate_time_coverage()

    # Assert
    assert result == expected
