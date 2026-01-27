# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for window iteration utilities.

Keep these tests small and deterministic: we only verify the behavior that
matters for production usage ("evaluate one specific window").
"""

from datetime import timedelta

import pandas as pd
import pytest

from openstef_beam.evaluation.models import Window
from openstef_beam.evaluation.window_iterators import iterate_by_window


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01T00:00", periods=6, freq="h")


@pytest.fixture
def window() -> Window:
    return Window(
        lag=timedelta(hours=0),
        size=timedelta(hours=2),
        stride=timedelta(hours=1),
    )


def test_iterate_by_window_returns_expected_windows_when_reference_date_is_none(
    index: pd.DatetimeIndex,
    window: Window,
) -> None:
    # Arrange
    sample_interval = timedelta(hours=1)

    # Act
    windows = list(iterate_by_window(index=index, window=window, sample_interval=sample_interval))

    # Assert
    # index = 00..05, size=2h, stride=1h => ends at 02,03,04,05
    assert len(windows) == 4

    for window_end, window_index in windows:
        expected_index = pd.date_range(
            start=window_end - window.size,
            end=window_end,
            freq=sample_interval,
            inclusive="left",
        )
        assert window_index.equals(expected_index)


def test_iterate_by_window_returns_single_window_when_reference_date_provided(
    index: pd.DatetimeIndex,
    window: Window,
) -> None:
    # Arrange
    sample_interval = timedelta(hours=1)
    reference_date = index[4]  # 2020-01-01 04:00

    # Act
    windows = list(
        iterate_by_window(
            index=index,
            window=window,
            sample_interval=sample_interval,
            reference_date=reference_date,
        )
    )

    # Assert
    assert len(windows) == 1
    window_end, window_index = windows[0]

    expected_index = pd.date_range(
        start=reference_date - window.size,
        end=reference_date,
        freq=sample_interval,
        inclusive="both",
    )
    assert window_end == reference_date
    assert window_index.equals(expected_index)
