# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for multiprocessing utilities."""

# Fix for macOS multiprocessing hanging in tests
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Literal

import pytest

from openstef_core.utils import align_datetime, run_parallel


def double_number(n: int) -> int:
    """Simple function for testing parallel execution."""
    return n * 2


def test_run_parallel_single_process():
    # Arrange
    items = [1, 2, 3, 4]
    expected = [2, 4, 6, 8]

    # Act
    result = run_parallel(double_number, items, n_processes=1)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("mode"),
    [
        pytest.param("loky", id="loky"),
        pytest.param("fork", id="fork"),
    ],
)
def test_run_parallel_multiple_processes(mode: Literal["loky", "fork"]):
    # Arrange - note: we can't use double number since it gives import issues with loki, since testing is not a real module
    items = [datetime(year=2025, month=1, day=i, hour=i, tzinfo=UTC) for i in range(1, 5)]
    expected = [datetime(year=2025, month=1, day=i + 1, hour=0, tzinfo=UTC) for i in range(1, 5)]

    # Act
    function = partial(align_datetime, interval=timedelta(days=1))
    result = run_parallel(process_fn=function, items=items, n_processes=2, mode=mode)

    # Assert
    assert result == expected
