# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for multiprocessing utilities."""

# Fix for macOS multiprocessing hanging in tests
from openstef_core.utils import run_parallel


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


def test_run_parallel_multiple_processes():
    # Arrange
    items = [1, 2, 3, 4]
    expected = [2, 4, 6, 8]

    # Act
    result = run_parallel(double_number, items, n_processes=2)

    # Assert
    assert result == expected
