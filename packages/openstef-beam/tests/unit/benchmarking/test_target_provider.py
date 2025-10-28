# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import pytest
from pydantic import ValidationError

from openstef_beam.benchmarking.models import BenchmarkTarget


@pytest.mark.parametrize(
    ("benchmark_start_str", "benchmark_end_str", "train_start_str", "expected_valid"),
    [
        pytest.param("2023-01-15", "2023-02-15", "2023-01-01", True, id="valid-time-ranges"),
        pytest.param("2023-01-15", "2023-01-15", "2023-01-01", False, id="invalid-benchmark-end-equals-start"),
    ],
)
def test_temporal_validation(
    benchmark_start_str: str, benchmark_end_str: str, train_start_str: str, expected_valid: bool
):
    """Test that temporal constraints are properly validated."""
    # Arrange
    benchmark_start = datetime.fromisoformat(benchmark_start_str)
    benchmark_end = datetime.fromisoformat(benchmark_end_str)
    train_start = datetime.fromisoformat(train_start_str)

    # Act & Assert
    if expected_valid:
        BenchmarkTarget(
            name="test_target",
            description="Test target",
            latitude=52.0,
            longitude=5.0,
            limit=100.0,
            benchmark_start=benchmark_start,
            benchmark_end=benchmark_end,
            train_start=train_start,
        )
    else:
        with pytest.raises(ValidationError):
            BenchmarkTarget(
                name="test_target",
                description="Test target",
                latitude=52.0,
                longitude=5.0,
                limit=100.0,
                benchmark_start=benchmark_start,
                benchmark_end=benchmark_end,
                train_start=train_start,
            )


@pytest.fixture
def test_target() -> BenchmarkTarget:
    """Create a list of test targets."""
    return BenchmarkTarget(
        name="location2",
        description="Test location 2",
        latitude=53.0,
        longitude=6.0,
        limit=200.0,
        benchmark_start=datetime.fromisoformat("2023-01-15"),
        benchmark_end=datetime.fromisoformat("2023-02-15"),
        train_start=datetime.fromisoformat("2023-01-01"),
    )
