# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from pathlib import Path
from typing import override

import pandas as pd
import pytest
from pydantic import ValidationError

from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.target_provider import (
    SimpleTargetProvider,
    filter_away_flatline_chunks,
)
from openstef_core.datasets import VersionedTimeSeriesDataset


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


def test_get_predictors_for_target(tmp_path: Path, test_target: BenchmarkTarget):
    """Test that predictors are correctly concatenated from multiple sources."""
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    interval = timedelta(hours=1)

    weather = VersionedTimeSeriesDataset.from_dataframe(
        pd.DataFrame({"temp": range(3), "available_at": index}, index=index), interval
    )
    profiles = VersionedTimeSeriesDataset.from_dataframe(
        pd.DataFrame({"prof": range(3), "available_at": index}, index=index), interval
    )
    prices = VersionedTimeSeriesDataset.from_dataframe(
        pd.DataFrame({"price": range(3), "available_at": index}, index=index), interval
    )

    class TestProvider(SimpleTargetProvider[BenchmarkTarget, None]):
        @override
        def get_weather_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
            return weather

        @override
        def get_profiles(self) -> VersionedTimeSeriesDataset:
            return profiles

        @override
        def get_prices(self) -> VersionedTimeSeriesDataset:
            return prices

    provider = TestProvider(data_dir=tmp_path, use_profiles=True, use_prices=True)

    # Act
    result = provider.get_predictors_for_target(test_target)

    # Assert
    assert isinstance(result, VersionedTimeSeriesDataset)
    assert {"temp", "prof", "price"} <= set(result.feature_names)
    assert len(result.index) == 3


@pytest.mark.parametrize(
    (
        "values",
        "min_length",
        "threshold",
        "expected",
    ),
    [
        pytest.param(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 8.0],
            3,
            0.1,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan"), float("nan"), float("nan"), 7.0, 8.0],
            id="flatline-chunk-masked",
        ),
        pytest.param(
            [1.0, 2.0, 3.0, 4.0],
            2,
            0.1,
            [1.0, 2.0, 3.0, 4.0],
            id="no-flatline",
        ),
    ],
)
def test_filter_away_flatline_chunks_expected_series(
    values: list[float],
    min_length: int,
    threshold: float,
    expected: list[float],
) -> None:
    """Compare the filtered output with the expected flatline suppression result."""
    # Arrange
    index = pd.date_range("2023-01-01", periods=len(values), freq="h")
    series = pd.Series(values, index=index)

    # Act
    filtered = filter_away_flatline_chunks(series, min_length=min_length, threshold=threshold)

    # Assert: the filtered series matches the expected output
    expected_series = pd.Series(expected, index=index)
    pd.testing.assert_series_equal(filtered, expected_series)
