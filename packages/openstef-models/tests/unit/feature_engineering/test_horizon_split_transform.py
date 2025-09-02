# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for HorizonSplitTransform and concat_horizon_datasets_rowwise."""

from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime
from openstef_models.feature_engineering.horizon_split_transform import (
    HorizonSplitTransform,
    concat_horizon_datasets_rowwise,
)


@pytest.fixture
def sample_versioned_dataset() -> VersionedTimeSeriesDataset:
    """Create sample versioned dataset for testing."""
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01T10:00:00",
            "2025-01-01T11:00:00",
            "2025-01-01T12:00:00",
            "2025-01-01T13:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2025-01-01T10:05:00",
            "2025-01-01T11:05:00",
            "2025-01-01T12:05:00",
            "2025-01-01T13:05:00",
        ]),
        "load": [100.0, 110.0, 120.0, 130.0],
        "temperature": [20.0, 21.0, 22.0, 23.0],
    })
    return VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))


@pytest.mark.parametrize(
    ("horizons", "expected_count"),
    [
        pytest.param([LeadTime.from_string("PT36H")], 1, id="default_horizon"),
        pytest.param([LeadTime.from_string("PT1H")], 1, id="single_custom_horizon"),
        pytest.param([LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H")], 2, id="two_horizons"),
        pytest.param(
            [LeadTime.from_string("PT15M"), LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")],
            3,
            id="three_horizons",
        ),
    ],
)
def test_horizon_split_transform_initialization_and_transform(
    sample_versioned_dataset: VersionedTimeSeriesDataset, horizons: list[LeadTime], expected_count: int
):
    """Test HorizonSplitTransform initialization and transformation with various horizon configurations."""
    # Arrange
    transform = HorizonSplitTransform(horizons=horizons)

    # Assert initialization
    assert transform.horizons == horizons
    assert len(transform.horizons) == expected_count

    # Act - transform dataset
    result = transform.transform(sample_versioned_dataset)

    # Assert transformation results
    assert len(result) == expected_count

    for horizon in horizons:
        assert horizon in result
        horizon_dataset = result[horizon]
        assert isinstance(horizon_dataset, TimeSeriesDataset)
        assert horizon_dataset.sample_interval == sample_versioned_dataset.sample_interval

        # Verify data structure is preserved correctly
        assert isinstance(horizon_dataset.data.index, pd.DatetimeIndex)
        expected_columns = {"load", "temperature"}
        assert set(horizon_dataset.data.columns) == expected_columns

        # Data should not be empty (though may be filtered)
        assert not horizon_dataset.data.empty


def test_horizon_split_transform_default_initialization():
    """Test HorizonSplitTransform creates with default parameters."""
    # Arrange & Act
    transform = HorizonSplitTransform()

    # Assert
    assert len(transform.horizons) == 1
    assert transform.horizons[0] == LeadTime.from_string("PT36H")


@pytest.mark.parametrize(
    ("datasets_config", "expected_behavior"),
    [
        pytest.param(
            [{"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2}],
            {"type": "single", "sample_interval": timedelta(hours=1)},
            id="single_dataset",
        ),
        pytest.param(
            [
                {"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2},
                {"load": [120.0, 130.0], "temperature": [22.0, 23.0], "start": "2025-01-01 12:00", "periods": 2},
            ],
            {"type": "multiple", "sample_interval": timedelta(hours=1), "total_rows": 4},
            id="multiple_datasets",
        ),
        pytest.param(
            [
                {"load": [100.0], "start": "2025-01-01 10:00", "periods": 1, "interval": timedelta(hours=1)},
                {"load": [110.0], "start": "2025-01-01 11:00", "periods": 1, "interval": timedelta(minutes=30)},
            ],
            {"type": "mixed_intervals", "sample_interval": timedelta(hours=1)},
            id="preserves_first_sample_interval",
        ),
        pytest.param(
            [
                {"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2},
                {"load": [105.0, 115.0], "temperature": [20.5, 21.5], "start": "2025-01-01 10:00", "periods": 2},
            ],
            {"type": "overlapping", "sample_interval": timedelta(hours=1), "total_rows": 4},
            id="handles_overlapping_indices",
        ),
    ],
)
def test_concat_horizon_datasets_rowwise(datasets_config: list[dict[str, Any]], expected_behavior: dict[str, Any]):
    """Test concatenating horizon datasets with various configurations."""
    # Arrange
    horizon_datasets: dict[LeadTime, TimeSeriesDataset] = {}
    expected_sample_interval = expected_behavior["sample_interval"]

    for i, config in enumerate(datasets_config):
        data_dict = {k: v for k, v in config.items() if k not in {"start", "periods", "interval"}}
        index = pd.date_range(config["start"], periods=config["periods"], freq="1h")
        data = pd.DataFrame(data_dict, index=index)

        interval = config.get("interval", timedelta(hours=1))
        dataset = TimeSeriesDataset(data, interval)
        horizon_datasets[LeadTime.from_string(f"PT{i + 1}H")] = dataset

    # Act
    result = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == expected_sample_interval

    if expected_behavior["type"] == "single":
        # Single dataset should be identical to input
        original_dataset = next(iter(horizon_datasets.values()))
        pd.testing.assert_frame_equal(result.data, original_dataset.data)
    elif "total_rows" in expected_behavior:
        # Check expected number of rows
        assert len(result.data) == expected_behavior["total_rows"]


def test_concat_horizon_datasets_rowwise_empty_datasets():
    """Test concatenation with empty datasets."""
    # Arrange
    empty_data = pd.DataFrame(columns=["load", "temperature"])
    empty_data.index = pd.DatetimeIndex([], name="timestamp")
    empty_dataset = TimeSeriesDataset(empty_data, timedelta(hours=1))
    horizon_datasets = {LeadTime.from_string("PT1H"): empty_dataset}

    # Act
    result = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.data.empty
    assert result.sample_interval == empty_dataset.sample_interval


def test_horizon_split_transform_integration_with_concat():
    """Test that HorizonSplitTransform output can be properly concatenated."""
    # Arrange
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01T10:00:00",
            "2025-01-01T11:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2025-01-01T10:05:00",
            "2025-01-01T11:05:00",
        ]),
        "load": [100.0, 110.0],
    })
    versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

    horizons = [LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H")]
    transform = HorizonSplitTransform(horizons=horizons)

    # Act
    horizon_datasets = transform.transform(versioned_dataset)
    concatenated = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(concatenated, TimeSeriesDataset)
    assert "load" in concatenated.data.columns
    assert concatenated.sample_interval == versioned_dataset.sample_interval
