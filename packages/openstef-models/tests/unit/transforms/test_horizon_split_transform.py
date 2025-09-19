# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for HorizonSplitTransform and concat_horizon_datasets_rowwise."""

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime
from openstef_models.transforms.horizon_split_transform import (
    HorizonSplitTransform,
)


@pytest.fixture
def horizon_differentiated_dataset() -> VersionedTimeSeriesDataset:
    """Create dataset with clear differences between horizons using string identifiers.

    Each feature has string values that indicate which horizon it should be available for,
    making it easy to verify the horizon filtering is working correctly.
    """
    timestamps = pd.date_range("2025-01-01T10:00:00", periods=4, freq="1h").to_series()

    return VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame({
            "timestamp": pd.concat([
                timestamps,
                timestamps,
                timestamps,
            ]),
            "available_at": pd.concat([
                timestamps - pd.Timedelta(hours=1),  # Short horizon: available 1 hour before
                timestamps - pd.Timedelta(hours=2),  # Medium horizon: available 2 hours before
                timestamps - pd.Timedelta(hours=24),  # Long horizon: available 24 hours before
            ]),
            "feature": [
                "short_h1",
                "short_h2",
                "short_h3",
                "short_h4",
                "medium_h1",
                "medium_h2",
                "medium_h3",
                "medium_h4",
                "long_h1",
                "long_h2",
                "long_h3",
                "long_h4",
            ],
        }),
        sample_interval=timedelta(hours=1),
    )


@pytest.mark.parametrize(
    ("horizons", "labels", "expected_count"),
    [
        pytest.param([LeadTime.from_string("PT1H")], ["short"], 1, id="single_custom_horizon"),
        pytest.param(
            [LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H"), LeadTime.from_string("PT24H")],
            ["short", "medium", "long"],
            3,
            id="three_horizons",
        ),
        pytest.param(
            [LeadTime.from_string("PT1H30M"), LeadTime.from_string("PT3H"), LeadTime.from_string("PT23H")],
            ["medium", "long", "long"],
            3,
            id="edge_case_horizons",
        ),
    ],
)
def test_horizon_split_transform_initialization_and_transform(
    horizon_differentiated_dataset: VersionedTimeSeriesDataset,
    horizons: list[LeadTime],
    labels: list[str],
    expected_count: int,
):
    """Test HorizonSplitTransform initialization and transformation with various horizon configurations."""
    # Arrange
    transform = HorizonSplitTransform(horizons=horizons)

    # Assert initialization
    assert transform.horizons == horizons
    assert len(transform.horizons) == expected_count

    # Act - transform dataset
    result = transform.transform(horizon_differentiated_dataset)

    # Assert transformation results
    assert len(result) == expected_count

    for horizon, label in zip(horizons, labels, strict=False):
        assert horizon in result
        horizon_dataset = result[horizon]
        assert isinstance(horizon_dataset, TimeSeriesDataset)
        assert horizon_dataset.sample_interval == horizon_differentiated_dataset.sample_interval

        values = [f"{label}_h{i}" for i in range(1, 5)]
        assert horizon_dataset.data["feature"].to_list() == values
