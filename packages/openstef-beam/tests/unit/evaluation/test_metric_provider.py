# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from openstef_beam.evaluation.metric_providers import RIQDProvider, RMAEPeakHoursProvider
from openstef_core.datasets import ForecastDataset
from openstef_core.types import Quantile


@pytest.mark.parametrize(
    ("start_peak_hours", "end_peak_hours", "num_timestamps_after_filtering"),
    [
        (7, 20, 52),
        (8, 20, 48),
        (0, 24, 96),
    ],
    ids=["default_peak_hours", "custom_peak_hours", "full_day_peak_hours"],
)
def test_peak_hours_filtering(start_peak_hours: int, end_peak_hours: int, num_timestamps_after_filtering: int):
    # Arrange
    provider = RMAEPeakHoursProvider(start_peak_hours=start_peak_hours, end_peak_hours=end_peak_hours)
    # Create 24-hour dataset
    start_time = datetime.fromisoformat("2025-01-01T00:00:00")
    times = [start_time + timedelta(minutes=i) for i in range(0, 1440, 15)]
    index = pd.DatetimeIndex(times)
    subset = ForecastDataset(
        data=pd.DataFrame(
            {
                "load": range(len(times)),
                "quantile_P50": range(len(times)),
                "horizon": timedelta(hours=24),
            },
            index=index,
        ),
        target_column="load",
        sample_interval=timedelta(minutes=15),
    )

    # Act
    with patch("openstef_beam.evaluation.metric_providers.rmae", return_value=0.1) as mock_rmae:
        result = provider(subset)

    # Assert
    call_args = mock_rmae.call_args[1]
    assert len(call_args["y_true"]) == num_timestamps_after_filtering
    assert len(call_args["y_pred"]) == num_timestamps_after_filtering

    assert Quantile(0.5) in result
    assert result[Quantile(0.5)]["rMAE_peak_hours"] == 0.1


@pytest.mark.parametrize(
    ("quantiles", "expected_pairs", "expected_skipped"),
    [
        # Standard symmetric quantiles
        ([0.1, 0.5, 0.9], [(0.1, 0.9), (0.9, 0.1)], [0.5]),
        # More quantiles with multiple symmetric pairs
        ([0.05, 0.25, 0.5, 0.75, 0.95], [(0.05, 0.95), (0.25, 0.75), (0.75, 0.25), (0.95, 0.05)], [0.5]),
        # Missing symmetric counterpart
        ([0.1, 0.5, 0.8], [], [0.1, 0.5, 0.8]),
        # Single quantile (median)
        ([0.5], [], [0.5]),
        # Asymmetric quantiles
        ([0.1, 0.3, 0.7], [(0.3, 0.7), (0.7, 0.3)], [0.1]),
    ],
    ids=["standard_symmetric", "multiple_pairs", "missing_counterpart", "single_median", "asymmetric_partial"],
)
def test_riqd_provider_symmetric_quantile_logic(
    quantiles: list[float], expected_pairs: list[tuple[float, float]], expected_skipped: list[float]
) -> None:
    """Test that RIQDProvider correctly identifies and processes symmetric quantile pairs."""
    # Arrange
    provider = RIQDProvider()

    # Create test data
    start_time = datetime.fromisoformat("2025-01-01T00:00:00")
    times = [start_time + timedelta(hours=i) for i in range(24)]
    index = pd.DatetimeIndex(times)

    # Create predictions with specified quantiles
    quantile_data = {}
    for i, q in enumerate(quantiles):
        quantile_data[f"quantile_P{int(q * 100):02d}"] = [i * 10 + j for j in range(24)]

    subset = ForecastDataset(
        data=pd.DataFrame(
            data={
                **quantile_data,
                "horizon": timedelta(hours=24),
                "load": range(24),
            },
            index=index,
        ),
        target_column="load",
        sample_interval=timedelta(hours=1),
    )

    # Act
    with patch("openstef_beam.evaluation.metric_providers.riqd", return_value=0.5) as mock_riqd:
        result = provider(subset)

    # Assert
    # Check that rIQD was called for each expected quantile pair
    assert mock_riqd.call_count == len(expected_pairs)

    # Verify that results contain metrics for quantiles with symmetric counterparts
    expected_result_quantiles = {Quantile(pair[0]) for pair in expected_pairs}
    assert set(result.keys()) == expected_result_quantiles

    # Verify each result contains the rIQD metric
    for quantile in expected_result_quantiles:
        assert "rIQD" in result[quantile]
        assert result[quantile]["rIQD"] == 0.5

    # Verify that skipped quantiles are not in the results
    for skipped_quantile in expected_skipped:
        assert Quantile(skipped_quantile) not in result
