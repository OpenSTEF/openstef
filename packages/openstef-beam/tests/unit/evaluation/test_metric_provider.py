# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from openstef_beam.evaluation.metric_providers import (
    CRPSSProvider,
    CRPSSSampleWeightedProvider,
    MetricProvider,
    RCSProvider,
    RIQDProvider,
    RMAEPeakHoursProvider,
)
from openstef_beam.evaluation.models import Filtering
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import AvailableAt, LeadTime, Quantile


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
def test_rcs_provider_symmetric_quantile_logic(
    quantiles: list[float], expected_pairs: list[tuple[float, float]], expected_skipped: list[float]
) -> None:
    """Test that RCSProvider correctly identifies and processes symmetric quantile pairs."""
    # Arrange
    provider = RCSProvider()

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
    with patch("openstef_beam.evaluation.metric_providers.rcs", return_value=0.75) as mock_rcs:
        result = provider(subset)

    # Assert
    # Check that RCS was called for each expected quantile pair
    assert mock_rcs.call_count == len(expected_pairs)

    # Verify that results contain metrics for quantiles with symmetric counterparts
    expected_result_quantiles = {Quantile(pair[0]) for pair in expected_pairs}
    assert set(result.keys()) == expected_result_quantiles

    # Verify each result contains the RCS metric
    for quantile in expected_result_quantiles:
        assert "RCS" in result[quantile]
        assert result[quantile]["RCS"] == 0.75

    # Verify that skipped quantiles are not in the results
    for skipped_quantile in expected_skipped:
        assert Quantile(skipped_quantile) not in result


def _skill_subset() -> ForecastDataset:
    """Median-only forecast, so that CRPS reduces to the mean absolute error."""
    index = pd.date_range(datetime.fromisoformat("2025-05-16T00:00:00"), periods=4, freq="1h")
    return ForecastDataset(
        data=pd.DataFrame({"load": [0.0, 10.0, 0.0, 10.0], "quantile_P50": [4.0, 9.0, 4.0, 9.0]}, index=index),
        target_column="load",
        sample_interval=timedelta(hours=1),
    )


def _skill_ground_truth(subset: ForecastDataset) -> ForecastInputDataset:
    """Two weeks of constant load of 5.0 before the subset, so the reference forecast is 5.0 everywhere."""
    history_index = pd.date_range(
        datetime.fromisoformat("2025-05-01T00:00:00"), subset.index[0], freq="1h", inclusive="left"
    )
    load = pd.concat([pd.Series(5.0, index=history_index), cast(pd.Series, subset.target_series)])
    return ForecastInputDataset(
        data=load.to_frame("load"), sample_interval=subset.sample_interval, target_column="load"
    )


@pytest.mark.parametrize(
    ("reference_filtering", "expected"),
    [
        # Cutoff 2025-05-02T06:00 (i=30); context (04:00, 06:00] holds 29 and 30
        pytest.param(LeadTime.from_string("PT6H"), 29.5, id="lead_time"),
        # Cutoff 2025-05-01T06:00 (i=6); context (04:00, 06:00] holds 5 and 6
        pytest.param(AvailableAt.from_string("D-1T0600"), 5.5, id="available_at"),
        # Cutoff 2025-05-01T00:00 is the first measurement; 1 of 2 required samples
        pytest.param(LeadTime.from_string("PT36H"), np.nan, id="insufficient_history"),
    ],
)
def test_reference_forecast_uses_rolling_quantile_at_cutoff(reference_filtering: Filtering, expected: float) -> None:
    # Arrange: hourly measurements whose value equals the hours since the start
    measurement_index = pd.date_range(datetime.fromisoformat("2025-05-01T00:00:00"), periods=49, freq="1h")
    measurements = pd.Series(np.arange(49, dtype=float), index=measurement_index)
    provider = CRPSSProvider(
        reference_filtering=reference_filtering, context_length=timedelta(hours=2), minimum_coverage=1.0
    )

    # Act
    reference = provider.reference_forecast(
        measurements=measurements,
        index=pd.DatetimeIndex([datetime.fromisoformat("2025-05-02T12:00:00")]),
        sample_interval=timedelta(hours=1),
        quantiles=[Quantile(0.5)],
    )

    # Assert
    np.testing.assert_array_equal(reference.to_numpy(), [[expected]])


@pytest.mark.parametrize(
    ("provider", "metric_name", "expected"),
    [
        # Errors of [4, 1, 4, 1] average to 2.5 against a reference error of 5.0
        pytest.param(CRPSSProvider(), "CRPSS", 50.0, id="unweighted"),
        # Weights of [0.1, 1, 0.1, 1] relative to the reference scale of 5.0 favour the high load samples
        pytest.param(CRPSSSampleWeightedProvider(), "CRPSS_sample_weighted", 74.545455, id="sample_weighted"),
    ],
)
def test_crpss_providers_score_forecast_against_reference_from_ground_truth(
    provider: MetricProvider, metric_name: str, expected: float
) -> None:
    # Arrange
    subset = _skill_subset()

    # Act
    result = provider(subset, _skill_ground_truth(subset))

    # Assert
    assert result["global"][metric_name] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("provider", "metric_name"),
    [
        pytest.param(CRPSSProvider(), "CRPSS", id="unweighted"),
        pytest.param(CRPSSSampleWeightedProvider(), "CRPSS_sample_weighted", id="sample_weighted"),
    ],
)
@pytest.mark.parametrize("with_ground_truth", [pytest.param(True, id="no_history"), pytest.param(False, id="none")])
def test_crpss_providers_return_nan_without_measurement_history(
    provider: MetricProvider, metric_name: str, with_ground_truth: bool
) -> None:
    # Arrange: ground truth without history before the reference cutoff, or no ground truth at all
    subset = _skill_subset()
    ground_truth = (
        ForecastInputDataset(data=subset.data[["load"]], sample_interval=subset.sample_interval, target_column="load")
        if with_ground_truth
        else None
    )

    # Act
    result = provider(subset, ground_truth)

    # Assert
    assert np.isnan(result["global"][metric_name])
