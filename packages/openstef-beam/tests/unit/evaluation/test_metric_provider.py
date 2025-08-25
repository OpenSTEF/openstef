# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from openstef_beam.evaluation.metric_providers import RMAEPeakHoursProvider
from openstef_beam.evaluation.models import EvaluationSubset
from openstef_core.datasets import TimeSeriesDataset
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

    predictions = TimeSeriesDataset(
        data=pd.DataFrame({"quantile_P50": range(len(times))}, index=index),
        sample_interval=timedelta(minutes=15),
    )
    ground_truth = TimeSeriesDataset(
        data=pd.DataFrame({"value": range(len(times))}, index=index),
        sample_interval=timedelta(minutes=15),
    )
    subset = EvaluationSubset(predictions=predictions, ground_truth=ground_truth)
    # Act
    with patch("openstef_beam.evaluation.metric_providers.rmae", return_value=0.1) as mock_rmae:
        result = provider(subset)

    # Assert
    call_args = mock_rmae.call_args[1]
    assert len(call_args["y_true"]) == num_timestamps_after_filtering
    assert len(call_args["y_pred"]) == num_timestamps_after_filtering

    assert Quantile(0.5) in result
    assert result[Quantile(0.5)]["rMAE_peak_hours"] == 0.1
