# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd

from openstef_beam.evaluation.reference_forecast import ReferenceForecastConfig, build_reference_forecast
from openstef_core.datasets import ForecastDataset


def _subset(index: pd.DatetimeIndex) -> ForecastDataset:
    return ForecastDataset(
        data=pd.DataFrame({"load": 0.0, "quantile_P50": 0.0}, index=index),
        sample_interval=timedelta(hours=1),
        target_column="load",
    )


def test_build_reference_forecast_uses_only_measurements_up_to_each_issue_time():
    # Arrange: the measurement value equals its hour, so the context median reveals the context window
    measurements = pd.Series(np.arange(24.0), index=pd.date_range("2025-01-01", periods=24, freq="h"))
    index = pd.date_range("2025-01-01T10:00", periods=4, freq="h")
    issue_times = pd.Series(pd.to_datetime(["2025-01-01T06:00"] * 2 + ["2025-01-01T08:00"] * 2), index=index)
    config = ReferenceForecastConfig(context_length=timedelta(hours=4), minimum_coverage=1.0)

    # Act
    reference = build_reference_forecast(
        config=config, measurements=measurements, subset=_subset(index), issue_times=issue_times
    )

    # Assert: contexts are hours 2-6 and 4-8
    assert reference.data["quantile_P50"].tolist() == [4.0, 4.0, 6.0, 6.0]


def test_build_reference_forecast_is_nan_when_context_coverage_is_too_low():
    # Arrange
    measurements = pd.Series(1.0, index=pd.date_range("2025-01-01", periods=3, freq="h"))
    index = pd.date_range("2025-01-01T10:00", periods=2, freq="h")
    issue_times = pd.Series(pd.Timestamp("2025-01-01T02:00"), index=index)
    config = ReferenceForecastConfig(context_length=timedelta(hours=4), minimum_coverage=1.0)

    # Act
    reference = build_reference_forecast(
        config=config, measurements=measurements, subset=_subset(index), issue_times=issue_times
    )

    # Assert
    assert reference.data["quantile_P50"].isna().all()
