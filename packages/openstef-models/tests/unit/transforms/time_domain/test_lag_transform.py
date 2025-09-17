# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the VersionedLagTransform."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import LeadTime
from openstef_models.transforms.time_domain.lag_transform import VersionedLagTransform


def test_multiple_lag_features():
    """Test that multiple lags create separate features with correctly shifted timestamps.

    This test covers both single and multiple lag functionality:
    - Single lag: How -1h lag shifts timestamps forward by 1 hour
    - Multiple lags: How -1h and -2h lags create different shift patterns
    - Dataset union: How multiple lag data parts combine into unified index
    """
    # Arrange
    simple_versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
            ]),
            "available_at": pd.to_datetime([
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
            ]),
            "load": [100.0, 110.0, 120.0, 130.0],
        }),
        sample_interval=timedelta(hours=1),
    )
    transform = VersionedLagTransform(column="load", lags=[timedelta(hours=-1), timedelta(hours=-2)])

    # Act
    result = transform.transform(simple_versioned_dataset)
    snapshot = result.select_version()

    # Assert
    lag_1h_col = "load_lag_-PT1H"
    lag_2h_col = "load_lag_-PT2H"

    # How multiple lags work:
    # Original data: 10:00->100, 11:00->110, 12:00->120, 13:00->130
    # 1h lag part: 11:00->100, 12:00->110, 13:00->120, 14:00->130 (shifted +1h)
    # 2h lag part: 12:00->100, 13:00->110, 14:00->120, 15:00->130 (shifted +2h)
    # Combined index (union): [10:00, 11:00, 12:00, 13:00, 14:00, 15:00]

    # Expected results:
    # 1h lag: [NaN, 100, 110, 120, 130, NaN] - available at timestamps 11:00-14:00
    # 2h lag: [NaN, NaN, 100, 110, 120, 130] - available at timestamps 12:00-15:00
    expected_1h = pd.Series(
        [np.nan, 100.0, 110.0, 120.0, 130.0, np.nan],
        index=pd.DatetimeIndex(
            [
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
                "2025-01-01T14:00:00",
                "2025-01-01T15:00:00",
            ],
            name="timestamp",
        ),
        name=lag_1h_col,
    )

    expected_2h = pd.Series(
        [np.nan, np.nan, 100.0, 110.0, 120.0, 130.0],
        index=pd.DatetimeIndex(
            [
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
                "2025-01-01T14:00:00",
                "2025-01-01T15:00:00",
            ],
            name="timestamp",
        ),
        name=lag_2h_col,
    )

    actual_1h = snapshot.data[lag_1h_col]
    actual_2h = snapshot.data[lag_2h_col]

    pd.testing.assert_series_equal(actual_1h, expected_1h, check_freq=False)
    pd.testing.assert_series_equal(actual_2h, expected_2h, check_freq=False)


def test_lag_transform_with_horizon_filtering():
    """Test how lag values determine which data version quality is accessed based on lead time constraints.

    This test demonstrates the core insight of versioned lag transforms: longer lags require more advance notice,
    which forces the system to use higher quality (but later available) data versions.

    Scenario Setup:
    - Signal has 3 versions: "bad" (3d delay), "okay" (5d delay), "best" (10d delay)
    - Forecasting with 2-day lead time
    - Different lag values require different amounts of advance notice

    The Mathematics:
    For a forecast at time T with lag L and lead time H:
    - Data must be available by: T - L - H
    - Available data versions depend on this availability deadline

    Expected Behavior:
    - 5-day lag: needs data available by T-(-5d)-2d = T+3d → "bad" version (3d delay) just qualifies
    - 7-day lag: needs data available by T-(-7d)-2d = T+5d → "okay" version (5d delay) qualifies
    - 12-day lag: needs data available by T-(-12d)-2d = T+10d → "best" version (10d delay) qualifies
    """
    # Arrange
    timestamps = pd.date_range("2025-01-01T10:00:00", periods=5, freq="D")

    multi_version_signal_dataset = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.concat([
            pd.DataFrame({
                "timestamp": timestamps,
                "available_at": timestamps + timedelta(days=3),
                "signal": ["bad"] * len(timestamps),
            }),
            pd.DataFrame({
                "timestamp": timestamps,
                "available_at": timestamps + timedelta(days=5),
                "signal": ["okay"] * len(timestamps),
            }),
            pd.DataFrame({
                "timestamp": timestamps,
                "available_at": timestamps + timedelta(days=10),
                "signal": ["best"] * len(timestamps),
            }),
        ]),
        sample_interval=timedelta(days=1),
    )

    lead_time = LeadTime.from_string("P2D")  # 2-day forecast horizon
    transform = VersionedLagTransform(
        column="signal", lags=[timedelta(days=-5), timedelta(days=-7), timedelta(days=-12)]
    )

    # Act
    result = transform.transform(multi_version_signal_dataset)

    # Assert
    snapshot = result.filter_by_lead_time(lead_time).select_version(available_before=None)

    assert snapshot.data["signal_lag_-P5D"].dropna().eq("bad").all(), "5-day lag should access 'bad' version"
    assert snapshot.data["signal_lag_-P7D"].dropna().eq("okay").all(), "7-day lag should access 'okay' version"
    assert snapshot.data["signal_lag_-P12D"].dropna().eq("best").all(), "12-day lag should access 'best' version"

    (
        pd.testing.assert_index_equal(
            snapshot.data["signal_lag_-P5D"].dropna().index,
            pd.date_range("2025-01-06T10:00:00", periods=5, freq="D", name="timestamp"),
        ),
        "5-day lag index mismatch",
    )
    (
        pd.testing.assert_index_equal(
            snapshot.data["signal_lag_-P7D"].dropna().index,
            pd.date_range("2025-01-08T10:00:00", periods=5, freq="D", name="timestamp"),
        ),
        "7-day lag index mismatch",
    )
    (
        pd.testing.assert_index_equal(
            snapshot.data["signal_lag_-P12D"].dropna().index,
            pd.date_range("2025-01-13T10:00:00", periods=5, freq="D", name="timestamp"),
        ),
        "12-day lag index mismatch",
    )


def test_missing_column_error():
    """Test that lag transform raises MissingColumnsError when trying to create lag of non-existent column.

    This ensures proper error handling when configuration specifies a column that doesn't exist
    in the dataset, preventing silent failures in production pipelines.
    """
    # Arrange
    simple_versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(
        data=pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
            ]),
            "available_at": pd.to_datetime([
                "2025-01-01T10:00:00",
                "2025-01-01T11:00:00",
                "2025-01-01T12:00:00",
                "2025-01-01T13:00:00",
            ]),
            "load": [100.0, 110.0, 120.0, 130.0],
        }),
        sample_interval=timedelta(hours=1),
    )
    transform = VersionedLagTransform(column="non_existent", lags=[timedelta(hours=-1)])

    # Act & Assert
    # Verify that attempting to create lag features for a non-existent column fails appropriately
    with pytest.raises(MissingColumnsError) as exc_info:
        transform.transform(simple_versioned_dataset)
    assert "non_existent" in str(exc_info.value)
