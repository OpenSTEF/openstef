# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.testing import create_timeseries_dataset
from openstef_core.types import LeadTime
from openstef_models.transforms.time_domain.lags_adder import (
    LagsAdder,
    generate_autocorr_lags,
    generate_day_lags,
    generate_minute_lags,
)


@pytest.mark.parametrize(
    ("max_horizon", "expected_lags"),
    [
        pytest.param(
            timedelta(minutes=15),
            26,  # 15min, 30min, 45min + 1h-23h = 3 + 23 = 26
            id="15min_horizon_includes_all_subhourly_and_hourly",
        ),
        pytest.param(
            timedelta(hours=1),
            23,  # 1h-23h = 23, no subhourly lags
            id="1h_horizon_excludes_subhourly",
        ),
        pytest.param(
            timedelta(hours=23),
            1,  # Only 23h lag
            id="23h_horizon_only_23h_lag",
        ),
        pytest.param(
            timedelta(hours=24),
            0,  # No minute lags >= 24h
            id="24h_horizon_no_minute_lags",
        ),
    ],
)
def test_generate_minute_lags_filters_by_horizon(max_horizon: timedelta, expected_lags: int):
    # Arrange
    # (parameters provided)

    # Act
    result = generate_minute_lags(max_horizon)

    # Assert
    assert len(result) == expected_lags
    # All lags must meet the horizon requirement (>= max_horizon)
    assert all(lag >= max_horizon for lag in result)


@pytest.mark.parametrize(
    ("max_horizon", "max_day_lags", "expected_lags"),
    [
        pytest.param(
            timedelta(hours=24),
            5,
            [timedelta(days=i) for i in range(1, 6)],  # 1-5 days
            id="24h_horizon_generates_1_to_5_days",
        ),
        pytest.param(
            timedelta(hours=47),
            5,
            [timedelta(days=i) for i in range(2, 6)],  # 2-5 days (min_days=2)
            id="47h_horizon_starts_at_2_days",
        ),
        pytest.param(
            timedelta(days=2),
            5,
            [timedelta(days=i) for i in range(2, 6)],  # 2-5 days (exactly 2 days)
            id="exactly_2_days_horizon_starts_at_2_days",
        ),
        pytest.param(
            timedelta(hours=24),
            1,
            [timedelta(days=1)],  # Only 1 day
            id="max_day_lags_1_returns_single_lag",
        ),
        pytest.param(
            timedelta(days=10),
            5,
            [],  # min_days=10 > max_day_lags=5, no lags
            id="min_days_exceeds_max_returns_empty",
        ),
    ],
)
def test_generate_day_lags_respects_horizon_and_max(
    max_horizon: timedelta, max_day_lags: int, expected_lags: list[timedelta]
):
    # Arrange
    # (parameters provided)

    # Act
    result = generate_day_lags(max_horizon, max_day_lags)

    # Assert
    assert result == expected_lags


def _create_periodic_signal(num_samples: int = 600) -> pd.Series:
    # Create a periodic signal for autocorrelation testing.
    t = np.arange(num_samples)
    return pd.Series(
        np.sin(2 * np.pi * t / 4) + np.sin(2 * np.pi * t / 8),  # 1h + 2h periods
        index=pd.date_range("2025-01-01", periods=num_samples, freq="15min"),
    )


@pytest.mark.parametrize(
    ("max_horizon", "expected_lags"),
    [
        pytest.param(
            timedelta(minutes=15),
            [
                timedelta(minutes=210),
                timedelta(minutes=150),
                timedelta(hours=2),
                timedelta(minutes=90),
                timedelta(minutes=30),
            ],
            id="short_horizon_finds_multiple_periodic_lags",
        ),
        pytest.param(
            timedelta(minutes=90),
            [timedelta(minutes=210), timedelta(minutes=150), timedelta(hours=2), timedelta(minutes=90)],
            id="longer_horizon_filters_out_short_lags",
        ),
    ],
)
def test_generate_autocorr_lags_finds_expected_peaks(max_horizon: timedelta, expected_lags: list[timedelta]):
    """Test autocorrelation with known periodic signal produces expected lags.

    Signal combines 1h and 2h periods. Autocorrelation detects peaks at these periods
    and their harmonics: 30min (subharmonic), 90min (1.5h), 120min (2h), 150min (2.5h), 210min (3.5h).
    """
    # Arrange - create signal with 1-hour and 2-hour periodic components
    # 4 samples/hour * 150 hours = 600 samples (well above 100 minimum)
    signal = _create_periodic_signal(num_samples=600)

    # Act
    result = generate_autocorr_lags(signal=signal, max_horizon=max_horizon)

    # Assert - should find expected lags based on signal periodicity
    assert result == expected_lags


@pytest.mark.parametrize(
    ("signal", "max_horizon"),
    [
        pytest.param(
            pd.Series(np.random.default_rng(42).standard_normal(50)),
            timedelta(minutes=15),
            id="insufficient_data_returns_empty",
        ),
        pytest.param(
            pd.Series(np.ones(200)),
            timedelta(minutes=15),
            id="constant_signal_returns_empty",
        ),
    ],
)
def test_generate_autocorr_lags_returns_empty_for_edge_cases(signal: pd.Series, max_horizon: timedelta):
    # Arrange
    # (parameters provided)

    # Act
    result = generate_autocorr_lags(signal, max_horizon)

    # Assert - should return empty list for edge cases
    assert result == []


@pytest.mark.parametrize(
    ("adder", "expected_lags"),
    [
        pytest.param(
            LagsAdder(
                history_available=timedelta(hours=48),
                horizons=[LeadTime(value=timedelta(hours=1))],
                add_trivial_lags=True,
                custom_lags=[timedelta(hours=25)],
            ),
            # Trivial (1h-23h + 1d-14d) + 25h custom, sorted descending
            # 25h = 1 day + 1 hour, sorts between 2d and 1d
            sorted(
                [timedelta(hours=h) for h in range(1, 24)]
                + [timedelta(days=d) for d in range(1, 15)]
                + [timedelta(hours=25)],
                reverse=True,
            ),
            id="trivial_plus_custom",
        ),
        pytest.param(
            LagsAdder(
                history_available=timedelta(hours=48),
                horizons=[LeadTime(value=timedelta(hours=1))],
                add_trivial_lags=True,
                custom_lags=[timedelta(hours=1), timedelta(hours=2)],
            ),
            # Deduplication: 1h and 2h already in trivial lags
            sorted(
                [timedelta(hours=h) for h in range(1, 24)] + [timedelta(days=d) for d in range(1, 15)],
                reverse=True,
            ),
            id="deduplication",
        ),
    ],
)
def test_lags_adder__lags(adder: LagsAdder, expected_lags: list[timedelta]):
    """Test that LagsAdder correctly infers lags based on configuration."""
    # Assert - verify lags match expected (sorted descending, no duplicates)
    assert adder.lags == expected_lags
    assert len(adder.lags) == len(set(adder.lags)), "Lags should not contain duplicates"


def test_lags_adder__lags_autocorr():
    """Test that autocorrelation lags are added after calling fit()."""
    # Arrange
    adder = LagsAdder(
        history_available=timedelta(hours=48),
        horizons=[LeadTime(value=timedelta(minutes=15))],
        add_trivial_lags=False,
        add_autocorr_lags=True,
    )

    # Verify initial lags are empty (no trivial lags)
    assert len(adder.lags) == 0

    # Create dataset with periodic signal for autocorrelation
    signal = _create_periodic_signal(num_samples=600)
    dataset = create_timeseries_dataset(
        index=pd.DatetimeIndex(signal.index),
        load=signal,
        sample_interval=timedelta(minutes=15),
    )

    # Act
    adder.fit(dataset)

    # Assert - autocorr lags should be added
    assert len(adder.lags) > 0, "Autocorr lags should be added after fit"
    # Verify all lags meet the min horizon requirement
    assert all(lag >= timedelta(minutes=15) for lag in adder.lags)


def test_lags_adder__horizon_lags():
    """Test that horizon_lags properly filters lags based on horizon and history_available."""
    # Arrange
    horizons = [LeadTime(value=timedelta(hours=1)), LeadTime(value=timedelta(hours=2))]
    history_available = timedelta(hours=24)
    custom_lags = [
        timedelta(minutes=30),
        timedelta(hours=1),
        timedelta(hours=2),
        timedelta(hours=3),
        timedelta(hours=25),  # Exceeds history_available
    ]

    # Act
    adder = LagsAdder(
        history_available=history_available,
        horizons=horizons,
        add_trivial_lags=False,
        custom_lags=custom_lags,
        add_autocorr_lags=False,
    )

    # Assert
    # 1h horizon: lags >= 1h and <= 24h → [3h, 2h, 1h] (30min filtered out, 25h filtered out)
    assert adder.horizon_lags[LeadTime(value=timedelta(hours=1))] == [
        timedelta(hours=3),
        timedelta(hours=2),
        timedelta(hours=1),
    ]

    # 2h horizon: lags >= 2h and <= 24h → [3h, 2h] (30min, 1h, 25h filtered out)
    assert adder.horizon_lags[LeadTime(value=timedelta(hours=2))] == [
        timedelta(hours=3),
        timedelta(hours=2),
    ]


def test_lags_adder__transform_non_versioned():
    """Test that transform correctly shifts lag values for non-versioned dataset."""
    # Arrange
    dataset = create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1D"),
        load=[10, 20, 30, 40, 50],
        sample_interval=timedelta(days=1),
    )
    custom_lags = [timedelta(days=1), timedelta(days=2)]

    adder = LagsAdder(
        history_available=timedelta(days=10),
        horizons=[LeadTime(value=timedelta(days=1))],
        add_trivial_lags=False,
        custom_lags=custom_lags,
    )

    # Act
    transformed = adder.transform(dataset)

    # Assert
    expected_lag_columns = pd.DataFrame(
        {
            "load_lag_P1D": [np.nan, 10.0, 20.0, 30.0, 40.0],
            "load_lag_P2D": [np.nan, np.nan, 10.0, 20.0, 30.0],
        },
        index=dataset.index,
    )
    actual_lag_columns = transformed.data[expected_lag_columns.columns]
    pd.testing.assert_frame_equal(actual_lag_columns, expected_lag_columns)


def test_lags_adder__transform_versioned():
    """Test that transform correctly shifts lag values for versioned dataset with horizon-aware filtering."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=3, freq="1D", name="timestamp")
    dataset = create_timeseries_dataset(
        index=pd.DatetimeIndex(list(index) * 2),
        horizons=[timedelta(days=1)] * 3 + [timedelta(days=2)] * 3,
        load=[10, 20, 30, 10, 20, 30],
        sample_interval=timedelta(days=1),
    )
    custom_lags = [timedelta(days=1), timedelta(days=2)]

    adder = LagsAdder(
        history_available=timedelta(days=10),
        horizons=[LeadTime(value=timedelta(days=1)), LeadTime(value=timedelta(days=2))],
        add_trivial_lags=False,
        custom_lags=custom_lags,
    )

    # Act
    transformed = adder.transform(dataset)

    # Assert - check 1-day horizon data
    horizon_1d = transformed.select_horizon(LeadTime(value=timedelta(days=1)))
    expected_1d = pd.DataFrame(
        {
            "load_lag_P1D": [np.nan, 10.0, 20.0],
            "load_lag_P2D": [np.nan, np.nan, 10.0],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(horizon_1d.data[expected_1d.columns], expected_1d, check_freq=False)

    # Assert - check 2-day horizon data
    horizon_2d = transformed.select_horizon(LeadTime(value=timedelta(days=2)))
    expected_2d = pd.DataFrame(
        {
            "load_lag_P1D": [np.nan, np.nan, np.nan],  # 1-day lag < 2-day horizon
            "load_lag_P2D": [np.nan, np.nan, 10.0],  # 2-day lag at 03rd looks back to 01st
        },
        index=index,
    )
    pd.testing.assert_frame_equal(horizon_2d.data[expected_2d.columns], expected_2d, check_freq=False)
