# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_beam.evaluation.models import Window


@pytest.mark.parametrize(
    ("window", "expected_string"),
    [
        pytest.param(
            Window(lag=timedelta(days=2), size=timedelta(days=7), stride=timedelta(days=1)),
            "(lag=P2D,size=P7D,stride=P1D)",
            id="standard_window",
        ),
        pytest.param(
            Window(lag=timedelta(hours=6), size=timedelta(hours=12), stride=timedelta(hours=3)),
            "(lag=PT6H,size=PT12H,stride=PT3H)",
            id="hourly_window",
        ),
        pytest.param(
            Window(lag=timedelta(days=3, hours=6), size=timedelta(days=14), stride=timedelta(days=1)),
            "(lag=P3DT6H,size=P14D,stride=P1D)",
            id="mixed_units",
        ),
    ],
)
def test_window_str(window: Window, expected_string: str):
    # Act
    result = str(window)

    # Assert
    assert result == expected_string


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(
            Window(lag=timedelta(days=1), size=timedelta(days=7), stride=timedelta(days=1)), id="daily_window"
        ),
        pytest.param(
            Window(lag=timedelta(hours=6), size=timedelta(hours=24), stride=timedelta(hours=6)), id="hourly_window"
        ),
        pytest.param(
            Window(lag=timedelta(days=3, hours=12), size=timedelta(days=14), stride=timedelta(days=2)),
            id="complex_window",
        ),
        pytest.param(
            Window(lag=timedelta(minutes=30), size=timedelta(hours=2), stride=timedelta(minutes=15)),
            id="fine_grained_window",
        ),
    ],
)
def test_window_from_string_roundtrip(window: Window):
    # Arrange
    original = window

    # Act
    str_repr = str(original)
    reconstructed = Window.from_string(str_repr)

    # Assert
    assert reconstructed.lag == original.lag
    assert reconstructed.size == original.size
    assert reconstructed.stride == original.stride
    # Note: minimum_coverage is not included in string representation,
    # so it will be the default value in the reconstructed object
