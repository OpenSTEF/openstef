# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef.enums import AggregateFunction
from openstef.feature_engineering.rolling_features import add_rolling_aggregate_features


@pytest.mark.parametrize("rolling_window", [timedelta(days=1), timedelta(hours=24)])
def test_add_rolling_aggregate_features(rolling_window):
    # Generate 2 days of data at 15-minute intervals
    num_points = int(24 * 60 / 15 * 2)
    data = pd.DataFrame(
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )
    data["load"] = list(range(num_points))

    pj = {
        "rolling_aggregate_features": [
            AggregateFunction.MEDIAN,
            AggregateFunction.MAX,
            AggregateFunction.MIN,
        ]
    }

    # Apply the function
    output_data = add_rolling_aggregate_features(
        data, pj=pj, rolling_window=rolling_window
    )

    # Verify the columns are created
    assert f"rolling_median_load_{rolling_window}" in output_data.columns
    assert f"rolling_max_load_{rolling_window}" in output_data.columns
    assert f"rolling_min_load_{rolling_window}" in output_data.columns

    # Validate the rolling features
    rolling_window_load = data["load"].rolling(window=rolling_window)
    rolling_median_expected = rolling_window_load.median()
    rolling_max_expected = rolling_window_load.max()
    rolling_min_expected = rolling_window_load.min()

    assert np.allclose(
        output_data[f"rolling_median_load_{rolling_window}"], rolling_median_expected
    )
    assert np.allclose(
        output_data[f"rolling_max_load_{rolling_window}"], rolling_max_expected
    )
    assert np.allclose(
        output_data[f"rolling_min_load_{rolling_window}"], rolling_min_expected
    )


def test_add_rolling_aggregate_features_flatline():
    # Generate 2 days of data at 15-minute intervals
    num_points = int(24 * 60 / 15 * 2)
    data = pd.DataFrame(
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )
    all_ones = [1.0] * num_points
    data["load"] = all_ones
    pj = {
        "rolling_aggregate_features": [
            AggregateFunction.MEDIAN,
            AggregateFunction.MAX,
            AggregateFunction.MIN,
        ]
    }

    # Apply the function
    output_data = add_rolling_aggregate_features(data, pj=pj)

    # Verify the columns are created
    assert "rolling_median_load_1 day, 0:00:00" in output_data.columns
    assert "rolling_max_load_1 day, 0:00:00" in output_data.columns
    assert "rolling_min_load_1 day, 0:00:00" in output_data.columns

    # Validate the rolling features
    assert np.all(output_data[f"rolling_median_load_1 day, 0:00:00"] == all_ones)
    assert np.all(output_data[f"rolling_max_load_1 day, 0:00:00"] == all_ones)
    assert np.all(output_data[f"rolling_min_load_1 day, 0:00:00"] == all_ones)


def test_add_rolling_aggregate_features_nans():
    # Generate 2 hours of data
    num_points = 8
    data = pd.DataFrame(
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )

    # Add NaNs to the data
    data["load"] = [1, 2, np.nan, 4, 5, 6, 7, 8]

    pj = {
        "rolling_aggregate_features": [
            AggregateFunction.MEDIAN,
            AggregateFunction.MAX,
            AggregateFunction.MIN,
        ]
    }

    # Apply the function
    output_data = add_rolling_aggregate_features(
        data, pj=pj, rolling_window=timedelta(hours=1)
    )

    # Verify the columns are created
    assert "rolling_median_load_1:00:00" in output_data.columns
    assert "rolling_max_load_1:00:00" in output_data.columns
    assert "rolling_min_load_1:00:00" in output_data.columns

    # Validate the rolling features
    assert np.allclose(
        output_data["rolling_median_load_1:00:00"], [1, 1.5, 1.5, 2, 4, 5, 5.5, 6.5]
    )
    assert np.allclose(
        output_data["rolling_max_load_1:00:00"], [1, 2, 2, 4, 5, 6, 7, 8]
    )
    assert np.allclose(
        output_data["rolling_min_load_1:00:00"], [1, 1, 1, 1, 2, 4, 4, 5]
    )


def test_add_rolling_aggregate_features_non_datetime_index():
    # Test for non-datetime index
    data = pd.DataFrame(index=range(10))

    with pytest.raises(
        ValueError, match="The DataFrame index must be a DatetimeIndex."
    ):
        add_rolling_aggregate_features(data, pj={})


def test_add_rolling_aggregate_features_no_load_column():
    # Test for dataframe without load column
    data = pd.DataFrame(
        index=pd.date_range(start="2023-01-01 00:00:00", freq="15min", periods=10),
        columns=["not_load"],
    )

    with pytest.raises(ValueError, match="The DataFrame must contain a 'load' column."):
        add_rolling_aggregate_features(data, pj={})
