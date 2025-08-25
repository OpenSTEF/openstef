from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.forecasting_transforms.trend_features import (
    AggregationFunction,
    RollingAggregateFeaturesConfig, RollingAggregateFeatures,
)


@pytest.fixture()
def rolling_aggregate_config():
    return RollingAggregateFeaturesConfig(
        rolling_window_size=timedelta(hours=24),
        aggregation_functions=[
            AggregationFunction.MEDIAN,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
        ],
    )


@pytest.mark.parametrize("rolling_window", [timedelta(days=1), timedelta(hours=24)])
def test_rolling_aggregate_features(rolling_window: timedelta):
    # Arrange
    num_points = int(24 * 60 / 15 * 2)  # 2 days of data at 15-minute intervals
    data = pd.DataFrame(
        data={
            "load": list(range(num_points))
        },
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))

    config = RollingAggregateFeaturesConfig(
        rolling_window_size=rolling_window,
        aggregation_functions=[
            AggregationFunction.MEDIAN,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
        ],
    )

    transform = RollingAggregateFeatures(config=config)

    # Act
    transformed_dataset = transform.fit_transform(dataset)
    output_data = transformed_dataset.data

    #Assert
    assert transformed_dataset.sample_interval == dataset.sample_interval
    # Verify the columns are created
    assert "rolling_median_load_P1D" in output_data.columns
    assert "rolling_max_load_P1D" in output_data.columns
    assert "rolling_min_load_P1D" in output_data.columns

    # Validate the rolling features
    rolling_window_load = data["load"].rolling(window=rolling_window)
    rolling_median_expected = rolling_window_load.median()
    rolling_max_expected = rolling_window_load.max()
    rolling_min_expected = rolling_window_load.min()

    assert np.allclose(output_data["rolling_median_load_P1D"], rolling_median_expected)
    assert np.allclose(output_data["rolling_max_load_P1D"], rolling_max_expected)
    assert np.allclose(output_data["rolling_min_load_P1D"], rolling_min_expected)


def test_rolling_aggregate_features_flatline():
    # Arrange
    num_points = int(24 * 60 / 15 * 2)  # 2 days of data at 15-minute intervals
    all_ones = [1.0] * num_points
    data = pd.DataFrame(
        data={"load": all_ones},
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))

    config = RollingAggregateFeaturesConfig(
        rolling_window_size=timedelta(hours=24),
        aggregation_functions=[
            AggregationFunction.MEDIAN,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
        ],
    )

    transform = RollingAggregateFeatures(config=config)

    # Act
    transformed_dataset = transform.fit_transform(dataset)
    output_data = transformed_dataset.data

    # Verify the columns are created
    assert "rolling_median_load_P1D" in output_data.columns
    assert "rolling_max_load_P1D" in output_data.columns
    assert "rolling_min_load_P1D" in output_data.columns

    # Validate the rolling features
    assert np.all(output_data[f"rolling_median_load_P1D"] == all_ones)
    assert np.all(output_data[f"rolling_max_load_P1D"] == all_ones)
    assert np.all(output_data[f"rolling_min_load_P1D"] == all_ones)


def test_rolling_aggregate_features_nans():
    # Arrange
    # Generate data with NaNs in middle and at the end
    load = [1, 2, np.nan, 4, 5, 6, 7, 8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    num_points = len(load)
    data = pd.DataFrame(
        data={"load": load},
        index=pd.date_range(
            start="2023-01-01 00:00:00", freq="15min", periods=num_points
        )
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))

    config = RollingAggregateFeaturesConfig(
        rolling_window_size=timedelta(hours=1),
        aggregation_functions=[
            AggregationFunction.MEDIAN,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
        ],
    )

    transform = RollingAggregateFeatures(config=config)

    # Act
    transformed_dataset = transform.fit_transform(dataset)
    output_data = transformed_dataset.data

    # Verify the columns are created
    assert "rolling_median_load_PT1H" in output_data.columns
    assert "rolling_max_load_PT1H" in output_data.columns
    assert "rolling_min_load_PT1H" in output_data.columns

    # Validate the rolling features
    assert np.allclose(
        output_data["rolling_median_load_PT1H"],
        [1, 1.5, 1.5, 2, 4, 5, 5.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5],
    )
    assert np.allclose(
        output_data["rolling_max_load_PT1H"], [1, 2, 2, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8]
    )
    assert np.allclose(
        output_data["rolling_min_load_PT1H"], [1, 1, 1, 1, 2, 4, 4, 5, 5, 5, 5, 5, 5, 5]
    )

def test_rolling_aggregate_features_no_load_column():
    # Test for dataframe without load column
    data = pd.DataFrame(
        index=pd.date_range(start="2023-01-01 00:00:00", freq="15min", periods=10),
        columns=["not_load"],
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))

    with pytest.raises(ValueError, match="The DataFrame must contain a 'load' column."):
        RollingAggregateFeatures(config=RollingAggregateFeaturesConfig()).fit(dataset)