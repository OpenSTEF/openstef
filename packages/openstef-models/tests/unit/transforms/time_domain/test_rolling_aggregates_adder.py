# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import LeadTime
from openstef_models.transforms.time_domain import RollingAggregatesAdder


def test_rolling_aggregate_features_basic():
    """Test basic rolling aggregation with simple data."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=5, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        rolling_window_size=timedelta(hours=2),  # 2-hour window
        aggregation_functions=["mean", "max", "min"],
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert result.sample_interval == dataset.sample_interval

    # Check columns exist
    assert "rolling_mean_load_PT2H" in result.data.columns
    assert "rolling_max_load_PT2H" in result.data.columns
    assert "rolling_min_load_PT2H" in result.data.columns

    # Check values (2-hour rolling window)
    # Index 0: [1] -> mean=1, max=1, min=1
    # Index 1: [1,2] -> mean=1.5, max=2, min=1
    # Index 2: [2,3] -> mean=2.5, max=3, min=2
    # Index 3: [3,4] -> mean=3.5, max=4, min=3
    # Index 4: [4,5] -> mean=4.5, max=5, min=4
    expected_mean = [1.0, 1.5, 2.5, 3.5, 4.5]
    expected_max = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_min = [1.0, 1.0, 2.0, 3.0, 4.0]

    assert result.data["rolling_mean_load_PT2H"].tolist() == expected_mean
    assert result.data["rolling_max_load_PT2H"].tolist() == expected_max
    assert result.data["rolling_min_load_PT2H"].tolist() == expected_min


def test_rolling_aggregate_features_with_nan():
    """Test rolling aggregation handles NaN values correctly."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, np.nan, 3.0, 4.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=4, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean"],
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # Index 0: [1] -> mean=1.0
    # Index 1: [1, NaN] -> mean=1.0 (pandas ignores NaN)
    # Index 2: [NaN, 3] -> mean=3.0 (pandas ignores NaN)
    # Index 3: [3, 4] -> mean=3.5
    expected_mean = [1.0, 1.0, 3.0, 3.5]

    assert result.data["rolling_mean_load_PT2H"].tolist() == expected_mean


def test_rolling_aggregate_features_missing_column_raises_error():
    """Test that transform raises error when required column is missing."""
    # Arrange
    data = pd.DataFrame(
        {"not_load": [1.0, 2.0, 3.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(minutes=15))
    transform = RollingAggregatesAdder(
        feature="load",
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act & Assert
    with pytest.raises(MissingColumnsError, match="Missing required columns"):
        transform.fit(dataset)

    with pytest.raises(MissingColumnsError, match="Missing required columns"):
        transform.transform(dataset)


def test_rolling_aggregate_features_empty_feature_on_fit():
    """Test that transform applies fallback strategy when feature is fully missing during inference."""
    # Arrange
    train_data = pd.DataFrame(
        {"load": [np.nan, np.nan, np.nan]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean"],
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(train_dataset)
    result = transform.transform(train_dataset)

    # Assert
    assert "rolling_mean_load_PT2H" in result.data.columns
    assert result.data["rolling_mean_load_PT2H"].isna().all()


def test_rolling_aggregate_features_partial_missing_during_inference():
    """Test that transform computes fresh aggregates when recent data is available."""
    # Arrange - training data
    train_data = pd.DataFrame(
        {"load": [10.0, 20.0, 30.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, sample_interval=timedelta(hours=1))

    # Inference data: some recent values available, then NaN for forecast horizon
    test_data = pd.DataFrame(
        {"load": [40.0, 50.0, np.nan, np.nan]},
        index=pd.date_range("2023-01-01 03:00:00", periods=4, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean", "max"],
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(train_dataset)
    result = transform.transform(test_dataset)

    # Assert
    assert not result.data["rolling_mean_load_PT2H"].isna().any()
    assert not result.data["rolling_max_load_PT2H"].isna().any()

    # First row: only 40 in window → mean=40, max=40
    assert result.data["rolling_mean_load_PT2H"].iloc[0] == 40.0
    assert result.data["rolling_max_load_PT2H"].iloc[0] == 40.0

    # Second row: [40, 50] in window → mean=45, max=50
    assert result.data["rolling_mean_load_PT2H"].iloc[1] == 45.0
    assert result.data["rolling_max_load_PT2H"].iloc[1] == 50.0

    # Third and fourth rows: NaN target, forward-fill from last computed
    assert result.data["rolling_mean_load_PT2H"].iloc[2] == 45.0
    assert result.data["rolling_max_load_PT2H"].iloc[2] == 50.0
    assert result.data["rolling_mean_load_PT2H"].iloc[3] == 45.0
    assert result.data["rolling_max_load_PT2H"].iloc[3] == 50.0


def test_rolling_aggregate_fallback_uses_last_valid_from_training():
    """Test fallback uses last valid aggregate from training when inference data is all NaN."""
    # Arrange
    train_data = pd.DataFrame(
        {"load": [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=5, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, sample_interval=timedelta(hours=1))

    # Inference data with no valid target values
    test_data = pd.DataFrame(
        {"load": [np.nan, np.nan, np.nan]},
        index=pd.date_range("2023-01-01 03:00:00", periods=3, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        rolling_window_size=timedelta(hours=2),
        aggregation_functions=["mean", "max"],
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(train_dataset)
    result = transform.transform(test_dataset)

    # Assert - all values filled with last valid aggregate from training
    # Last valid from training: mean of [40, 50] = 45.0, max = 50.0
    assert "rolling_mean_load_PT2H" in result.data.columns
    assert "rolling_max_load_PT2H" in result.data.columns
    assert not result.data["rolling_mean_load_PT2H"].isna().any()
    assert not result.data["rolling_max_load_PT2H"].isna().any()

    for i in range(3):
        assert result.data["rolling_mean_load_PT2H"].iloc[i] == 45.0
        assert result.data["rolling_max_load_PT2H"].iloc[i] == 50.0


def test_rolling_aggregate_features_default_parameters():
    """Test transform works with default parameters."""
    # Arrange
    data = pd.DataFrame(
        {"load": [1.0, 2.0, 3.0]},
        index=pd.date_range("2023-01-01 00:00:00", periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))

    transform = RollingAggregatesAdder(
        feature="load",
        horizons=[LeadTime.from_string("PT36H")],
    )

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert - default is 24-hour window with median, min, max
    assert "rolling_median_load_P1D" in result.data.columns
    assert "rolling_min_load_P1D" in result.data.columns
    assert "rolling_max_load_P1D" in result.data.columns
