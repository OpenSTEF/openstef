# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the CyclicFeatures transform."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.feature_engineering.temporal_transforms.cyclic_features import CyclicFeatures


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """
    Create a sample TimeSeriesDataset for testing.

    Returns:
        TimeSeriesDataset: A dataset with a 6-hour frequency and sample data.
    """
    data = pd.DataFrame({"load": [100.0, 110.0, 120.0, 130.0]}, index=pd.date_range("2025-01-01", periods=4, freq="6h"))
    return TimeSeriesDataset(data, timedelta(hours=6))


def test_cyclic_features_initialization():
    """Test CyclicFeatures can be initialized properly."""
    transform = CyclicFeatures()
    assert hasattr(transform, "cyclic_features")
    assert transform.cyclic_features.empty


def test_sine_cosine_computation():
    """Test sine and cosine computation with basic trigonometric values."""
    # Test with degrees converted to radians for exact values
    phase = pd.Index([0, 90, 180, 270])
    period = 360

    sine_result = CyclicFeatures._compute_sine(phase, period)
    cosine_result = CyclicFeatures._compute_cosine(phase, period)

    # Check sine values
    expected_sine = np.array([0.0, 1.0, 0.0, -1.0])
    np.testing.assert_array_almost_equal(sine_result, expected_sine, decimal=10)

    # Check cosine values
    expected_cosine = np.array([1.0, 0.0, -1.0, 0.0])
    np.testing.assert_array_almost_equal(cosine_result, expected_cosine, decimal=10)


def test_fit_creates_all_features(sample_dataset: TimeSeriesDataset):
    """Test that fit creates all expected cyclic features."""
    transform = CyclicFeatures()
    transform.fit(sample_dataset)

    expected_columns = [
        "season_sine",
        "season_cosine",
        "day0fweek_sine",
        "day0fweek_cosine",
        "month_sine",
        "month_cosine",
    ]

    assert not transform.cyclic_features.empty
    assert list(transform.cyclic_features.columns) == expected_columns
    assert len(transform.cyclic_features) == len(sample_dataset.index)


def test_transform_adds_features(sample_dataset: TimeSeriesDataset):
    """Test that transform adds cyclic features to the dataset."""
    transform = CyclicFeatures()
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Check structure
    assert isinstance(result, TimeSeriesDataset)
    assert len(result.feature_names) == len(sample_dataset.feature_names) + 6
    assert result.sample_interval == sample_dataset.sample_interval

    # Check that original features are preserved
    for feature in sample_dataset.feature_names:
        assert feature in result.feature_names
        pd.testing.assert_series_equal(result.data[feature], sample_dataset.data[feature])


def test_feature_value_ranges(sample_dataset: TimeSeriesDataset):
    """Test that all cyclic features are within expected [-1, 1] range."""
    transform = CyclicFeatures()
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    cyclic_columns = [
        "season_sine",
        "season_cosine",
        "day0fweek_sine",
        "day0fweek_cosine",
        "month_sine",
        "month_cosine",
    ]

    for col in cyclic_columns:
        values = result.data[col]
        assert values.min() >= -1.0, f"{col} has values below -1"
        assert values.max() <= 1.0, f"{col} has values above 1"
        assert not values.isna().any(), f"{col} has NaN values"


def test_empty_dataset():
    """Test handling of empty dataset."""
    data = pd.DataFrame(columns=["load"]).astype(float)
    data.index = pd.DatetimeIndex([])
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    transform = CyclicFeatures()
    transform.fit(dataset)
    result = transform.transform(dataset)

    assert len(result.data) == 0
    assert len(result.feature_names) == 7  # 1 original + 6 cyclic columns
