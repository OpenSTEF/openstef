# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for Kalman Filter transforms."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.testing import create_timeseries_dataset
from openstef_models.transforms.general.kalman_filter import (
    KalmanPostprocessor,
    KalmanPreprocessor,
)
from openstef_models.utils.feature_selection import (
    FeatureSelection,
)


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Create a sample dataset for testing Kalman transforms."""
    return create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        load=[10.0, 50.0, 100.0, 200.0, 150.0],
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def sample_forecast_dataset() -> ForecastDataset:
    """Create a sample forecast dataset for testing KalmanPostprocessor."""
    forecast_data = pd.DataFrame(
        {
            "load": [100, np.nan],
            "quantile_P10": [90, 95],
            "quantile_P50": [100, 110],
            "quantile_P90": [115, 125],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    return ForecastDataset(forecast_data, timedelta(hours=1))


def test_kalman_preprocessor(sample_dataset: TimeSeriesDataset):
    """Test KalmanPreprocessor on sample dataset."""
    dataset = sample_dataset
    transform = KalmanPreprocessor()
    result = transform.fit_transform(dataset)

    expected_values = pd.DataFrame(
        [6.666667, 33.750000, 74.761905, 152.181818, 150.833333],
        index=dataset.data.index,
        columns=["load"],
    )
    pd.testing.assert_frame_equal(result.data, expected_values)


def test_kalman_preprocessor_shape(sample_dataset: TimeSeriesDataset):
    """Test that KalmanPreprocessor preserves dataset shape."""
    dataset = sample_dataset
    transform = KalmanPreprocessor()
    result = transform.fit_transform(dataset)

    assert result.data.shape == dataset.data.shape


def test_kalman_preprocessor_init_default():
    """KalmanPreprocessor can be instantiated and exposes expected callables."""
    transform = KalmanPreprocessor()
    assert hasattr(transform, "fit")
    assert hasattr(transform, "transform")
    assert hasattr(transform, "fit_transform")
    assert callable(transform.fit)
    assert callable(transform.transform)
    assert callable(transform.fit_transform)


def test_kalman_preprocessor_feature_selection_applies_only_selected_features(sample_dataset: TimeSeriesDataset):
    """KalmanPreprocessor should only transform the selected feature(s)."""
    dataset = sample_dataset
    # add an additional feature that has varying values
    dataset.data["other"] = pd.Series([5.0, 10.0, 5.0, 10.0, 5.0], index=dataset.data.index)

    original_load = dataset.data["load"].copy()
    original_other = dataset.data["other"].copy()

    # apply filter only to "load"
    transform = KalmanPreprocessor(selection=FeatureSelection(include={"load"}, exclude=None))
    result = transform.fit_transform(dataset)

    # "other" must remain unchanged
    pd.testing.assert_series_equal(result.data["other"], original_other)

    # "load" should be changed by the transform
    assert not result.data["load"].equals(original_load)


def test_kalman_postprocessor(sample_forecast_dataset: ForecastDataset):
    """Test KalmanPostprocessor on sample forecast dataset."""
    dataset = sample_forecast_dataset
    transform = KalmanPostprocessor()
    result = transform.fit_transform(dataset)

    expected_data = pd.DataFrame(
        {
            "load": [100.0, np.nan],
            "quantile_P10": [60.000, 81.875],
            "quantile_P50": [66.666667, 93.750000],
            "quantile_P90": [76.666667, 106.875000],
        },
        index=dataset.data.index,
    )

    pd.testing.assert_frame_equal(result.data, expected_data)


def test_kalman_postprocessor_shape(sample_forecast_dataset: ForecastDataset):
    """Test that KalmanPostprocessor preserves dataset shape."""
    dataset = sample_forecast_dataset
    transform = KalmanPostprocessor()
    result = transform.fit_transform(dataset)

    assert result.data.shape == dataset.data.shape


def test_kalman_postprocessor_init_default():
    """KalmanPostprocessor can be instantiated and exposes expected callables."""
    transform = KalmanPostprocessor()
    assert hasattr(transform, "fit")
    assert hasattr(transform, "transform")
    assert hasattr(transform, "fit_transform")
    assert callable(transform.fit)
    assert callable(transform.transform)
    assert callable(transform.fit_transform)


def test_kalman_postprocessor_feature_selection_applies_only_selected_features(
    sample_forecast_dataset: ForecastDataset,
):
    """KalmanPostprocessor should only transform the selected feature(s)."""
    dataset = sample_forecast_dataset
    # add an additional feature that has varying values
    dataset.data["other"] = pd.Series([1.0, 2.0], index=dataset.data.index)

    original_other = dataset.data["other"].copy()
    original_q50 = dataset.data["quantile_P50"].copy()
    # apply postprocessor only to "quantile_P50"
    transform = KalmanPostprocessor(selection=FeatureSelection(include={"quantile_P50"}, exclude=None))
    result = transform.fit_transform(dataset)

    # "other" must remain unchanged
    pd.testing.assert_series_equal(result.data["other"], original_other)

    # "quantile_P50" should be changed by the transform
    assert not result.data["quantile_P50"].equals(original_q50)
