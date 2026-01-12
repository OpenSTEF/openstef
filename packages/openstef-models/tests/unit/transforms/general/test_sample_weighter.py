# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.testing import create_timeseries_dataset
from openstef_models.transforms.general.sample_weighter import (
    SampleWeighter,
    exponential_sample_weight,
)


@pytest.mark.parametrize(
    ("exponent", "expected_weights"),
    [
        pytest.param(
            0.0,
            [1.0, 1.0, 1.0, 1.0, 1.0],  # exponent=0: all values become 1.0 (uniform)
            id="exponent_0_uniform_weights",
        ),
        pytest.param(
            1.0,
            [0.1, 0.25, 0.5, 0.75, 1.0],  # exponent=1: linear scaling
            id="exponent_1_linear_weights",
        ),
        pytest.param(
            2.0,
            [0.1, 0.1, 0.25, 0.5625, 1.0],  # exponent=2: 0.1^2=0.01, 0.25^2=0.0625 clipped to floor
            id="exponent_2_quadratic_weights",
        ),
    ],
)
def test_exponential_sample_weight__exponents(exponent: float, expected_weights: list[float]):
    """Test exponential_sample_weight with different exponent values.

    Uses simple values [10, 25, 50, 75, 100] with percentile=100 for easy calculation.
    With percentile=100, scale value is 100, so:
    - 10/100=0.1, 25/100=0.25, 50/100=0.5, 75/100=0.75, 100/100=1.0
    - With floor=0.1, values below 0.1 are clipped to 0.1
    """
    # Arrange
    x = np.array([10.0, 25.0, 50.0, 75.0, 100.0])

    # Act
    result = exponential_sample_weight(
        x=x,
        scale_percentile=100,
        exponent=exponent,
        floor=0.1,
    )

    # Assert
    np.testing.assert_allclose(result, expected_weights, rtol=1e-10)


def test_sample_weighter__fit_transform():
    """Test that SampleWeighter correctly computes sample weights based on target values."""
    # Arrange
    dataset = create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        load=[10.0, 50.0, 100.0, 200.0, 150.0],
        sample_interval=timedelta(hours=1),
    )

    transform = SampleWeighter(
        weight_scale_percentile=95,
        weight_exponent=1.0,
        weight_floor=0.1,
        target_column="load",
        normalize_target=True,
    )

    # Act
    result = transform.fit_transform(dataset)

    # Assert - verify sample_weight column was added
    assert "sample_weight" in result.data.columns

    # Expected weights calculation:
    # With scale_percentile=95, the 95th percentile of [10, 50, 100, 200, 150] is 190
    # Each value is scaled by dividing by 190, raised to exponent 1.0, and clipped to [0.1, 1.0]
    expected_weights = pd.Series(
        data=[0.9504, 0.5372, 0.1, 1.0, 0.495868],
        index=dataset.index,
        name="sample_weight",
    )

    pd.testing.assert_series_equal(
        result.data["sample_weight"],
        expected_weights,
        atol=0.001,
    )


def test_sample_weighter__no_target_column():
    """Test that SampleWeighter raises error when target column is missing."""
    # Arrange
    dataset = create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        load=[10.0, 50.0, 100.0, 200.0, 150.0],
        sample_interval=timedelta(hours=1),
    )

    transform = SampleWeighter(
        target_column="non_existent_column",
    )

    # Act
    transform.fit_transform(dataset)

    # Assert
    assert not transform.is_fitted
    result = transform.transform(dataset)
    assert "sample_weight" not in result.data.columns


def test_sample_weighter__transform_all_nan_target():
    """Test that SampleWeighter transform handles all nan target column gracefully."""
    # Arrange
    train_dataset = dataset = create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        load=[10.0, 50.0, 100.0, 200.0, 150.0],
        sample_interval=timedelta(hours=1),
    )
    predict_dataset = create_timeseries_dataset(
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        load=[np.nan] * 5,
        sample_interval=timedelta(hours=1),
    )

    transform = SampleWeighter(
        weight_scale_percentile=95,
        weight_exponent=1.0,
        weight_floor=0.1,
        target_column="load",
        normalize_target=True,
    )
    transform.fit(train_dataset)

    # Act
    result = transform.transform(predict_dataset)

    # Assert
    assert "sample_weight" in result.data.columns

    expected_weights = pd.Series(
        data=[1.0] * 5,
        index=dataset.index,
        name="sample_weight",
    )

    pd.testing.assert_series_equal(
        result.data["sample_weight"],
        expected_weights,
        atol=0.001,
    )
