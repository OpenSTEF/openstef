# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.general.dimensionality_reducer import DimensionalityReducer


@pytest.fixture
def sample_forecast_input_dataset() -> TimeSeriesDataset:
    """Create sample input dataset for forecaster training and prediction."""
    rng = np.random.default_rng(42)
    num_samples = 14
    start_date = datetime.fromisoformat("2025-01-01T00:00:00")

    feature_1 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_2 = rng.normal(loc=0, scale=1, size=num_samples)
    feature_3 = rng.uniform(low=-1, high=1, size=num_samples)

    return TimeSeriesDataset(
        data=pd.DataFrame(
            {
                "load": (feature_1 + feature_2 + feature_3) / 3,
                "feature1": feature_1,
                "feature2": feature_2,
                "feature3": feature_3,
            },
            index=pd.date_range(start=start_date, periods=num_samples, freq="1d"),
        ),
        sample_interval=timedelta(days=1),
    )


@pytest.mark.parametrize(
    ("method", "expected_components"),
    [
        ("pca", 2),
        ("factor_analysis", 2),
        ("fastica", 2),
        ("kernel_pca", 2),
    ],
)
def test_dimensionality_reduction(
    sample_forecast_input_dataset: TimeSeriesDataset,
    method: Literal["pca", "factor_analysis", "fastica"],
    expected_components: int,
) -> None:
    """Test dimensionality reduction with different methods."""
    # Arrange
    transform = DimensionalityReducer(
        columns=["feature1", "feature2", "feature3"], method=method, n_components=expected_components
    )
    assert not transform.is_fitted

    # Act
    transform.fit(sample_forecast_input_dataset)
    output = transform.transform(sample_forecast_input_dataset)

    # Assert
    expected_feature_count = expected_components + 1  # +1 for the 'load' column
    assert len(output.feature_names) == expected_feature_count
    assert transform.is_fitted
    assert len(output.data) == len(sample_forecast_input_dataset.data)
    assert "load" in output.feature_names

    # Check that the component columns are created with expected names
    component_columns = [col for col in output.feature_names if col.startswith("component_")]
    assert len(component_columns) == expected_components


@pytest.mark.parametrize(
    ("method", "method_params"),
    [
        ("pca", {}),
        ("factor_analysis", {"max_iter": 500}),
        ("fastica", {"max_iter": 1500}),
        ("kernel_pca", {"kernel": "poly", "gamma": 0.1}),
    ],
)
def test_dimensionality_reduction_with_custom_parameters(
    sample_forecast_input_dataset: TimeSeriesDataset,
    method: Literal["pca", "factor_analysis", "fastica", "kernel_pca"],
    method_params: dict[str, int],
) -> None:
    """Test dimensionality reduction with method-specific parameters."""
    # Arrange
    transform = DimensionalityReducer(
        columns=["feature1", "feature2", "feature3"], method=method, n_components=2, **method_params
    )

    # Act
    transform.fit(sample_forecast_input_dataset)
    output = transform.transform(sample_forecast_input_dataset)

    # Assert
    assert transform.is_fitted
    assert len(output.feature_names) == 3  # 2 components + 1 outcome variable
    assert "load" in output.feature_names

    # Verify method-specific parameters are set correctly
    if method in {"factor_analysis", "fastica"}:
        assert transform.max_iter == method_params.get("max_iter", 1000)


def test_dimensionality_reduction__state_roundtrip(sample_forecast_input_dataset: TimeSeriesDataset) -> None:
    """Test dimensionality reduction state round trip."""
    # Arrange
    original_transform = DimensionalityReducer(
        columns=["feature1", "feature2", "feature3"], method="pca", n_components=2
    )

    original_transform.fit(sample_forecast_input_dataset)

    state = original_transform.to_state()

    restored_transform = DimensionalityReducer()
    restored_transform = restored_transform.from_state(state)

    original_result = original_transform.transform(sample_forecast_input_dataset)
    restored_result = restored_transform.transform(sample_forecast_input_dataset)

    # Assert
    pd.testing.assert_frame_equal(original_result.data, restored_result.data)
