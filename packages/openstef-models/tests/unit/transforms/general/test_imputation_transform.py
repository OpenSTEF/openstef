# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_models.transforms.general import ImputationTransform, RemoveEmptyColumnsTransform


@pytest.fixture
def sample_dataset() -> TimeSeriesDataset:
    """Create a sample dataset with missing values for testing.

    Returns:
        A TimeSeriesDataset with 3 features: radiation, temperature, wind_speed (with some NaN),
        spanning 4 hours. Designed to have valid rows after trailing null removal.
    """
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 110.0, 120.0],
            "temperature": [20.0, 21.0, np.nan, 23.0],
            "wind_speed": [5.0, 6.0, 8.0, 9.0],  # No missing values
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_validation_constant_strategy_requires_fill_value():
    """Test that CONSTANT strategy raises ValidationError when fill_value is missing."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="fill_value must be provided when imputation_strategy is CONSTANT"):
        ImputationTransform(imputation_strategy="constant")


def test_basic_imputation_works(sample_dataset: TimeSeriesDataset):
    """Test that basic imputation removes NaN values."""
    # Arrange
    transform = ImputationTransform(imputation_strategy="mean")

    # Act
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Assert
    # Non-trailing NaN values should be imputed
    assert not result.data.isna().any().any()


def test_constant_imputation_uses_fill_value():
    """Test that constant strategy uses the specified fill_value."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [100.0, np.nan, 120.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = ImputationTransform(imputation_strategy="constant", fill_value=999.0)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # The middle NaN should be replaced with fill_value
    assert result.data.loc[result.data.index[1], "radiation"] == 999.0


def test_custom_missing_value_placeholder():
    """Test that custom missing value placeholders are handled correctly."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [100.0, -999.0, 120.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = ImputationTransform(imputation_strategy="mean", missing_value=-999.0)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # Custom missing values should be imputed
    assert not (result.data == -999.0).any().any()
    assert not result.data.isna().any().any()


@pytest.mark.parametrize(
    ("columns", "expected_rows"),
    [
        pytest.param({"radiation"}, 4, id="single_feature"),
        pytest.param({"wind_speed"}, 4, id="feature_with_middle_nan"),
        pytest.param({"nonexistent"}, 4, id="nonexistent_feature"),
    ],
)
def test_trailing_null_preservation(
    columns: set[str],
    expected_rows: int,
):
    """Test that trailing NaNs are preserved after imputation."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0, np.nan],  # Trailing NaN
            "wind_speed": [5.0, np.nan, 7.0, 8.0],  # Middle NaN
            "temperature": [20.0, 21.0, 22.0, 23.0],  # No missing values
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = ImputationTransform(imputation_strategy="mean", columns=columns)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == expected_rows

    # Check behavior for selected columns
    for col in columns.intersection(result.data.columns):
        if col == "radiation":
            # Radiation: [100, 110, 120, NaN] - trailing NaN should remain
            assert pd.isna(result.data.loc[result.data.index[-1], col])
        elif col == "wind_speed":
            # Wind_speed: [5, NaN, 7, 8] - middle NaN imputed, no trailing NaN
            assert not pd.isna(result.data.loc[result.data.index[1], col])  # Middle imputed
            assert not pd.isna(result.data.loc[result.data.index[-1], col])  # No trailing NaN


def test_empty_columns_raise_error():
    """Test that completely empty columns raise a helpful error."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 110.0],
            "empty_feature": [np.nan, np.nan, np.nan],  # All missing
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = ImputationTransform(imputation_strategy="mean")

    # Act & Assert
    with pytest.raises(ValueError, match=r"Cannot impute completely empty columns.*Use RemoveEmptyColumnsTransform"):
        transform.fit(dataset)


def test_remove_empty_then_impute_workflow():
    """Test the recommended workflow: remove empty columns first, then impute."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 110.0],
            "temperature": [20.0, np.nan, 22.0],
            "empty_feature": [np.nan, np.nan, np.nan],  # All missing
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    # Act
    # First remove empty columns
    remove_transform = RemoveEmptyColumnsTransform()
    remove_transform.fit(dataset)
    cleaned_dataset = remove_transform.transform(dataset)

    # Then apply imputation
    impute_transform = ImputationTransform(imputation_strategy="mean")
    impute_transform.fit(cleaned_dataset)
    result = impute_transform.transform(cleaned_dataset)

    # Assert
    # Empty column should be removed
    assert "empty_feature" not in result.data.columns
    assert set(result.data.columns) == {"radiation", "temperature"}
    # Remaining NaNs should be imputed
    assert not result.data.isna().any().any()


def test_transform_not_fitted_error():
    """Test that TransformNotFittedError is raised when transform is called before fit."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [100.0, 110.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=2, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = ImputationTransform(imputation_strategy="mean")

    # Act & Assert
    with pytest.raises(TransformNotFittedError, match="The transform 'ImputationTransform' has not been fitted"):
        transform.transform(dataset)


def test_no_missing_values_data_preservation():
    """Test that data without missing values is preserved exactly."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, 110.0, 120.0],
            "temperature": [20.0, 21.0, 23.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=3, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    original_data = dataset.data.copy()
    transform = ImputationTransform(imputation_strategy="mean")

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # Data should be identical to original
    pd.testing.assert_frame_equal(result.data, original_data)
    # Sample interval should be preserved
    assert result.sample_interval == dataset.sample_interval
