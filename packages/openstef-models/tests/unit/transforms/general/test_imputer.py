# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# iterative imputer is experimental. Defaults do not follow deprecation cycle.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 # type: ignore
from sklearn.linear_model import BayesianRidge

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import EmptyFeatureRemover, Imputer
from openstef_models.transforms.general.imputer import ImputationStrategy
from openstef_models.utils.feature_selection import FeatureSelection


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


@pytest.fixture
def correlated_dataset() -> TimeSeriesDataset:
    """Create sample dataset with simple linear relation between features where temperature = 2 * radiation.

    Returns:
        A TimeSeriesDataset with 2 features: radiation and temperature (with some NaN),
        spanning 5 hours.
    """
    data = pd.DataFrame(
        {
            "radiation": [10.0, np.nan, 30.0, 40.0, 50.0],
            "temperature": [20.0, 40.0, 60.0, np.nan, 100.0],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=5, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


def test_validation_constant_strategy_requires_fill_value():
    """Test that CONSTANT strategy raises ValidationError when fill_value is missing."""
    # Arrange & Act & Assert
    with pytest.raises(ValidationError, match="fill_value must be provided when imputation_strategy is CONSTANT"):
        Imputer(imputation_strategy="constant")


@pytest.mark.parametrize(
    "strategy",
    ["mean", "median", "most_frequent"],
)
def test_basic_imputation_works_simple_strategies(sample_dataset: TimeSeriesDataset, strategy: ImputationStrategy):
    """Test that basic imputation removes NaN values for simple strategies."""
    # Arrange
    transform = Imputer(imputation_strategy=strategy)

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
    transform = Imputer(imputation_strategy="constant", fill_value=999.0)

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
    transform = Imputer(imputation_strategy="mean", missing_value=-999.0)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # Custom missing values should be imputed
    assert not (result.data == -999.0).any().any()
    assert not result.data.isna().any().any()


@pytest.mark.parametrize(
    ("selection", "expected_rows"),
    [
        pytest.param(FeatureSelection(include={"radiation"}), 4, id="single_feature"),
        pytest.param(FeatureSelection(include={"wind_speed"}), 4, id="feature_with_middle_nan"),
        pytest.param(FeatureSelection(include={"nonexistent"}), 4, id="nonexistent_feature"),
    ],
)
def test_trailing_null_preservation(
    selection: FeatureSelection,
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
    transform = Imputer(imputation_strategy="mean", selection=selection)

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    assert len(result.data) == expected_rows

    # Check behavior for selected columns
    for col in selection.resolve(result.feature_names):
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
    transform = Imputer(imputation_strategy="mean")

    # Act & Assert
    with pytest.raises(ValueError, match=r"Cannot impute completely empty columns.*Use EmptyFeatureRemover"):
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
    remove_transform = EmptyFeatureRemover()
    remove_transform.fit(dataset)
    cleaned_dataset = remove_transform.transform(dataset)

    # Then apply imputation
    impute_transform = Imputer(imputation_strategy="mean")
    impute_transform.fit(cleaned_dataset)
    result = impute_transform.transform(cleaned_dataset)

    # Assert
    # Empty column should be removed
    assert "empty_feature" not in result.data.columns
    assert set(result.data.columns) == {"radiation", "temperature"}
    # Remaining NaNs should be imputed
    assert not result.data.isna().any().any()


def test_transform_not_fitted_error():
    """Test that NotFittedError is raised when transform is called before fit."""
    # Arrange
    data = pd.DataFrame(
        {"radiation": [100.0, 110.0]},
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=2, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    transform = Imputer(imputation_strategy="mean")

    # Act & Assert
    with pytest.raises(NotFittedError):
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
    transform = Imputer(imputation_strategy="mean")

    # Act
    transform.fit(dataset)
    result = transform.transform(dataset)

    # Assert
    # Data should be identical to original
    pd.testing.assert_frame_equal(result.data, original_data)
    # Sample interval should be preserved
    assert result.sample_interval == dataset.sample_interval


@pytest.mark.parametrize(
    "impute_estimator",
    [
        pytest.param(None, id="default"),
        pytest.param(
            RandomForestRegressor(
                n_estimators=2,  # not many trees for test speed
                max_depth=3,  # shallow tree for test speed
                bootstrap=True,
                max_samples=0.5,
                n_jobs=1,
                random_state=0,
            ),
            id="randomforest",
        ),
        pytest.param(BayesianRidge(), id="bayesianridge"),
        pytest.param(
            ExtraTreesRegressor(
                n_estimators=2,
                max_depth=3,
                bootstrap=True,
                max_samples=0.5,
                n_jobs=1,
                random_state=0,
            ),
            id="extra_trees",
        ),
    ],
)
def test_iterative_imputation_works(
    sample_dataset: TimeSeriesDataset, impute_estimator: RandomForestRegressor | BayesianRidge | ExtraTreesRegressor
):
    """Test that basic imputation removes NaN values for iterative strategy with different estimators."""
    # Arrange
    transform = Imputer(
        imputation_strategy="iterative",
        impute_estimator=impute_estimator,
        max_iterations=2,
        tolerance=10,  # high tolerance for test speed
    )

    # Act
    transform.fit(sample_dataset)
    result = transform.transform(sample_dataset)

    # Assert
    # Non-trailing NaN values should be imputed
    assert not result.data.isna().any().any()


def test_iterative_imputer_linear_relations(correlated_dataset: TimeSeriesDataset):
    """Test that iterative imputer learns linear relationship between features.

    Test with simple linear relation where temperature = 2 * radiation.
    The imputed value should be close to this relationship.
    """
    # Arrange

    transform = Imputer(
        imputation_strategy="iterative",
        impute_estimator=BayesianRidge(),
        tolerance=0.1,
    )

    # Act
    transform.fit(correlated_dataset)
    result = transform.transform(correlated_dataset)

    # Assert
    imputed_value_temperature = cast(float, result.data.loc[result.data.index[3], "temperature"])
    expected_value_temperature = (
        cast(float, correlated_dataset.data.loc[correlated_dataset.data.index[3], "radiation"]) * 2
    )

    imputed_value_radiation = cast(float, result.data.loc[result.data.index[1], "radiation"])
    expected_value_radiation = (
        cast(float, correlated_dataset.data.loc[correlated_dataset.data.index[1], "temperature"]) / 2
    )

    # Allow for tolerance
    assert abs(imputed_value_temperature - expected_value_temperature) < 1, (
        f"Expected ~{expected_value_temperature}, got {imputed_value_temperature}"
    )
    assert abs(imputed_value_radiation - expected_value_radiation) < 1, (
        f"Expected ~{expected_value_radiation}, got {imputed_value_radiation}"
    )


def test_iterative_imputer_only_one_feature():
    """Tests that warning is raised when using iterative imputer with only one feature.

    Initial_strategy is used for imputation."""
    # Arrange
    data = pd.DataFrame(
        {
            "radiation": [100.0, np.nan, 120.0, np.nan],
        },
        index=pd.date_range(datetime.fromisoformat("2025-01-01T00:00:00"), periods=4, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))

    transform = Imputer(
        imputation_strategy="iterative",
        impute_estimator=BayesianRidge(),
        initial_strategy="median",
        max_iterations=5,
        tolerance=1,
    )

    # Act
    transform.fit(dataset)

    # Assert
    # Check if warning is thrown by logger.
    logger = logging.getLogger("openstef_models.transforms.general.imputer")
    log_handler = logging.handlers.MemoryHandler(capacity=10)
    log_handler.setLevel(logging.WARNING)
    logger.addHandler(log_handler)

    try:
        result = transform.transform(dataset)
        log_handler.flush()

        log_messages = [record.getMessage() for record in log_handler.buffer]
        assert any(
            "Iterative imputer with only one feature will fall back to initial_strategy" in msg for msg in log_messages
        )
        assert any("Using 'median' for imputation" in msg for msg in log_messages)
    finally:
        logger.removeHandler(log_handler)

    # The first NaN should be imputed using the median strategy
    expected_median = 110.0  # Median of [100.0, 120.0]
    assert result.data.loc[result.data.index[1], "radiation"] == expected_median
    # The last NaN is a trailing NaN and should be preserved
    assert pd.isna(result.data.loc[result.data.index[3], "radiation"])
