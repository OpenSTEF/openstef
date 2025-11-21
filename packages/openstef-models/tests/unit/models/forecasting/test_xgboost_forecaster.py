# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
    XGBoostHyperParams,
)


@pytest.fixture
def base_config() -> XGBoostForecasterConfig:
    """Base configuration for XGBoost forecaster tests."""
    return XGBoostForecasterConfig(
        horizons=[LeadTime(timedelta(days=1))],
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        hyperparams=XGBoostHyperParams(
            n_estimators=10,  # Small for fast tests
        ),
        verbosity=0,  # Quiet for tests
    )


def test_xgboost_forecaster__fit_predict(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: XGBoostForecasterConfig,
):
    """Test basic fit and predict workflow with output validation."""
    # Arrange
    expected_quantiles = base_config.quantiles
    forecaster = XGBoostForecaster(config=base_config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Basic functionality
    assert forecaster.is_fitted, "Model should be fitted after calling fit()"

    # Check that necessary quantiles are present
    expected_columns = [q.format() for q in expected_quantiles]
    assert list(result.data.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(result.data.columns)}"
    )

    # Forecast data quality
    assert not result.data.isna().any().any(), "Forecast should not contain NaN or None values"

    # Since forecast is deterministic with fixed random seed, check value spread (vectorized)
    # All quantiles should have some variation (not all identical values)
    stds = result.data.std()
    assert (stds > 0).all(), f"All columns should have variation, got stds: {dict(stds)}"


def test_xgboost_forecaster__predict_not_fitted_raises_error(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: XGBoostForecasterConfig,
):
    """Test that predict() raises NotFittedError when called before fit()."""
    # Arrange
    forecaster = XGBoostForecaster(config=base_config)

    # Act & Assert
    with pytest.raises(NotFittedError, match="XGBoostForecaster"):
        forecaster.predict(sample_forecast_input_dataset)


def test_xgboost_forecaster__with_sample_weights(
    sample_dataset_with_weights: ForecastInputDataset,
    base_config: XGBoostForecasterConfig,
):
    """Test that forecaster works with sample weights and produces different results."""
    # Arrange
    forecaster_with_weights = XGBoostForecaster(config=base_config)

    # Create dataset without weights for comparison
    data_without_weights = ForecastInputDataset(
        data=sample_dataset_with_weights.data.drop(columns=["sample_weight"]),
        sample_interval=sample_dataset_with_weights.sample_interval,
        target_column=sample_dataset_with_weights.target_column,
        forecast_start=sample_dataset_with_weights.forecast_start,
    )
    forecaster_without_weights = XGBoostForecaster(config=base_config)

    # Act
    forecaster_with_weights.fit(sample_dataset_with_weights)
    forecaster_without_weights.fit(data_without_weights)

    # Predict using data without sample_weight column (since that's used for training, not prediction)
    result_with_weights = forecaster_with_weights.predict(data_without_weights)
    result_without_weights = forecaster_without_weights.predict(data_without_weights)

    # Assert
    # Both should produce valid forecasts
    assert not result_with_weights.data.isna().any().any(), "Weighted forecast should not contain NaN values"
    assert not result_without_weights.data.isna().any().any(), "Unweighted forecast should not contain NaN values"

    # Sample weights should affect the model, so results should be different
    # (This is a statistical test - with different weights, predictions should differ)
    differences = (result_with_weights.data - result_without_weights.data).abs()
    assert differences.sum().sum() > 0, "Sample weights should affect model predictions"


@pytest.mark.parametrize("objective", ["pinball_loss", "arctan_loss"])
def test_xgboost_forecaster__different_objectives(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: XGBoostForecasterConfig,
    objective: str,
):
    """Test that forecaster works with different objective functions."""
    # Arrange
    config = base_config.model_copy(
        update={
            "hyperparams": base_config.hyperparams.model_copy(
                update={"objective": objective}  # type: ignore[arg-type]
            )
        }
    )
    forecaster = XGBoostForecaster(config=config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    # Basic functionality should work regardless of objective
    assert forecaster.is_fitted, f"Model with {objective} should be fitted"
    assert not result.data.isna().any().any(), f"Forecast with {objective} should not contain NaN values"

    # Check value spread for each objective
    # Note: Some objectives (like arctan_loss) may produce zero variation for some quantiles with small datasets
    stds = result.data.std()
    # At least one quantile should have variation (the model should not be completely degenerate)
    assert (stds > 0).any(), f"At least one column should have variation with {objective}, got stds: {dict(stds)}"


def test_xgboost_forecaster__feature_importances(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: XGBoostForecasterConfig,
):
    """Test that feature_importances returns correct normalized importance scores."""
    # Arrange
    forecaster = XGBoostForecaster(config=base_config)
    forecaster.fit(sample_forecast_input_dataset)

    # Act
    feature_importances = forecaster.feature_importances

    # Assert
    assert len(feature_importances.index) > 0

    # Columns should match expected quantile formats
    expected_columns = pd.Index([q.format() for q in base_config.quantiles], name="quantiles")
    pd.testing.assert_index_equal(feature_importances.columns, expected_columns)

    # Values should be normalized (sum to 1.0 per quantile column) and non-negative
    col_sums = feature_importances.sum(axis=0)
    pd.testing.assert_series_equal(col_sums, pd.Series(1.0, index=expected_columns), atol=1e-10)
    assert (feature_importances >= 0).all().all()
