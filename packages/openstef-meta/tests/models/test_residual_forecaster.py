# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.datasets import ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_meta.framework.base_learner import BaseLearnerHyperParams
from openstef_meta.models.residual_forecaster import (
    ResidualForecaster,
    ResidualForecasterConfig,
    ResidualHyperParams,
)
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearHyperParams
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearHyperParams
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams


@pytest.fixture(params=["gblinear", "lgbmlinear"])
def primary_model(request: pytest.FixtureRequest) -> BaseLearnerHyperParams:
    """Fixture to provide different primary models types."""
    learner_type = request.param
    if learner_type == "gblinear":
        return GBLinearHyperParams()
    if learner_type == "lgbm":
        return LGBMHyperParams()
    if learner_type == "lgbmlinear":
        return LGBMLinearHyperParams()
    return XGBoostHyperParams()


@pytest.fixture(params=["gblinear", "lgbm", "lgbmlinear", "xgboost"])
def secondary_model(request: pytest.FixtureRequest) -> BaseLearnerHyperParams:
    """Fixture to provide different secondary models types."""
    learner_type = request.param
    if learner_type == "gblinear":
        return GBLinearHyperParams()
    if learner_type == "lgbm":
        return LGBMHyperParams()
    if learner_type == "lgbmlinear":
        return LGBMLinearHyperParams()
    return XGBoostHyperParams()


@pytest.fixture
def base_config(
    primary_model: BaseLearnerHyperParams,
    secondary_model: BaseLearnerHyperParams,
) -> ResidualForecasterConfig:
    """Base configuration for Residual forecaster tests."""

    params = ResidualHyperParams(
        primary_hyperparams=primary_model,
        secondary_hyperparams=secondary_model,
    )
    return ResidualForecasterConfig(
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime(timedelta(days=1))],
        hyperparams=params,
        verbosity=False,
    )


def test_residual_forecaster_fit_predict(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: ResidualForecasterConfig,
):
    """Test basic fit and predict workflow with comprehensive output validation."""
    # Arrange
    expected_quantiles = base_config.quantiles
    forecaster = ResidualForecaster(config=base_config)

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


def test_residual_forecaster_predict_not_fitted_raises_error(
    sample_forecast_input_dataset: ForecastInputDataset,
    base_config: ResidualForecasterConfig,
):
    """Test that predict() raises NotFittedError when called before fit()."""
    # Arrange
    forecaster = ResidualForecaster(config=base_config)

    # Act & Assert
    with pytest.raises(NotFittedError, match="ResidualForecaster"):
        forecaster.predict(sample_forecast_input_dataset)


def test_residual_forecaster_with_sample_weights(
    sample_dataset_with_weights: ForecastInputDataset,
    base_config: ResidualForecasterConfig,
):
    """Test that forecaster works with sample weights and produces different results."""
    # Arrange
    forecaster_with_weights = ResidualForecaster(config=base_config)

    # Create dataset without weights for comparison
    data_without_weights = ForecastInputDataset(
        data=sample_dataset_with_weights.data.drop(columns=["sample_weight"]),
        sample_interval=sample_dataset_with_weights.sample_interval,
        target_column=sample_dataset_with_weights.target_column,
        forecast_start=sample_dataset_with_weights.forecast_start,
    )
    forecaster_without_weights = ResidualForecaster(config=base_config)

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
