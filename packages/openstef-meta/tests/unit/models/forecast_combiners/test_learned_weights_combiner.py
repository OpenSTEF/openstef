# # SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
# #
# # SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.learned_weights_combiner import (
    WeightsCombiner,
    WeightsCombinerConfig,
)
from openstef_meta.utils.datasets import EnsembleForecastDataset


@pytest.fixture(params=["lgbm", "xgboost", "rf", "logistic"])
def classifier(request: pytest.FixtureRequest) -> str:
    """Fixture to provide different classifier types for LearnedWeightsCombiner tests."""
    return request.param


@pytest.fixture
def config(classifier: str) -> WeightsCombinerConfig:
    """Fixture to create WeightsCombinerConfig based on the classifier type."""
    if classifier == "lgbm":
        hp = WeightsCombiner.LGBMHyperParams(n_leaves=5, n_estimators=10)
    elif classifier == "xgboost":
        hp = WeightsCombiner.XGBHyperParams(n_estimators=10)
    elif classifier == "rf":
        hp = WeightsCombiner.RFHyperParams(n_estimators=10, n_leaves=5)
    elif classifier == "logistic":
        hp = WeightsCombiner.LogisticHyperParams()
    else:
        msg = f"Unsupported classifier type: {classifier}"
        raise ValueError(msg)

    return WeightsCombiner.Config(
        hyperparams=hp, quantiles=[Q(0.1), Q(0.5), Q(0.9)], horizons=[LeadTime(timedelta(days=1))]
    )


@pytest.fixture
def forecaster(config: WeightsCombinerConfig) -> WeightsCombiner:
    return WeightsCombiner(config)


def test_initialization(forecaster: WeightsCombiner):
    assert isinstance(forecaster, WeightsCombiner)


def test_quantile_weights_combiner__fit_predict(
    ensemble_dataset: EnsembleForecastDataset,
    config: WeightsCombinerConfig,
):
    """Test basic fit and predict workflow with comprehensive output validation."""
    # Arrange
    expected_quantiles = config.quantiles
    forecaster = WeightsCombiner(config=config)

    # Act
    forecaster.fit(ensemble_dataset)
    result = forecaster.predict(ensemble_dataset)

    # Assert
    # Basic functionality
    assert forecaster.is_fitted, "Model should be fitted after calling fit()"

    # Check that necessary quantiles are present
    expected_columns = [q.format() for q in expected_quantiles]
    expected_columns.append("load")
    assert list(result.data.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(result.data.columns)}"
    )

    # Forecast data quality
    assert not result.data.isna().any().any(), "Forecast should not contain NaN or None values"

    # Since forecast is deterministic with fixed random seed, check value spread (vectorized)
    # All quantiles should have some variation (not all identical values)
    stds = result.data.std()
    assert (stds > 0).all(), f"All columns should have variation, got stds: {dict(stds)}"


def test_weights_combiner_not_fitted_error(
    ensemble_dataset: EnsembleForecastDataset,
    config: WeightsCombinerConfig,
):
    """Test that NotFittedError is raised when predicting before fitting."""
    # Arrange
    forecaster = WeightsCombiner(config=config)
    # Act & Assert
    with pytest.raises(NotFittedError):
        forecaster.predict(ensemble_dataset)
