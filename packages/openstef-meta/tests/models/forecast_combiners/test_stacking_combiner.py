# # SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
# #
# # SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
import pandas as pd
import pytest

from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.stacking_combiner import (
    StackingCombiner,
    StackingCombinerConfig,
)
from openstef_meta.utils.datasets import EnsembleForecastDataset


@pytest.fixture(params=["lgbm", "gblinear"])
def regressor(request: pytest.FixtureRequest) -> str:
    """Fixture to provide different regressor types for Stacking tests."""
    return request.param


@pytest.fixture
def config(regressor: str) -> StackingCombinerConfig:
    """Fixture to create StackingCombinerConfig based on the regressor type."""
    if regressor == "lgbm":
        hp = StackingCombiner.LGBMHyperParams(num_leaves=5, n_estimators=10)
    elif regressor == "gblinear":
        hp = StackingCombiner.GBLinearHyperParams(n_steps=10)
    else:
        msg = f"Unsupported regressor type: {regressor}"
        raise ValueError(msg)

    return StackingCombiner.Config(
        hyperparams=hp, quantiles=[Q(0.1), Q(0.5), Q(0.9)], horizons=[LeadTime(timedelta(days=1))]
    )


@pytest.fixture
def forecaster(config: StackingCombinerConfig) -> StackingCombiner:
    return StackingCombiner(config)


def test_initialization(forecaster: StackingCombiner):
    assert isinstance(forecaster, StackingCombiner)


def test_quantile_weights_combiner__fit_predict(
    ensemble_dataset: EnsembleForecastDataset,
    config: StackingCombinerConfig,
):
    """Test basic fit and predict workflow with comprehensive output validation."""
    # Arrange
    expected_quantiles = config.quantiles
    forecaster = StackingCombiner(config=config)

    # Act
    forecaster.fit(ensemble_dataset)
    result = forecaster.predict(ensemble_dataset)

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


def test_stacking_combiner_not_fitted_error(
    ensemble_dataset: EnsembleForecastDataset,
    config: StackingCombinerConfig,
):
    """Test that NotFittedError is raised when predicting before fitting."""
    # Arrange
    forecaster = StackingCombiner(config=config)
    # Act & Assert
    with pytest.raises(NotFittedError):
        forecaster.predict(ensemble_dataset)


def test_stacking_combiner_predict_contributions(
    ensemble_dataset: EnsembleForecastDataset,
    config: StackingCombinerConfig,
):
    """Test that predict_contributions method returns contributions with correct shape."""
    # Arrange
    forecaster = StackingCombiner(config=config)
    forecaster.fit(ensemble_dataset)

    # Act
    contributions = forecaster.predict_contributions(ensemble_dataset)

    # Assert
    assert isinstance(contributions, pd.DataFrame), "Contributions should be returned as a DataFrame."
    assert len(contributions.columns) == len(ensemble_dataset.quantiles) * len(ensemble_dataset.forecaster_names)
