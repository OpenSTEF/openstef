# # SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
# #
# # SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.rules_combiner import (
    RulesCombiner,
    RulesCombinerConfig,
)
from openstef_meta.utils.datasets import EnsembleForecastDataset


@pytest.fixture
def config() -> RulesCombinerConfig:
    """Fixture to create RulesCombinerConfig."""
    return RulesCombiner.Config(
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime(timedelta(days=1))],
    )


@pytest.fixture
def forecaster(config: RulesCombinerConfig) -> RulesCombiner:
    return RulesCombiner(config=config)


def test_initialization(forecaster: RulesCombiner):
    assert isinstance(forecaster, RulesCombiner)


def test_quantile_weights_combiner__fit_predict(
    ensemble_dataset: EnsembleForecastDataset,
    config: RulesCombinerConfig,
):
    """Test basic fit and predict workflow with comprehensive output validation."""
    # Arrange
    expected_quantiles = config.quantiles
    forecaster = RulesCombiner(config=config)
    additional_features = ensemble_dataset.select_quantile(Q(0.5))
    additional_features.data = additional_features.data.drop(columns=additional_features.target_column)
    additional_features.data.columns = ["feature1", "feature2"]

    # Act
    forecaster.fit(ensemble_dataset, additional_features=additional_features)
    result = forecaster.predict(ensemble_dataset, additional_features=additional_features)

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
