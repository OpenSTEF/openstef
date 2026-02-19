# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.mixins import HyperParams
from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.forecast_combiner import (
    ForecastCombiner,
    ForecastCombinerConfig,
)


@pytest.fixture
def horizons() -> list[LeadTime]:
    return [LeadTime(timedelta(hours=6)), LeadTime(timedelta(hours=12)), LeadTime(timedelta(days=1))]


@pytest.fixture
def config(horizons: list[LeadTime]) -> ForecastCombinerConfig:
    return ForecastCombinerConfig(
        hyperparams=HyperParams(),
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=horizons,
    )


def test_config_max_horizon(config: ForecastCombinerConfig, horizons: list[LeadTime]):
    """max_horizon returns the largest configured horizon."""
    # Act
    result = config.max_horizon

    # Assert
    assert result == max(horizons)


def test_config_with_horizon_returns_new_instance(config: ForecastCombinerConfig):
    """with_horizon creates a new config with the specified single horizon."""
    # Arrange
    new_horizon = LeadTime(timedelta(hours=3))

    # Act
    new_config = config.with_horizon(new_horizon)

    # Assert
    assert new_config.horizons == [new_horizon]
    assert len(config.horizons) == 3  # Original is unchanged


def test_config_with_horizon_preserves_other_fields(config: ForecastCombinerConfig):
    """with_horizon preserves quantiles and hyperparams."""
    # Act
    new_config = config.with_horizon(LeadTime(timedelta(hours=1)))

    # Assert
    assert new_config.quantiles == config.quantiles
    assert new_config.hyperparams == config.hyperparams


def test_config_requires_at_least_one_quantile():
    """Config validation rejects empty quantiles list."""
    # Act & Assert
    with pytest.raises(ValueError, match="quantiles"):
        ForecastCombinerConfig(
            hyperparams=HyperParams(),
            quantiles=[],
            horizons=[LeadTime(timedelta(hours=1))],
        )


def test_config_requires_at_least_one_horizon():
    """Config validation rejects empty horizons list."""
    # Act & Assert
    with pytest.raises(ValueError, match="horizons"):
        ForecastCombinerConfig(
            hyperparams=HyperParams(),
            quantiles=[Q(0.5)],
            horizons=[],
        )


def test_forecast_combiner_is_abstract():
    """ForecastCombiner cannot be instantiated directly — it has abstract methods."""
    with pytest.raises(TypeError, match="abstract"):
        ForecastCombiner()  # type: ignore[abstract]
