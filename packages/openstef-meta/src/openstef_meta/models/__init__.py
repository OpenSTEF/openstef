# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""This module provides meta-forecasting models and related hyperparameters for the OpenSTEF project."""

from .learned_weights_forecaster import (
    LearnedWeightsForecaster,
    LearnedWeightsForecasterConfig,
    LearnedWeightsHyperParams,
)
from .residual_forecaster import ResidualForecaster, ResidualForecasterConfig, ResidualHyperParams
from .stacking_forecaster import StackingForecaster, StackingForecasterConfig, StackingHyperParams

__all__ = [
    "LearnedWeightsForecaster",
    "LearnedWeightsForecasterConfig",
    "LearnedWeightsHyperParams",
    "ResidualForecaster",
    "ResidualForecasterConfig",
    "ResidualHyperParams",
    "StackingForecaster",
    "StackingForecasterConfig",
    "StackingHyperParams",
]
