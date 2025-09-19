# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Base classes and mixins for OpenSTEF models.

This package provides the foundational interfaces and utilities for building
forecasting and component splitting models. Includes abstract base classes,
configuration classes, and mixins that define the standard contracts and
behaviors expected across the OpenSTEF modeling ecosystem.
"""

from .component_splitter_mixin import (
    ComponentSplitterConfig,
    ComponentSplitterMixin,
)
from .forecaster_mixin import (
    BaseForecaster,
    BaseForecasterMixin,
    BaseHorizonForecaster,
    ForecasterConfig,
    ForecasterHyperParams,
    HorizonForecasterConfig,
)
from .stateful_model_mixin import (
    ModelState,
    StatefulModelMixin,
)

__all__ = [
    "BaseForecaster",
    "BaseForecasterMixin",
    "BaseHorizonForecaster",
    "ComponentSplitterConfig",
    "ComponentSplitterMixin",
    "ForecasterConfig",
    "ForecasterHyperParams",
    "HorizonForecasterConfig",
    "ModelState",
    "StatefulModelMixin",
]
