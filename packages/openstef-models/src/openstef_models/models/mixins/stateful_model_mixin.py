# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""State management utilities for machine learning models.

Provides the foundation for model persistence and serialization across the
OpenSTEF ecosystem. Enables models to save their trained state and restore
it later, supporting use cases like model deployment, and distributed training
scenarios.
"""

from abc import ABC, abstractmethod
from typing import Self

type ModelState = object


class StatefulModelMixin(ABC):
    """Mixin for forecasters that can save and restore their internal state.

    Enables model persistence by providing serialization capabilities. Implementations
    must handle state management in a way that allows models to be saved to disk,
    transmitted over networks, or stored in databases, then restored to their exact
    previous state.

    State typically includes trained parameters, configuration, and any learned
    patterns. The state format should be JSON-serializable for maximum compatibility.

    Guarantees:
        - get_state() followed by from_state() must restore identical behavior
        - State format should be forward-compatible across minor version updates

    Example:
        Basic state management implementation:

        >>> class MyForecaster(StatefulModelMixin):
        ...     def __init__(self, config):
        ...         self.config = config
        ...         self.trained_params = None
        ...
        ...     def get_state(self):
        ...         return {
        ...             "version": 1,
        ...             "config": self.config.model_dump(),
        ...             "trained_params": self.trained_params
        ...         }
        ...
        ...     @classmethod
        ...     def from_state(cls, state):
        ...         instance = cls(config=Config.model_validate(state["config"]))
        ...         instance.trained_params = state["trained_params"]
        ...         return instance

    See Also:
        BaseForecasterMixin: Base interface for forecaster capabilities.
        BaseForecaster: Multi-horizon forecaster using state management.
        BaseHorizonForecaster: Single-horizon forecaster using state management.
    """

    @abstractmethod
    def get_state(self) -> ModelState:
        """Serialize the current state of the forecaster.

        Must capture all information needed to restore the model to its current state,
        including configuration, trained parameters, and any internal state variables.

        Returns:
            Serializable representation of the model state.
        """
        raise NotImplementedError("Subclasses must implement state serialization")

    @classmethod
    @abstractmethod
    def from_state(cls, state: ModelState) -> Self:
        """Restore a forecaster from its serialized state.

        Must reconstruct the forecaster to match the exact state when get_state()
        was called. Should handle version compatibility for older state formats
        when possible.

        Args:
            state: Serialized state returned from get_state().

        Returns:
            Forecaster instance restored to the exact previous state.

        Raises:
            ModelLoadingError: If state is invalid or incompatible.
        """
        raise NotImplementedError("Subclasses must implement state deserialization")


__all__ = ["StatefulModelMixin"]
