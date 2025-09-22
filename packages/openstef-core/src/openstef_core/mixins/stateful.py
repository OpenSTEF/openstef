# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""State management utilities for serializable objects.

Provides the foundation for object persistence and serialization across the
OpenSTEF ecosystem. Enables objects to save their state and restore
it later, supporting use cases like model deployment, caching, and distributed
processing scenarios.
"""

from abc import ABC, abstractmethod
from typing import Self

type State = object


class Stateful(ABC):
    """Mixin for objects that can save and restore their internal state.

    Enables object persistence by providing serialization capabilities. Implementations
    must handle state management in a way that allows objects to be saved to disk,
    transmitted over networks, or stored in databases, then restored to their exact
    previous state.

    State typically includes configuration, trained parameters, and any learned
    patterns. The state format should be JSON-serializable for maximum compatibility.

    Guarantees:
        - to_state() followed by from_state() must restore identical behavior
        - State format should be forward-compatible across minor version updates

    Example:
        Basic state management implementation:

        >>> class MyPredictor(Stateful):
        ...     def __init__(self, config):
        ...         self.config = config
        ...         self.trained_params = None
        ...
        ...     def to_state(self):
        ...         return {
        ...             "version": 1,
        ...             "config": self.config,
        ...             "trained_params": self.trained_params
        ...         }
        ...
        ...     def from_state(self, state):
        ...         instance = self.__class__(config=state["config"])
        ...         instance.trained_params = state["trained_params"]
        ...         return instance

    See Also:
        Transform: Data transformation interface using state management.
        Predictor: Prediction interface using state management.
    """

    @abstractmethod
    def to_state(self) -> State:
        """Serialize the current state of the object.

        Must capture all information needed to restore the object to its current state,
        including configuration, trained parameters, and any internal state variables.

        Returns:
            Serializable representation of the object state.
        """

    @abstractmethod
    def from_state(self, state: State) -> Self:
        """Restore an object from its serialized state.

        Must reconstruct the object to match the exact state when to_state()
        was called. Should handle version compatibility for older state formats
        when possible.

        Args:
            state: Serialized state returned from to_state().

        Returns:
            Object instance restored to the exact previous state.

        Raises:
            ValueError: If state is invalid or incompatible.
        """
