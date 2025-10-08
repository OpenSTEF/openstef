# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract interface for persisting and loading models.

Defines the contract for model storage systems that can save and restore
trained models. Implementations might use local file systems, cloud storage,
databases, or model registries.
"""

from abc import ABC, abstractmethod
from typing import BinaryIO, ClassVar

from openstef_core.base_model import BaseConfig
from openstef_core.mixins import Stateful

type ModelIdentifier = str


class ModelSerializer(BaseConfig, ABC):
    """Abstract base class for model serialization.

    Defines the interface for converting trained models to and from binary format.
    Implementations handle the mechanics of serializing model state using specific
    libraries like joblib, pickle, or custom formats.

    The serializer ensures that all stateful components of a model can be persisted
    and restored, enabling model reuse across sessions and deployments.

    Invariants:
        - Serializing and deserializing a model preserves its state
        - The extension attribute specifies the file extension for saved models
        - Deserialized models are functionally equivalent to their original state

    See Also:
        JoblibModelSerializer: Concrete implementation using joblib.
    """

    extension: ClassVar[str]

    @abstractmethod
    def serialize(self, model: Stateful, file: BinaryIO) -> None:
        """Write a model's state to a binary file.

        Converts the model's internal state to a binary format and writes it to
        the provided file object. The serialization must capture all information
        needed to restore the model to its current state.

        Args:
            model: The stateful model to serialize.
            file: Binary file object opened for writing.
        """

    @abstractmethod
    def deserialize[T: Stateful](self, model: T, file: BinaryIO) -> T:
        """Read a model's state from a binary file and restore it.

        Loads the model state from the binary file and applies it to the provided
        model instance. The model should be functionally equivalent to the state
        when it was serialized.

        Args:
            model: The model instance to populate with the loaded state.
            file: Binary file object opened for reading, positioned at the start
                of the serialized model data.

        Returns:
            The same model instance with its state restored from the file.
        """


__all__ = ["ModelIdentifier", "ModelSerializer"]
