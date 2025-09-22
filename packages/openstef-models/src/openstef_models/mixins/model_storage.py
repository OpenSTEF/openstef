# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract interface for persisting and loading models.

Defines the contract for model storage systems that can save and restore
trained models. Implementations might use local file systems, cloud storage,
databases, or model registries.
"""

from abc import ABC, abstractmethod

from openstef_core.mixins import Stateful

type ModelIdentifier = str


class ModelStorage(ABC):
    """Abstract interface for storing and retrieving models.

    Defines the contract for model persistence systems. Implementations
    should handle serialization/deserialization of complete model instances
    including their preprocessing pipelines, forecasters, and
    postprocessing components.

    Invariants:
        - save_model_state() followed by load_model_state() with the same ID should return
          an equivalent model that produces the same predictions
        - Model IDs should be unique within the storage system

    Example:
        Implementing a simple file-based storage:

        >>> import pickle
        >>> class FileModelStorage(ModelStorage):
        ...     def __init__(self, base_path: str):
        ...         self.base_path = Path(base_path)
        ...         self.base_path.mkdir(exist_ok=True)
        ...
        ...     def save_model_state(self, model_id: ModelIdentifier, model: Stateful) -> None:
        ...         path = self.base_path / f"{model_id}.pkl"
        ...         with open(path, 'wb') as f:
        ...             pickle.dump(model, f)
        ...
        ...     def load_model_state(self, model_id: ModelIdentifier, model: Stateful) -> Stateful:
        ...         path = self.base_path / f"{model_id}.pkl"
        ...         if not path.exists():
        ...             raise ModelNotFoundError(model_id)
        ...         with open(path, 'rb') as f:
        ...             return pickle.load(f)
    """

    @abstractmethod
    def load_model_state[T: Stateful](self, model_id: ModelIdentifier, model: T) -> T:
        """Load a previously saved forecasting model.

        Args:
            model_id: Unique identifier for the model to load.
            model: An instance of the model class to populate with loaded state.

        Returns:
            Complete forecasting model ready for prediction.

        Raises:
            ModelNotFoundError: If no model exists with the given ID.
        """

    @abstractmethod
    def save_model_state(self, model_id: ModelIdentifier, model: Stateful) -> None:
        """Save a forecasting model for later retrieval.

        Persists the complete model state including preprocessing pipeline,
        forecaster parameters, and postprocessing configuration. Implementations
        can choose whether to overwrite existing models or handle duplicates differently.

        Args:
            model_id: Unique identifier for storing the model.
            model: Complete forecasting model to save.
        """


__all__ = ["ModelIdentifier", "ModelStorage"]
