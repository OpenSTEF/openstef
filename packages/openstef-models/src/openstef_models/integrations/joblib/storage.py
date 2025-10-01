# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Local model storage implementation using joblib serialization.

Provides file-based persistence for ForecastingModel instances using joblib's
pickle-based serialization. This storage backend is suitable for development,
testing, and single-machine deployments where models need to be persisted
to the local filesystem.
"""

from pathlib import Path
from typing import cast, override

from openstef_core.base_model import BaseModel
from openstef_core.exceptions import MissingExtraError, ModelNotFoundError
from openstef_core.mixins import State, Stateful
from openstef_models.mixins.model_storage import ModelIdentifier, ModelStorage

try:
    import joblib
except ImportError as e:
    raise MissingExtraError("joblib", package="openstef-models") from e


class LocalModelStorage(BaseModel, ModelStorage):
    """File-based model storage using joblib serialization.

    Provides persistent storage for ForecastingModel instances on the local
    filesystem. Models are serialized using joblib and stored as pickle files
    in the specified directory.

    This storage implementation is suitable for development, testing, and
    single-machine deployments where simple file-based persistence is sufficient.

    Note:
        joblib.dump() and joblib.load() are based on the Python pickle serialization model,
        which means that arbitrary Python code can be executed when loading a serialized object
        with joblib.load().

        joblib.load() should therefore never be used to load objects from an untrusted source
        or otherwise you will introduce a security vulnerability in your program.

    Invariants:
        - Models are stored as .pkl files in the configured storage directory
        - Model files use the pattern: {model_id}.pkl
        - Storage directory is created automatically if it doesn't exist
        - Load operations fail with ModelNotFoundError if model file doesn't exist

    Args:
        storage_dir: Directory path where model files will be stored.

    Example:
        Basic usage with model persistence:

        >>> from pathlib import Path
        >>> from openstef_models.models.forecasting_model import ForecastingModel
        >>> storage = LocalModelStorage(storage_dir=Path("./models"))  # doctest: +SKIP
        >>> storage.save_model("my_model", my_forecasting_model)  # doctest: +SKIP
        >>> loaded_model = storage.load_model("my_model")  # doctest: +SKIP
    """

    storage_dir: Path

    def _get_model_path(self, model_id: ModelIdentifier) -> Path:
        return self.storage_dir / f"{model_id}.pkl"

    @override
    def load_model_state[T: Stateful](self, model_id: ModelIdentifier, model: T) -> T:
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_id))

        state = cast(State, joblib.load(model_path))  # type: ignore[reportUnknownMemberType]
        return model.from_state(state)

    @override
    def save_model_state(self, model_id: ModelIdentifier, model: Stateful) -> None:
        model_path = self._get_model_path(model_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        state = model.to_state()
        joblib.dump(state, model_path)  # type: ignore[reportUnknownMemberType]


__all__ = ["LocalModelStorage"]
