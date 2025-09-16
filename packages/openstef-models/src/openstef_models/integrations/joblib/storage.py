# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import cast, override

from openstef_core.base_model import BaseModel
from openstef_core.exceptions import MissingExtraError, ModelNotFoundError
from openstef_models.integrations.model_storage import ModelIdentifier, ModelStorage
from openstef_models.models.forecasting_model import ForecastingModel

try:
    import joblib
except MissingExtraError as e:
    raise ImportError("integrations") from e


class LocalModelStorage(ModelStorage, BaseModel):
    """Note:
    joblib.dump() and joblib.load() are based on the Python pickle serialization model,
    which means that arbitrary Python code can be executed when loading a serialized object
      with joblib.load().

    joblib.load() should therefore never be used to load objects from an untrusted source
    or otherwise you will introduce a security vulnerability in your program.
    """

    storage_dir: Path

    def _get_model_path(self, model_id: ModelIdentifier) -> Path:
        return self.storage_dir / f"{model_id}.pkl"

    @override
    def load_model(self, model_id: ModelIdentifier) -> ForecastingModel:
        model_path = self._get_model_path(model_id)
        if not model_path.exists():
            raise ModelNotFoundError(str(model_id))

        return cast(ForecastingModel, joblib.load(model_path))  # type: ignore[reportUnknownMemberType]

    @override
    def save_model(self, model_id: ModelIdentifier, model: ForecastingModel) -> None:
        model_path = self._get_model_path(model_id)

        joblib.dump(model, model_path)  # type: ignore[reportUnknownMemberType]


__all__ = ["LocalModelStorage"]
