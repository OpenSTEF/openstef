# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Local model storage implementation using joblib serialization.

Provides file-based persistence for ForecastingModel instances using joblib's
pickle-based serialization. This storage backend is suitable for development,
testing, and single-machine deployments where models need to be persisted
to the local filesystem.
"""

from typing import BinaryIO, ClassVar, override

from openstef_core.exceptions import MissingExtraError
from openstef_models.mixins.model_serializer import ModelSerializer

try:
    from skops.io import dump, get_untrusted_types, load
except ImportError as e:
    raise MissingExtraError("joblib", package="openstef-models") from e


class SkopsModelSerializer(ModelSerializer):
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

    Example:
        Basic usage with model persistence:

        >>> from pathlib import Path
        >>> from openstef_models.models.forecasting_model import ForecastingModel
        >>> storage = LocalModelStorage(storage_dir=Path("./models"))  # doctest: +SKIP
        >>> storage.save_model("my_model", my_forecasting_model)  # doctest: +SKIP
        >>> loaded_model = storage.load_model("my_model")  # doctest: +SKIP
    """

    extension: ClassVar[str] = ".skops"

    @override
    def serialize(self, model: object, file: BinaryIO) -> None:
        dump(model, file)  # type: ignore[reportUnknownMemberType]

    @staticmethod
    def _get_stateful_types() -> set[str]:
        return {
            "tests.unit.integrations.skops.test_skops_model_serializer.SimpleSerializableModel",
            "openstef_core.mixins.predictor.BatchPredictor",
            "openstef_models.models.forecasting.forecaster.Forecaster",
            "openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster",
            "openstef_models.models.component_splitting_model.ComponentSplittingModel",
            "openstef_core.mixins.transform.TransformPipeline",
            "openstef_core.mixins.transform.TransformPipeline[EnergyComponentDataset]",
            "openstef_core.mixins.transform.TransformPipeline[TimeSeriesDataset]",
            "openstef_models.models.forecasting.lgbm_forecaster.LGBMForecaster",
            "openstef_models.models.component_splitting.component_splitter.ComponentSplitter",
            "openstef_models.models.forecasting_model.ForecastingModel",
            "openstef_core.mixins.transform.Transform",
            "openstef_core.mixins.transform.TransformPipeline[ForecastDataset]",
            "openstef_core.mixins.predictor.Predictor",
            "openstef_models.models.forecasting.lgbmlinear_forecaster.LGBMLinearForecaster",
        }

    @override
    def deserialize(self, file: BinaryIO) -> object:
        """Load a model's state from a binary file and restore it.

        Returns:
            The restored model instance.

        Raises:
            ValueError: If no safe types are found in the serialized model.
        """
        safe_types = self._get_stateful_types()

        # Weak security measure that checks a safe class is present.
        # Can be improved to ensure no unsafe classes are present.
        model_types: set[str] = set(get_untrusted_types(file=file))  # type: ignore

        if len(safe_types.intersection(model_types)) == 0:
            raise ValueError("Deserialization aborted: No safe types found in the serialized model.")

        return load(file, trusted=list(model_types))  # type: ignore[reportUnknownMemberType]


__all__ = ["SkopsModelSerializer"]
