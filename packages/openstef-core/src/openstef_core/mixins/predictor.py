# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Prediction model interfaces and base classes.

This module provides abstract base classes for implementing prediction models
with state management capabilities. Predictors follow the scikit-learn pattern
with separate fit and predict phases, and support serialization through the
Stateful interface.
"""

from typing import Any, Self, cast

from pydantic import PrivateAttr, ValidatorFunctionWrapHandler, model_validator

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.mixins import abstractmethod
from openstef_core.exceptions import PredictError
from openstef_core.mixins.param_ranges import CategoricalRange, FloatRange, IntRange, TuningRange, get_class_range
from openstef_core.mixins.stateful import Stateful


class Predictor[I, O](Stateful):
    """Abstract base class for prediction models.

    This class provides the basic interface for models that can be fitted to data
    of type I and then used to generate predictions of type O. It follows the
    scikit-learn pattern with separate fit and predict phases, and includes
    state management capabilities.

    Type parameters:
        I: The input data type for fitting and prediction.
        O: The output prediction type.

    Subclasses must implement the is_fitted property, fit method, predict method,
    and the state management methods from Stateful.

    Example:
        Implementing a simple linear predictor:

        >>> class LinearPredictor(Predictor[list[float], float]):
        ...     def __init__(self):
        ...         self.slope = None
        ...         self.intercept = None
        ...
        ...     @property
        ...     def is_fitted(self) -> bool:
        ...         return self.slope is not None and self.intercept is not None
        ...
        ...     def fit(self, data: list[float]) -> None:
        ...         # Simple linear fit
        ...         self.slope = 1.0
        ...         self.intercept = 0.0
        ...
        ...     def predict(self, data: list[float]) -> float:
        ...         return self.slope * sum(data) + self.intercept
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the predictor has been fitted."""

    @abstractmethod
    def fit(self, data: I, data_val: I | None = None) -> Any:  # noqa: ANN401
        """Fit the predictor to the input data.

        This method should be called before generating predictions.
        It allows the predictor to learn parameters from the training data.

        Args:
            data: The training data to fit the predictor on.
            data_val: The validation data to evaluate and tune the predictor on (optional).
        """

    @abstractmethod
    def predict(self, data: I) -> O:
        """Generate predictions for the input data.

        This method should use the fitted parameters to generate predictions.

        Args:
            data: The input data to generate predictions for.

        Returns:
            Predictions for the input data.

        Raises:
            NotFittedError: If the predictor has not been fitted yet.
        """

    def fit_predict(self, data: I, data_val: I | None = None) -> O:
        """Fit the predictor to the data and then generate predictions.

        This method combines fitting and prediction into a single step.

        Args:
            data: The input data to fit and generate predictions for.
            data_val: The validation data to evaluate and tune the predictor on (optional).

        Returns:
            Predictions for the input data.
        """
        if not self.is_fitted:
            self.fit(data=data, data_val=data_val)

        return self.predict(data=data)


type BatchResult[T] = list[T | PredictError]


class BatchPredictor[I, O](Predictor[I, O]):
    """Abstract base class for batch prediction models.

    This class extends Predictor to provide batch prediction capabilities,
    allowing multiple predictions to be made efficiently while handling
    individual prediction errors gracefully.

    Batch predictions allow multiple prediction requests to be processed simultaneously,
    which is more efficient for models that support it, especially on GPUs.

    Type parameters:
        I: The input data type for fitting and prediction.
        O: The output prediction type.

    Subclasses inherit all requirements from Predictor and must implement
    the predict_batch method appropriately.

    Example:
        Implementing a batch linear predictor:

        >>> class BatchLinearPredictor(BatchPredictor[list[float], float]):
        ...     def __init__(self):
        ...         self.slope = None
        ...         self.intercept = None
        ...
        ...     @property
        ...     def is_fitted(self) -> bool:
        ...         return self.slope is not None and self.intercept is not None
        ...
        ...     def fit(self, data: list[float]) -> None:
        ...         # Simple linear fit
        ...         self.slope = 1.0
        ...         self.intercept = 0.0
        ...
        ...     def predict(self, data: list[float]) -> float:
        ...         return self.slope * sum(data) + self.intercept
        ...
        ...     def predict_batch(self, data: list[list[float]]) -> BatchResult[float]:
        ...         result = []
        ...         for item in data:
        ...             try:
        ...                 result.append(self.predict(item))
        ...             except PredictError as e:
        ...                 result.append(e)
        ...         return result
    """

    def predict_batch(self, data: list[I]) -> BatchResult[O]:
        """Generate predictions for multiple input data items.

        This method processes a batch of input data, generating predictions
        for each item. If any individual prediction fails, the error is
        captured and included in the results instead of failing the entire batch.

        Args:
            data: List of input data items to generate predictions for.

        Returns:
            List of predictions or PredictError instances for each input item.
            Successful predictions are of type O, failed predictions are PredictError.

        Note:
            This method does not raise exceptions; errors are captured in the result
            to allow partial batch processing to continue.
        """
        result: BatchResult[O] = []
        for data_part in data:
            try:
                result.append(self.predict(data_part))
            except PredictError as e:
                result.append(e)

        return result


class HyperParams(BaseConfig):
    """Base configuration for model hyperparameters.

    Serves as the foundation for model-specific hyperparameter configurations.
    Supports tuning ranges: pass a ``FloatRange``, ``IntRange``, or
    ``CategoricalRange`` as a field value at construction time and it will be
    extracted into ``_instance_ranges`` while the field keeps its declared default.

    Example:
        >>> from typing import Annotated
        >>> from openstef_core.mixins.param_ranges import FloatRange, IntRange
        >>> class MyHP(HyperParams):
        ...     lr: Annotated[float, FloatRange(low=0.01, high=1.0)] = 0.3
        ...     depth: Annotated[int, IntRange(low=1, high=15)] = 6
        >>> hp = MyHP(lr=FloatRange(low=0.001, high=0.5, tune=True))
        >>> hp.lr  # field keeps its default
        0.3
        >>> hp.get_search_space()  # extracted range
        {'lr': FloatRange(low=0.001, high=0.5, log=False, tune=True)}
    """

    _instance_ranges: dict[str, TuningRange] = PrivateAttr(default_factory=dict[str, TuningRange])

    @model_validator(mode="wrap")
    @classmethod
    def _extract_tuning_ranges(
        cls,
        data: dict[str, object] | object,
        handler: ValidatorFunctionWrapHandler,
    ) -> Self:
        """Strip TuningRange values from kwargs and store as instance metadata.

        Returns:
            Validated ``HyperParams`` with tuning ranges stored separately.
        """
        instance_ranges: dict[str, TuningRange] = {}
        if isinstance(data, dict):
            raw = cast(dict[str, object], data)
            cleaned: dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(value, (FloatRange, IntRange, CategoricalRange)):
                    instance_ranges[key] = value
                else:
                    cleaned[key] = value
            data = cleaned
        result: HyperParams = handler(data)
        if instance_ranges and result.__pydantic_private__ is not None:
            result._instance_ranges = instance_ranges
        return result  # type: ignore[return-value]

    def get_search_space(self, include: set[str] | None = None) -> dict[str, TuningRange]:
        """Merge instance and class-level ranges, returning only ``tune=True`` fields.

        Args:
            include: If given, restrict output to these field names.

        Returns:
            Mapping of field name to resolved ``TuningRange``.

        Raises:
            KeyError: If *include* contains names not in the tunable space.
        """
        result: dict[str, TuningRange] = {}
        for field_name, field_info in type(self).model_fields.items():
            class_range = get_class_range(field_info)
            override = self._instance_ranges.get(field_name)

            if override is not None:
                if not override.tune:
                    continue
                result[field_name] = override.resolve(class_range)  # type: ignore[arg-type]
            elif class_range is not None and class_range.tune:
                result[field_name] = class_range

        if include is not None:
            missing = include - result.keys()
            if missing:
                msg = (
                    f"Fields {sorted(missing)!r} not found in the tunable search space. "
                    "Check that they exist on the HyperParams class and were passed as "
                    "TuningRange(tune=True) in the constructor."
                )
                raise KeyError(msg)
            result = {k: result[k] for k in include}

        return result


__all__ = [
    "BatchPredictor",
    "BatchResult",
    "HyperParams",
    "Predictor",
]
