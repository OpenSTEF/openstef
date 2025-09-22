# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Prediction model interfaces and base classes.

This module provides abstract base classes for implementing prediction models
with state management capabilities. Predictors follow the scikit-learn pattern
with separate fit and predict phases, and support serialization through the
Stateful interface.
"""

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.mixins import abstractmethod
from openstef_core.exceptions import PredictError
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
        ...
        ...     def to_state(self):
        ...         return {"slope": self.slope, "intercept": self.intercept}
        ...
        ...     @classmethod
        ...     def from_state(cls, state):
        ...         instance = cls()
        ...         instance.slope = state["slope"]
        ...         instance.intercept = state["intercept"]
        ...         return instance
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the predictor has been fitted."""

    @abstractmethod
    def fit(self, data: I, data_val: I | None = None) -> None:
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
        ...
        ...     def to_state(self):
        ...         return {"slope": self.slope, "intercept": self.intercept}
        ...
        ...     @classmethod
        ...     def from_state(cls, state):
        ...         instance = cls()
        ...         instance.slope = state["slope"]
        ...         instance.intercept = state["intercept"]
        ...         return instance
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
    Inheriting classes should add their specific parameters as Pydantic fields
    with appropriate validation and documentation.

    Example:
        Creating custom hyperparameters for a specific model:

        >>> from pydantic import Field
        >>> class MyModelHyperParams(HyperParams):
        ...     learning_rate: float = Field(default=0.01, gt=0, description="Learning rate for training")
        ...     max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    """


__all__ = [
    "BatchPredictor",
    "BatchResult",
    "HyperParams",
    "Predictor",
]
