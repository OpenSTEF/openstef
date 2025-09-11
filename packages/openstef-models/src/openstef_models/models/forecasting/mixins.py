"""Core forecasting model interfaces and configurations.

Provides the fundamental building blocks for implementing forecasting models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different model types
while supporting both single and multi-horizon forecasting scenarios.

The module follows a layered design:
- Configuration classes define model parameters and horizons
- Base mixins provide common functionality and properties
- Specialized mixins handle horizon-specific vs multi-horizon forecasting
- State management enables model serialization and loading

Key concepts:
- **Horizon**: The lead time for predictions (e.g., 1 hour, 24 hours ahead)
- **Quantiles**: Probability levels for uncertainty estimation (e.g., 0.1, 0.5, 0.9)
- **State**: Serializable model parameters that enable saving/loading trained models
- **Batching**: Processing multiple prediction requests simultaneously for efficiency

Example implementation:
    Creating a simple forecasting model:

    >>> from openstef_models.models.forecasting.mixins import HorizonForecasterMixin
    >>> from openstef_core.datasets.validated_datasets import ForecastDataset
    >>> import pandas as pd
    >>>
    >>> class SimpleForecaster(HorizonForecasterMixin):
    ...     def __init__(self, config):
    ...         self.config = config
    ...         self._fitted = False
    ...
    ...     @property
    ...     def is_fitted(self):
    ...         return self._fitted
    ...
    ...     def fit_horizon(self, input_data):
    ...         # Train on the data
    ...         self._fitted = True
    ...
    ...     def predict_horizon(self, input_data):
    ...         # Generate predictions
    ...         return ForecastDataset(...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ForecastError
from openstef_core.types import LeadTime, Quantile


class ForecasterHyperParams(BaseConfig):
    """Base configuration for forecasting model hyperparameters.

    Serves as the foundation for model-specific hyperparameter configurations.
    Inheriting classes should add their specific parameters as Pydantic fields
    with appropriate validation and documentation.

    Example:
        Creating custom hyperparameters for a specific model:

        >>> from pydantic import Field
        >>> class MyModelHyperParams(ForecasterHyperParams):
        ...     learning_rate: float = Field(default=0.01, gt=0, description="Learning rate for training")
        ...     max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    """


class ForecasterConfig(BaseConfig):
    """Configuration for forecasting models with support for multiple quantiles and horizons.

    Defines the operational parameters that control forecasting behavior across different
    prediction horizons and uncertainty levels. This configuration is used by both
    single-horizon and multi-horizon forecasting models.

    The configuration enforces that at least one quantile and one horizon must be specified,
    ensuring that every forecaster produces meaningful output. Models use these parameters
    to determine what predictions to generate and how to structure their outputs.

    Attributes:
        quantiles: Probability levels for uncertainty estimation. Each quantile represents
            a confidence level (e.g., 0.1 = 10th percentile, 0.5 = median, 0.9 = 90th percentile).
            Models must generate predictions for all specified quantiles.
        horizons: Lead times for predictions, typically expressed as time offsets from the
            forecast start time. Each horizon defines how far ahead the model should predict.

    Example:
        Basic configuration for daily energy forecasting:

        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> config = ForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)]
        ... )
        >>> isinstance(config.max_horizon, LeadTime)
        True
    """

    quantiles: list[Quantile] = Field(
        default=[Quantile(0.5)],
        description="List of quantiles that the forecaster will predict.",
        min_length=1,
    )

    horizons: list[LeadTime] = Field(
        default=...,
        description="List of lead times (horizons) that the forecaster will predict.",
        min_length=1,
    )

    @property
    def max_horizon(self) -> LeadTime:
        """Returns the maximum lead time (horizon) from the configured horizons.

        Useful for determining the furthest prediction distance required by the model.
        This is commonly used for data preparation and validation logic.

        Returns:
            The maximum lead time.
        """
        return max(self.horizons)


class HorizonForecasterConfig(ForecasterConfig):
    """Configuration for single-horizon forecasting models.

    Specialized configuration that restricts forecasters to operate on exactly one horizon
    at a time. This is used by models that need to be trained and predict for specific
    lead times separately, rather than handling multiple horizons simultaneously.

    The configuration enforces that exactly one horizon is specified, making it suitable
    for use with HorizonForecasterMixin implementations that focus on single-horizon
    predictions.

    Example:
        Configuration for a 6-hour ahead forecaster:

        >>> from datetime import timedelta
        >>> config = HorizonForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[timedelta(hours=6)]
        ... )
        >>> new_config = config.with_horizon(timedelta(hours=24))
        >>> len(new_config.horizons)
        1
    """

    horizons: list[LeadTime] = Field(
        default=...,
        max_length=1,
    )

    def with_horizon(self, horizon: LeadTime) -> Self:
        """Create a new configuration with a different horizon.

        Useful for creating multiple forecaster instances for different prediction
        horizons from a single base configuration. This is commonly used in
        multi-horizon adapters that need separate models for each lead time.

        Args:
            horizon: The new lead time to use for predictions.

        Returns:
            New configuration instance with the specified horizon.
        """
        return self.model_copy(update={"horizons": [horizon]})


type ModelState = object


@dataclass
class BatchResult[T]:
    """Container for batch operation results with error tracking.

    Provides a structured way to return results from batch operations where some
    individual items may succeed while others fail. This is essential for robust
    batch processing where partial failures should not abort the entire operation.

    The length of results and errors lists must always be equal to the number of
    input items, with None values indicating failures at specific positions.

    Attributes:
        results: List of successful results, with None for failed items.
        errors: List of error information, with None for successful items.

    Example:
        Processing a batch where some items fail:

        >>> from openstef_core.exceptions import ForecastError
        >>> # Mock some example predictions
        >>> prediction1 = "forecast_1"
        >>> prediction3 = "forecast_3"
        >>> batch_result = BatchResult(
        ...     results=[prediction1, None, prediction3],
        ...     errors=[None, ForecastError("Invalid data"), None]
        ... )
        >>> successful_count = sum(1 for r in batch_result.results if r is not None)
        >>> successful_count
        2
    """

    results: list[T | None]
    errors: list[ForecastError | None]


class StatefulForecasterMixin(ABC):
    """Mixin for forecasters that can save and restore their internal state.

    Enables model persistence by providing serialization capabilities. Implementations
    must handle state management in a way that allows models to be saved to disk,
    transmitted over networks, or stored in databases, then restored to their exact
    previous state.

    State typically includes trained parameters, configuration, and any learned
    patterns. The state format should be JSON-serializable for maximum compatibility.

    Guarantees:
        - get_state() followed by from_state() should restore identical behavior
        - State format should be forward-compatible across minor version updates
        - State should include version information for compatibility checking

    Example:
        Basic state management implementation:

        >>> class MyForecaster(StatefulForecasterMixin):
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
    """

    @abstractmethod
    def get_state(self) -> ModelState:
        """Serialize the current state of the forecaster.

        Must capture all information needed to restore the model to its current state,
        including configuration, trained parameters, and any internal state variables.

        Returns:
            Serializable representation of the model state. Should be JSON-compatible
            for maximum portability.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state(cls, state: ModelState) -> Self:
        """Restore a forecaster from its serialized state.

        Must reconstruct the forecaster to match the exact state when get_state()
        was called. Should handle version compatibility and graceful degradation
        for older state formats when possible.

        Args:
            state: Serialized state returned from get_state().

        Returns:
            Forecaster instance restored to the exact previous state.

        Raises:
            ModelLoadingError: If state is invalid or incompatible.
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class BaseForecasterMixin(ABC):
    """Foundation mixin providing essential forecaster capabilities and metadata.

    Establishes the basic contract that all forecasting models must implement,
    including state management, configuration access, and capability flags.
    This mixin focuses on operational aspects rather than prediction logic.

    Key responsibilities:
        - Provide access to model configuration and hyperparameters
        - Report training and capability status
        - Enable runtime behavior customization

    Implementing classes should override abstract properties and can customize
    default behaviors by overriding non-abstract properties.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model has been trained and is ready for predictions.

        This property should return True only when the model has been successfully
        trained and contains all necessary parameters to generate forecasts.
        It's used to prevent predictions on untrained models.

        Returns:
            True if model is trained and ready, False otherwise.

        Example:
            Typical implementation pattern:

            >>> class MyForecaster(BaseForecasterMixin):
            ...     def __init__(self):
            ...         self._trained_params = None
            ...
            ...     @property
            ...     def is_fitted(self):
            ...         return self._trained_params is not None
        """
        raise NotImplementedError

    @property
    def requires_fitting(self) -> bool:
        """Indicate whether this model type requires training before prediction.

        Most forecasting models require training, but some (like simple baselines)
        may not need explicit fitting. This property allows the system to skip
        training steps for models that don't need them.

        Returns:
            True if model needs training, False if it can predict immediately.
        """
        return True

    @property
    def supports_batching(self) -> bool:
        """Indicate whether this model can process multiple inputs efficiently.

        Models that support batching can process multiple prediction requests
        simultaneously, which is typically more efficient than individual calls.
        This is especially important for GPU-based models.

        Returns:
            True if model supports batch operations, False for individual-only processing.
        """
        return False

    @property
    def config(self) -> ForecasterConfig:
        """Access the model's configuration parameters.

        Provides access to the configuration object that defines the model's
        operational parameters including quantiles, horizons, and other settings.

        Returns:
            Configuration object containing model parameters.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def hyperparams(self) -> ForecasterHyperParams:
        """Access the model's hyperparameters for training and prediction.

        Hyperparameters control model behavior during training and inference.
        Default implementation returns empty hyperparameters, which is suitable
        for models without configurable parameters.

        Returns:
            Hyperparameter configuration object.
        """
        return ForecasterHyperParams()


class HorizonForecasterMixin(BaseForecasterMixin, StatefulForecasterMixin, ABC):
    """Mixin for forecasters that predict one specific horizon at a time.

    Designed for models that are trained and operate on a single prediction horizon.
    This is common for models that need specialized training for each lead time,
    such as those that adapt their features or parameters based on prediction distance.

    These forecasters are typically used as building blocks in multi-horizon systems
    where separate models handle different prediction distances.

    Key guarantees:
        - fit_horizon() must be called before predict_horizon() for the same horizon
        - predict_horizon() should only be called with data for the trained horizon
        - Predictions must include all quantiles specified in the configuration
        - State management enables saving/loading trained models

    Example:
        Basic implementation for a simple horizon-specific model:

        >>> class SimpleHorizonForecaster(HorizonForecasterMixin):
        ...     def __init__(self, config: HorizonForecasterConfig):
        ...         self._config = config
        ...         self._model_params = None
        ...
        ...     @property
        ...     def config(self):
        ...         return self._config
        ...
        ...     @property
        ...     def is_fitted(self):
        ...         return self._model_params is not None
        ...
        ...     def fit_horizon(self, input_data):
        ...         # Train model for the specific horizon
        ...         self._model_params = self._extract_patterns(input_data)
        ...
        ...     def predict_horizon(self, input_data):
        ...         # Generate predictions for the trained horizon
        ...         return self._generate_forecasts(input_data)
    """

    def fit_horizon(self, input_data: ForecastInputDataset) -> None:
        """Train the model for a specific prediction horizon.

        Prepares the model to generate forecasts for the horizon specified in the
        configuration. The training data should contain sufficient historical information
        to learn patterns relevant to the target prediction distance.

        Args:
            input_data: Historical data for training, including features and targets.
                Must contain enough history to support the model's learning requirements.

        Raises:
            NotImplementedError: Default implementation raises this error.

        Note:
            Most implementations should override this method. The default raises
            NotImplementedError to indicate models that don't require training.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_horizon(self, input_data: ForecastInputDataset) -> ForecastDataset:
        """Generate forecasts for the trained horizon.

        Produces probabilistic forecasts across all configured quantiles for the
        specific horizon this model was trained on. The model must be fitted before
        calling this method.

        Args:
            input_data: Current data for generating predictions, including recent
                history and features needed for forecasting.

        Returns:
            Forecasts containing predictions for all configured quantiles.
            The output must include columns for each quantile formatted as strings.

        Raises:
            ModelNotFittedError: If the model hasn't been trained yet.
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def predict_horizon_batch(self, input_data: list[ForecastInputDataset]) -> BatchResult[ForecastDataset]:
        """Generate forecasts for multiple input datasets efficiently.

        Processes multiple prediction requests in a single call, which can be more
        efficient than individual predict_horizon() calls for models that support
        batch operations.

        Args:
            input_data: List of datasets for batch prediction.

        Returns:
            Results containing forecasts and errors for each input dataset.
            Successful predictions are in results list, failures in errors list.

        Raises:
            NotImplementedError: Default implementation for models without batch support.
        """
        raise NotImplementedError


class ForecasterMixin(BaseForecasterMixin, StatefulForecasterMixin, ABC):
    """Mixin for forecasters that handle multiple horizons simultaneously.

    Designed for models that can train and predict across multiple prediction horizons
    in a unified manner. These models typically handle the complexity of different
    lead times internally, providing a simpler interface for multi-horizon forecasting.

    This approach is efficient for models that can share parameters or features across
    horizons, avoiding the need to train separate models for each prediction distance.

    Key guarantees:
        - fit() must be called with data for all required horizons before prediction
        - predict() should handle all horizons specified in the configuration
        - Output must combine forecasts across all horizons into a single dataset
        - State management preserves multi-horizon capabilities

    Example:
        Implementation for a model that handles multiple horizons:

        >>> class MultiHorizonForecaster(ForecasterMixin):
        ...     def __init__(self, config: ForecasterConfig):
        ...         self._config = config
        ...         self._fitted = False
        ...
        ...     @property
        ...     def config(self):
        ...         return self._config
        ...
        ...     @property
        ...     def is_fitted(self):
        ...         return self._fitted
        ...
        ...     def fit(self, input_data):
        ...         # Train on data for all horizons
        ...         self._fitted = True
        ...
        ...     def predict(self, input_data):
        ...         # Generate predictions for all horizons
        ...         return self._combine_horizon_forecasts(input_data)
    """

    def fit(self, input_data: dict[LeadTime, ForecastInputDataset]) -> None:
        """Train the model using data for multiple horizons.

        Prepares the model to generate forecasts across all configured horizons.
        The input provides separate datasets for each lead time, allowing the model
        to learn horizon-specific patterns or shared representations.

        Args:
            input_data: Mapping from lead times to corresponding training datasets.
                Must include data for all horizons specified in the configuration.

        Raises:
            NotImplementedError: Default implementation for models that don't require training.

        Note:
            Models that don't require training can leave this as the default implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: dict[LeadTime, ForecastInputDataset]) -> ForecastDataset:
        """Generate forecasts for all configured horizons.

        Produces a unified forecast dataset containing predictions across all
        configured lead times and quantiles. The model must be fitted before
        calling this method.

        Args:
            input_data: Mapping from lead times to corresponding input datasets.
                Must include data for all horizons specified in the configuration.

        Returns:
            Combined forecasts containing predictions for all configured horizons
            and quantiles in a single dataset.

        Raises:
            ModelNotFittedError: If the model hasn't been trained yet.
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def predict_batch(self, input_data: list[dict[LeadTime, ForecastInputDataset]]) -> BatchResult[ForecastDataset]:
        """Generate forecasts for multiple multi-horizon input datasets.

        Processes multiple multi-horizon prediction requests efficiently in a single
        call. Each input item contains data for all configured horizons.

        Args:
            input_data: List of multi-horizon datasets for batch prediction.

        Returns:
            Results containing forecasts and errors for each multi-horizon input.
            Successful predictions are in results list, failures in errors list.

        Raises:
            NotImplementedError: Default implementation for models without batch support.
        """
        raise NotImplementedError
