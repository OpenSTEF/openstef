"""Core forecasting model interfaces and configurations.

Provides the fundamental building blocks for implementing forecasting models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different model types
while supporting both single and multi-horizon forecasting scenarios.

Key concepts:
- **Horizon**: The lead time for predictions, accounting for data availability and versioning cutoffs
- **Quantiles**: Probability levels for uncertainty estimation
- **State**: Serializable model parameters that enable saving/loading trained models
- **Batching**: Processing multiple prediction requests simultaneously for efficiency

Multi-horizon forecasting considerations:
Some models (like linear models) cannot handle missing data or conditional features effectively,
making them suitable only for single-horizon approaches. Other models (like XGBoost) can
handle such data complexities and work well for multi-horizon scenarios.
"""

from abc import ABC, abstractmethod
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

    Fundamental configuration parameters that determine forecasting model behavior across
    different prediction horizons and uncertainty levels. These are operational parameters
    rather than hyperparameters that affect training.

    Example:
        Basic configuration for daily energy forecasting:

        >>> from openstef_core.types import LeadTime, Quantile
        >>> config = ForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT6H"), LeadTime.from_string("PT24H")]
        ... )
        >>> len(config.horizons)
        3
        >>> str(config.max_horizon)
        'P1D'
    """

    quantiles: list[Quantile] = Field(
        default=[Quantile(0.5)],
        description=(
            "Probability levels for uncertainty estimation. Each quantile represents a confidence level "
            "(e.g., 0.1 = 10th percentile, 0.5 = median, 0.9 = 90th percentile). "
            "Models must generate predictions for all specified quantiles."
        ),
        min_length=1,
    )

    horizons: list[LeadTime] = Field(
        default=...,
        description=(
            "Lead times for predictions, accounting for data availability and versioning cutoffs. "
            "Each horizon defines how far ahead the model should predict."
        ),
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
    at a time. Used by models that need to be trained and predict for specific lead times
    separately, such as those that cannot handle missing data or conditional features.

    Example:
        Configuration for a 6-hour ahead forecaster:

        >>> config = HorizonForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime.from_string("PT6H")]
        ... )
        >>> new_config = config.with_horizon(LeadTime.from_string("PT24H"))
        >>> str(new_config.horizons[0])
        'P1D'
    """

    horizons: list[LeadTime] = Field(
        default=...,
        max_length=1,
        description="Single horizon for prediction. Must contain exactly one lead time.",
    )

    def with_horizon(self, horizon: LeadTime) -> Self:
        """Create a new configuration with a different horizon.

        Useful for creating multiple forecaster instances for different prediction
        horizons from a single base configuration.

        Args:
            horizon: The new lead time to use for predictions.

        Returns:
            New configuration instance with the specified horizon.
        """
        return self.model_copy(update={"horizons": [horizon]})


type ModelState = object


type BatchResult[T] = list[T | ForecastError]


class StatefulForecasterMixin(ABC):
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


class BaseForecasterMixin(ABC):
    """Foundation mixin providing essential forecaster capabilities and metadata.

    Establishes the basic contract that all forecasting models must implement,
    including configuration access and capability flags.

    Key responsibilities:
        - Provide access to model configuration and hyperparameters
        - Report training and capability status
        - Define runtime behavior capabilities

    Implementing classes should override abstract properties and can customize
    default behaviors by overriding non-abstract properties.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model has been trained and is ready for predictions.

        Used to prevent predictions on untrained models. For models that don't require
        training, this should always return True.

        Invariants:
            - fit() methods will not be called if this returns True
            - predict() methods should only be called when this returns True

        Returns:
            True if model is trained and ready, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement is_fitted")

    @property
    def supports_batching(self) -> bool:
        """Indicate whether this model supports batching predictions.

        Models that support batching can process multiple prediction requests
        simultaneously, which is typically more efficient than individual calls.
        This is especially important for GPU-based models.

        Invariants:
            - predict_batch methods will only be called if this returns True
            - If True, batch methods must be implemented and functional

        Returns:
            True if model supports batch operations, False for individual-only processing.
        """
        return False

    @property
    @abstractmethod
    def config(self) -> ForecasterConfig:
        """Access the model's configuration parameters.

        Returns:
            Configuration object containing fundamental model parameters.
        """
        raise NotImplementedError("Subclasses must implement config")

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


class BaseHorizonForecaster(BaseForecasterMixin, StatefulForecasterMixin, ABC):
    """Mixin for forecasters that predict one specific horizon at a time.

    Designed for models that operate on a single prediction horizon. Common for models
    that need specialized training for each lead time, such as those that cannot handle
    missing data or conditional features.

    These forecasters are typically used as building blocks in multi-horizon systems
    where separate models handle different prediction distances.

    Invariants:
        - fit_horizon() must be called before predict_horizon() for the same horizon
        - predict_horizon() should only be called when is_fitted returns True
        - Predictions must include all quantiles specified in the configuration
        - predict_horizon_batch() only called when supports_batching returns True

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
        configuration. Only called when is_fitted returns False.

        Args:
            input_data: Historical data for training, including features and targets.
        """

    @abstractmethod
    def predict_horizon(self, input_data: ForecastInputDataset) -> ForecastDataset:
        """Generate forecasts for the trained horizon.

        Produces probabilistic forecasts across all configured quantiles for the
        specific horizon this model was trained on.

        Args:
            input_data: Current data for generating predictions, including recent
                history and features needed for forecasting.

        Returns:
            Forecasts containing predictions for all configured quantiles.

        Raises:
            ModelNotFittedError: If the model hasn't been trained yet.
        """
        raise NotImplementedError("Subclasses must implement predict_horizon")

    def predict_horizon_batch(self, input_data: list[ForecastInputDataset]) -> BatchResult[ForecastDataset]:
        """Generate forecasts for multiple input datasets efficiently.

        Processes multiple prediction requests in a single call. Only called when
        supports_batching returns True.

        Args:
            input_data: List of datasets for batch prediction.

        Returns:
            Results containing forecasts and errors for each input dataset.
        """
        raise NotImplementedError("Models supporting batching must implement predict_horizon_batch")


class BaseForecaster(BaseForecasterMixin, StatefulForecasterMixin, ABC):
    """Mixin for forecasters that handle multiple horizons simultaneously.

    Designed for models that train and predict across multiple prediction horizons
    in a unified manner. These models handle the complexity of different lead times
    internally, providing a simpler interface for multi-horizon forecasting.

    Ideal for models that can share parameters or features across horizons, avoiding
    the need to train separate models for each prediction distance.

    Invariants:
        - fit() must be called with data for all required horizons before prediction
        - predict() handles all horizons specified in the configuration
        - Output combines forecasts across all horizons into a single dataset

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

        Note:
            Models that don't require training can leave this as the default implementation.
        """

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
        """
        raise NotImplementedError


__all__ = [
    "BaseForecaster",
    "BaseForecasterMixin",
    "BaseHorizonForecaster",
    "ForecasterConfig",
    "ForecasterHyperParams",
    "HorizonForecasterConfig",
    "StatefulForecasterMixin",
]