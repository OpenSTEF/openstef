# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

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

from abc import abstractmethod
from typing import Self

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import BatchPredictor, HyperParams
from openstef_core.types import LeadTime, Quantile


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

    See Also:
        HorizonForecasterConfig: Single-horizon variant of this configuration.
        BaseForecaster: Multi-horizon forecaster that uses this configuration.
        ForecasterHyperParams: Hyperparameter configuration used alongside this.
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

    supports_batching: bool = Field(
        default=False,
        description=(
            "Indicates if the model can handle batch predictions. "
            "Batching allows multiple prediction requests to be processed simultaneously, "
            "which is more efficient for models that support it, especially on GPUs."
        ),
    )

    hyperparams: HyperParams = Field(
        default=HyperParams(),
        description=(
            "Optional hyperparameter configuration for the forecaster. "
            "These parameters influence model training and prediction behavior."
        ),
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


class ConfigurableForecaster:
    @property
    @abstractmethod
    def config(self) -> ForecasterConfig:
        """Access the model's configuration parameters.

        Returns:
            Configuration object containing fundamental model parameters.
        """
        raise NotImplementedError("Subclasses must implement config")

    @property
    def hyperparams(self) -> HyperParams:
        """Access the model's hyperparameters for training and prediction.

        Hyperparameters control model behavior during training and inference.
        Default implementation returns empty hyperparameters, which is suitable
        for models without configurable parameters.

        Returns:
            Hyperparameter configuration object.
        """
        return HyperParams()


class Forecaster(BatchPredictor[ForecastInputDataset, ForecastDataset], ConfigurableForecaster):
    """Base for forecasters that handle multiple horizons simultaneously.

    Designed for models that train and predict across multiple prediction horizons
    in a unified manner. These models handle the complexity of different lead times
    internally, providing a simpler interface for multi-horizon forecasting.

    Ideal for models that can share parameters or features across horizons, avoiding
    the need to train separate models for each prediction distance.

    Invariants:
        - Predictions must include all quantiles specified in the configuration
        - predict_batch() only called when supports_batching returns True

    Example:
        Implementation for a model that handles multiple horizons:

        >>> from typing import override
        >>> class CustomForecaster(Forecaster):
        ...     def __init__(self, config: ForecasterConfig):
        ...         self._config = config
        ...         self._fitted = False
        ...
        ...     @property
        ...     @override
        ...     def config(self):
        ...         return self._config
        ...
        ...     @property
        ...     @override
        ...     def is_fitted(self):
        ...         return self._fitted
        ...
        ...     @override
        ...     def get_state(self):
        ...         return {"config": self._config, "fitted": self._fitted}
        ...
        ...     @override
        ...     def from_state(self, state):
        ...         instance = self.__class__(state["config"])
        ...         instance._fitted = state["fitted"]
        ...         return instance
        ...
        ...     @override
        ...     def fit(self, input_data, data_val):
        ...         # Train on data for all horizons
        ...         self._fitted = True
        ...
        ...     @override
        ...     def predict(self, input_data):
        ...         # Generate predictions for all horizons
        ...         from openstef_core.datasets.validated_datasets import ForecastDataset
        ...         import pandas as pd
        ...         return ForecastDataset(
        ...             data=pd.DataFrame(),
        ...             sample_interval=pd.Timedelta("15min"),
        ...             forecast_start=pd.Timestamp.now()
        ...         )
    """

    @abstractmethod
    def __init__(self, config: ForecasterConfig) -> None:
        """Initialize the forecaster with the given configuration.

        Args:
            config: Configuration object specifying quantiles, horizons, and batching support.
        """
        raise NotImplementedError("Subclasses must implement __init__")

    @property
    @abstractmethod
    def config(self) -> ForecasterConfig:
        """Access the model's configuration parameters.

        Returns:
            Configuration object containing fundamental model parameters.
        """
        raise NotImplementedError("Subclasses must implement config")


__all__ = [
    "Forecaster",
    "ForecasterConfig",
]
