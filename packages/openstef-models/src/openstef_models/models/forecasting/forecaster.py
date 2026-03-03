# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
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

from abc import ABC, abstractmethod
from typing import Self

from pydantic import ConfigDict, Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import BatchPredictor, HyperParams
from openstef_core.types import LeadTime, Quantile


class Forecaster(BaseConfig, BatchPredictor[ForecastInputDataset, ForecastDataset], ABC):
    """Base for forecasters that handle multiple horizons simultaneously.

    Designed for models that train and predict across multiple prediction horizons
    in a unified manner. These models handle the complexity of different lead times
    internally, providing a simpler interface for multi-horizon forecasting.

    Ideal for models that can share parameters or features across horizons, avoiding
    the need to train separate models for each prediction distance.

    Concrete forecasters subclass this directly and declare their fields (quantiles,
    horizons, hyperparams, etc.) as Pydantic model fields. Mutable runtime state
    (e.g. the underlying ML model) should be stored in ``PrivateAttr`` fields and
    initialised in ``model_post_init``.

    Invariants:
        - Predictions must include all quantiles specified in the configuration
        - predict_batch() only called when supports_batching returns True

    Example:
        Creating a forecaster with multiple horizons:

        >>> from openstef_core.types import LeadTime, Quantile
        >>> from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster
        >>> forecaster = FlatlinerForecaster(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT6H")],
        ... )
        >>> len(forecaster.horizons)
        2
        >>> str(forecaster.max_horizon)
        'PT6H'

    See Also:
        XGBoostForecaster: Tree-based forecaster using XGBoost.
        GBLinearForecaster: Linear forecaster using XGBoost's gblinear booster.
        LGBMForecaster: Tree-based forecaster using LightGBM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        Args:
            horizon: The new lead time to use for predictions.

        Returns:
            New configuration instance with the specified horizon.
        """
        return self.model_copy(update={"horizons": [horizon]})

    @property
    @abstractmethod
    def hparams(self) -> HyperParams:
        """Model hyperparameters for training and prediction.

        Concrete forecasters implement this by returning their narrowed
        ``hyperparams`` field, giving callers a polymorphic read-only view
        while each subclass keeps full type safety on its own field.
        """


__all__ = [
    "Forecaster",
]
