# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Forecast combiner base classes.

Provides abstract base classes for implementing forecast combiners that aggregate
predictions from multiple base forecasters.
"""

from abc import ABC, abstractmethod
from typing import Self

from pydantic import ConfigDict, Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset
from openstef_core.mixins import Predictor
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.explainability.mixins import ExplainableForecaster


class ForecastCombiner(BaseConfig, Predictor[EnsembleForecastDataset, ForecastDataset], ExplainableForecaster, ABC):
    """Combines base Forecaster predictions for each quantile into final predictions.

    Subclasses implement specific combination strategies (stacking, learned weights,
    etc.).  The combiner IS its own config — fields like ``quantiles`` and ``horizons``
    are declared as pydantic fields directly.
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

    @property
    def max_horizon(self) -> LeadTime:
        """Returns the maximum lead time (horizon) from the configured horizons."""
        return max(self.horizons)

    def with_horizon(self, horizon: LeadTime) -> Self:
        """Create a copy with a different horizon.

        Args:
            horizon: The new lead time to use for predictions.

        Returns:
            New instance with the specified horizon.
        """
        return self.model_copy(update={"horizons": [horizon]})

    @property
    @abstractmethod
    def hparams(self) -> HyperParams:
        """Combiner hyperparameters.

        Concrete combiners implement this by returning their narrowed
        ``hyperparams`` field, giving callers a polymorphic read-only view
        while each subclass keeps full type safety on its own field.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the combiner has been fitted."""

    @abstractmethod
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        """Fit the combiner using base forecaster predictions.

        Args:
            data: Ensemble dataset with base forecaster predictions.
            data_val: Optional validation data.
            additional_features: Optional additional features for the combiner.
        """

    @abstractmethod
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        """Generate final predictions based on base forecaster predictions.

        Args:
            data: Ensemble dataset with base forecaster predictions.
            additional_features: Optional additional features for the combiner.

        Returns:
            Combined forecast dataset.
        """

    @abstractmethod
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> TimeSeriesDataset:
        """Per-sample feature/model contributions for the combined prediction.

        Args:
            data: Ensemble dataset with base forecaster predictions.
            additional_features: Optional additional features for the combiner.

        Returns:
            TimeSeriesDataset where columns are features/models and rows are timesteps.
        """
