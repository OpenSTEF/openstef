# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

from abc import abstractmethod
from typing import Self

import pandas as pd
from pydantic import ConfigDict, Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import HyperParams, Predictor
from openstef_core.types import LeadTime, Quantile
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.transforms.general.selector import Selector
from openstef_models.utils.feature_selection import FeatureSelection

SELECTOR = Selector(
    selection=FeatureSelection(include=None),
)


class ForecastCombinerConfig(BaseConfig):
    """Hyperparameters for the Final Learner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hyperparams: HyperParams = Field(
        description="Hyperparameters for the final learner.",
    )

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


class ForecastCombiner(Predictor[EnsembleForecastDataset, ForecastDataset]):
    """Combines base learner predictions for each quantile into final predictions."""

    config: ForecastCombinerConfig

    @abstractmethod
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        """Fit the final learner using base learner predictions.

        Args:
            data: EnsembleForecastDataset
            data_val: Optional EnsembleForecastDataset for validation during fitting. Will be ignored
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        """Generate final predictions based on base learner predictions.

        Args:
            data: EnsembleForecastDataset containing base learner predictions.
            data_val: Optional EnsembleForecastDataset for validation during prediction. Will be ignored
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.

        Returns:
            ForecastDataset containing the final predictions.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the final learner has been fitted."""
        raise NotImplementedError("Subclasses must implement the is_fitted property.")

    @abstractmethod
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> pd.DataFrame:
        """Generate final predictions based on base learner predictions.

        Args:
            data: EnsembleForecastDataset containing base learner predictions.
            data_val: Optional EnsembleForecastDataset for validation during prediction. Will be ignored
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.

        Returns:
            ForecastDataset containing the final contributions.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")
