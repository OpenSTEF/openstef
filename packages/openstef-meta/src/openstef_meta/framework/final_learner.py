# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

from abc import ABC, abstractmethod

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import HyperParams
from openstef_core.types import Quantile


class FinalLearnerHyperParams(HyperParams):
    """Hyperparameters for the Final Learner."""


class FinalLearnerConfig:
    """Configuration for the Final Learner."""


class FinalLearner(ABC):
    """Combines base learner predictions for each quantile into final predictions."""

    @abstractmethod
    def fit(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> None:
        """Fit the final learner using base learner predictions.

        Args:
            base_learner_predictions: Dictionary mapping Quantiles to ForecastInputDatasets containing base learner
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> ForecastDataset:
        """Generate final predictions based on base learner predictions.

        Args:
            base_learner_predictions: Dictionary mapping Quantiles to ForecastInputDatasets containing base learner
                predictions.

        Returns:
            ForecastDataset containing the final predictions.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the final learner has been fitted."""
        raise NotImplementedError("Subclasses must implement the is_fitted property.")
