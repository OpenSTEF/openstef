# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd
from pydantic import ConfigDict, Field

from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.mixins import HyperParams, TransformPipeline
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import Quantile
from openstef_meta.transforms.selector import Selector
from openstef_models.utils.feature_selection import FeatureSelection

WEATHER_FEATURES = {
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "load",
}

SELECTOR = (
    Selector(
        selection=FeatureSelection.NONE,
    ),
)
from openstef_models.transforms.general import Flagger


class FinalLearnerHyperParams(HyperParams):
    """Hyperparameters for the Final Learner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_adders: list[TimeSeriesTransform] = Field(
        default=[Flagger()],
        description="Additional features to add to the base learner predictions before fitting the final learner.",
    )


class FinalLearner(ABC):
    """Combines base learner predictions for each quantile into final predictions."""

    def __init__(self, quantiles: list[Quantile], hyperparams: FinalLearnerHyperParams) -> None:
        """Initialize the Final Learner."""
        self.quantiles = quantiles
        self.hyperparams = hyperparams
        self.final_learner_processing: TransformPipeline[TimeSeriesDataset] = TransformPipeline(
            transforms=hyperparams.feature_adders
        )
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        base_learner_predictions: dict[Quantile, ForecastInputDataset],
        additional_features: ForecastInputDataset | None,
        sample_weights: pd.Series | None = None,
    ) -> None:
        """Fit the final learner using base learner predictions.

        Args:
            base_learner_predictions: Dictionary mapping Quantiles to ForecastInputDatasets containing base learner
            predictions.
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.
            sample_weights: Optional series of sample weights for fitting.
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(
        self,
        base_learner_predictions: dict[Quantile, ForecastInputDataset],
        additional_features: ForecastInputDataset | None,
    ) -> ForecastDataset:
        """Generate final predictions based on base learner predictions.

        Args:
            base_learner_predictions: Dictionary mapping Quantiles to ForecastInputDatasets containing base learner
                predictions.
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.

        Returns:
            ForecastDataset containing the final predictions.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    def calculate_features(self, data: ForecastInputDataset) -> ForecastInputDataset:
        """Calculate additional features for the final learner.

        Args:
            data: Input TimeSeriesDataset to calculate features on.

        Returns:
            TimeSeriesDataset with additional features.
        """
        data_transformed = self.final_learner_processing.transform(data)

        return ForecastInputDataset(
            data=data_transformed.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.forecast_start,
        )

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the final learner has been fitted."""
        raise NotImplementedError("Subclasses must implement the is_fitted property.")

    @property
    def has_features(self) -> bool:
        """Indicates whether the final learner uses additional features."""
        return len(self.final_learner_processing.transforms) > 0
