# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

from abc import abstractmethod
from collections.abc import Sequence

import pandas as pd
from pydantic import ConfigDict, Field

from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.mixins import HyperParams, Predictor, TransformPipeline
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import Quantile
from openstef_meta.transforms.selector import Selector
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.utils.feature_selection import FeatureSelection

SELECTOR = Selector(
    selection=FeatureSelection(include=None),
)


class ForecastCombinerHyperParams(HyperParams):
    """Hyperparameters for the Final Learner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_adders: Sequence[TimeSeriesTransform] = Field(
        default=[],
        description="Additional features to add to the base learner predictions before fitting the final learner.",
    )


class ForecastCombiner(Predictor[EnsembleForecastDataset, ForecastDataset]):
    """Combines base learner predictions for each quantile into final predictions."""

    def __init__(self, quantiles: list[Quantile], hyperparams: ForecastCombinerHyperParams) -> None:
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
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
        sample_weights: pd.Series | None = None,
    ) -> None:
        """Fit the final learner using base learner predictions.

        Args:
            data: EnsembleForecastDataset
            data_val: Optional EnsembleForecastDataset for validation during fitting. Will be ignored
            additional_features: Optional ForecastInputDataset containing additional features for the final learner.
            sample_weights: Optional series of sample weights for fitting.
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

    def calculate_features(self, data: ForecastInputDataset) -> ForecastInputDataset:
        """Calculate additional features for the final learner.

        Args:
            data: Input ForecastInputDataset to calculate features on.

        Returns:
            ForecastInputDataset with additional features.
        """
        data_transformed = self.final_learner_processing.transform(data)

        return ForecastInputDataset(
            data=data_transformed.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.forecast_start,
        )

    @staticmethod
    def _prepare_input_data(
        dataset: ForecastInputDataset, additional_features: ForecastInputDataset | None
    ) -> pd.DataFrame:
        """Prepare input data by combining base predictions with additional features if provided.

        Args:
            dataset: ForecastInputDataset containing base predictions.
            additional_features: Optional ForecastInputDataset containing additional features.

        Returns:
            pd.DataFrame: Combined DataFrame of base predictions and additional features if provided.
        """
        df = dataset.input_data(start=dataset.index[0])
        if additional_features is not None:
            df_a = additional_features.input_data(start=dataset.index[0])
            df = pd.concat(
                [df, df_a],
                axis=1,
            )
        return df

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Indicates whether the final learner has been fitted."""
        raise NotImplementedError("Subclasses must implement the is_fitted property.")

    @property
    def has_features(self) -> bool:
        """Indicates whether the final learner uses additional features."""
        return len(self.final_learner_processing.transforms) > 0
