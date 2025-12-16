# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Stacking Forecast Combiner.

This module implements a Stacking Combiner that integrates predictions from multiple base Forecasters.
It uses a regression approach to combine the predictions for each quantile into final forecasts.
"""

import logging
from functools import partial
from typing import TYPE_CHECKING, cast, override

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner, ForecastCombinerConfig
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster, LGBMHyperParams

if TYPE_CHECKING:
    from openstef_models.models.forecasting.forecaster import Forecaster

logger = logging.getLogger(__name__)

ForecasterHyperParams = GBLinearHyperParams | LGBMHyperParams
ForecasterType = GBLinearForecaster | LGBMForecaster


class StackingCombinerConfig(ForecastCombinerConfig):
    """Configuration for the Stacking final learner."""

    hyperparams: HyperParams = Field(
        description="Hyperparameters for the Stacking Combiner.",
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

    @field_validator("hyperparams", mode="after")
    @staticmethod
    def validate_forecaster(
        v: HyperParams,
    ) -> HyperParams:
        """Validate that the forecaster class is set in the hyperparameters.

        Args:
            v: Hyperparameters to validate.

        Returns:
            Validated hyperparameters.

        Raises:
            ValueError: If the forecaster class is not set.
        """
        if not hasattr(v, "forecaster_class"):
            raise ValueError("forecaster_class must be set in hyperparameters for StackingCombinerConfig.")
        return v


class StackingCombiner(ForecastCombiner):
    """Combines base Forecaster predictions per quantile into final predictions using a regression approach."""

    Config = StackingCombinerConfig
    LGBMHyperParams = LGBMHyperParams
    GBLinearHyperParams = GBLinearHyperParams

    def __init__(
        self,
        config: StackingCombinerConfig,
    ) -> None:
        """Initialize the Stacking final learner.

        Args:
            config: Configuration for the Stacking combiner.
        """
        forecaster_hyperparams = cast(ForecasterHyperParams, config.hyperparams)
        self.quantiles = config.quantiles
        self.config = config
        self.hyperparams = forecaster_hyperparams
        self._is_fitted: bool = False

        # Split forecaster per quantile
        models: list[Forecaster] = []
        for q in self.quantiles:
            forecaster_cls = forecaster_hyperparams.forecaster_class()
            forecaster_config = forecaster_cls.Config(
                horizons=[config.max_horizon],
                quantiles=[q],
            )
            if "hyperparams" in forecaster_cls.Config.model_fields:
                forecaster_config = forecaster_config.model_copy(update={"hyperparams": forecaster_hyperparams})

            model = forecaster_config.forecaster_from_config()
            models.append(model)
        self.models = models

    @staticmethod
    def _combine_datasets(
        data: ForecastInputDataset, additional_features: ForecastInputDataset
    ) -> ForecastInputDataset:
        """Combine base Forecaster predictions with additional features for final learner input.

        Args:
            data: ForecastInputDataset containing base Forecaster predictions.
            additional_features: ForecastInputDataset containing additional features.

        Returns:
            ForecastInputDataset with combined features.
        """
        additional_df = additional_features.data.loc[
            :, [col for col in additional_features.data.columns if col not in data.data.columns]
        ]
        # Merge on index to combine datasets
        combined_df = data.data.join(additional_df)

        return ForecastInputDataset(
            data=combined_df,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:

        for i, q in enumerate(self.quantiles):
            if additional_features is not None:
                dataset = data.select_quantile(quantile=q)
                input_data = self._combine_datasets(
                    data=dataset,
                    additional_features=additional_features,
                )
            else:
                input_data = data.select_quantile(quantile=q)

            # Prepare input data by dropping rows with NaN target values
            target_dropna = partial(pd.DataFrame.dropna, subset=[input_data.target_column])  # pyright: ignore[reportUnknownMemberType]
            input_data = input_data.pipe_pandas(target_dropna)

            self.models[i].fit(data=input_data, data_val=None)

    @override
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions: list[pd.DataFrame] = []
        for i, q in enumerate(self.quantiles):
            if additional_features is not None:
                input_data = self._combine_datasets(
                    data=data.select_quantile(quantile=q),
                    additional_features=additional_features,
                )
            else:
                input_data = data.select_quantile(quantile=q)

            if isinstance(self.models[i], GBLinearForecaster):
                feature_cols = [x for x in input_data.data.columns if x != data.target_column]
                feature_dropna = partial(pd.DataFrame.dropna, subset=feature_cols)  # pyright: ignore[reportUnknownMemberType]
                input_data = input_data.pipe_pandas(feature_dropna)

            p = self.models[i].predict(data=input_data).data
            predictions.append(p)

        # Concatenate predictions along columns to form a DataFrame with quantile columns
        df = pd.concat(predictions, axis=1)

        return ForecastDataset(
            data=df,
            sample_interval=data.sample_interval,
        )

    @override
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> pd.DataFrame:

        predictions: list[pd.DataFrame] = []
        for i, q in enumerate(self.quantiles):
            if additional_features is not None:
                input_data = self._combine_datasets(
                    data=data.select_quantile(quantile=q),
                    additional_features=additional_features,
                )
            else:
                input_data = data.select_quantile(quantile=q)
            model = self.models[i]
            if not isinstance(model, ExplainableForecaster):
                raise NotImplementedError(
                    "Predicting contributions is only supported for ExplainableForecaster models."
                )
            p = model.predict_contributions(data=input_data, scale=True)
            predictions.append(p)

        contributions = pd.concat(predictions, axis=1)

        target_series = data.target_series
        if target_series is not None:
            contributions[data.target_column] = target_series

        return contributions

    @property
    def is_fitted(self) -> bool:
        """Check the StackingForecastCombiner is fitted."""
        return all(x.is_fitted for x in self.models)
