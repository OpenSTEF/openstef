# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Hybrid Forecaster (Stacked LightGBM + Linear Model Gradient Boosting).

Provides method that attempts to combine the advantages of a linear model (Extraplolation)
and tree-based model (Non-linear patterns). This is acieved by training two base learners,
followed by a small linear model that regresses on the baselearners' predictions.
The implementation is based on sklearn's StackingRegressor.
"""

import logging
from collections.abc import Sequence
from typing import override

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import LeadTime, Quantile
from openstef_meta.framework.base_learner import (
    BaseLearner,
    BaseLearnerHyperParams,
)
from openstef_meta.framework.final_learner import FinalLearner, FinalLearnerHyperParams
from openstef_meta.framework.meta_forecaster import (
    EnsembleForecaster,
)
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.models.forecasting.forecaster import (
    Forecaster,
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams

logger = logging.getLogger(__name__)


class StackingFinalLearnerHyperParams(FinalLearnerHyperParams):
    """HyperParams for Stacking Final Learner."""

    feature_adders: Sequence[TimeSeriesTransform] = Field(
        default=[],
        description="Additional features to add to the base learner predictions before fitting the final learner.",
    )

    forecaster_hyperparams: BaseLearnerHyperParams = Field(
        default=GBLinearHyperParams(),
        description="Forecaster hyperparameters for the final learner. Defaults to GBLinearHyperParams.",
    )


class StackingFinalLearner(FinalLearner):
    """Combines base learner predictions per quantile into final predictions using a regression approach."""

    def __init__(
        self, quantiles: list[Quantile], hyperparams: StackingFinalLearnerHyperParams, horizon: LeadTime
    ) -> None:
        """Initialize the Stacking final learner.

        Args:
            quantiles: List of quantiles to predict.
            hyperparams: Hyperparameters for the final learner.
            horizon: Forecast horizon for which to create the final learner.
        """
        super().__init__(quantiles=quantiles, hyperparams=hyperparams)

        forecaster_hyperparams: BaseLearnerHyperParams = hyperparams.forecaster_hyperparams

        # Split forecaster per quantile
        models: list[Forecaster] = []
        for q in self.quantiles:
            forecaster_cls = forecaster_hyperparams.forecaster_class()
            config = forecaster_cls.Config(horizons=[horizon], quantiles=[q])
            if "hyperparams" in forecaster_cls.Config.model_fields:
                config = config.model_copy(update={"hyperparams": forecaster_hyperparams})

            model = config.forecaster_from_config()
            models.append(model)
        self.models = models

    @staticmethod
    def _combine_datasets(
        data: ForecastInputDataset, additional_features: ForecastInputDataset
    ) -> ForecastInputDataset:
        """Combine base learner predictions with additional features for final learner input.

        Args:
            data: ForecastInputDataset containing base learner predictions.
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
        base_predictions: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None,
        sample_weights: pd.Series | None = None,
    ) -> None:

        for i, q in enumerate(self.quantiles):
            if additional_features is not None:
                dataset = base_predictions.select_quantile(quantile=q)
                data = self._combine_datasets(
                    data=dataset,
                    additional_features=additional_features,
                )
            else:
                data = base_predictions.select_quantile(quantile=q)

            self.models[i].fit(data=data, data_val=None)

    @override
    def predict(
        self,
        base_predictions: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None,
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions: list[pd.DataFrame] = []
        for i, q in enumerate(self.quantiles):
            if additional_features is not None:
                data = self._combine_datasets(
                    data=base_predictions.select_quantile(quantile=q),
                    additional_features=additional_features,
                )
            else:
                data = base_predictions.select_quantile(quantile=q)
            p = self.models[i].predict(data=data).data
            predictions.append(p)

        # Concatenate predictions along columns to form a DataFrame with quantile columns
        df = pd.concat(predictions, axis=1)

        return ForecastDataset(
            data=df,
            sample_interval=base_predictions.sample_interval,
        )

    @property
    def is_fitted(self) -> bool:
        """Check the StackingFinalLearner is fitted."""
        return all(x.is_fitted for x in self.models)


class StackingHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    base_hyperparams: list[BaseLearnerHyperParams] = Field(
        default=[LGBMHyperParams(), GBLinearHyperParams()],
        description="List of hyperparameter configurations for base learners. "
        "Defaults to [LGBMHyperParams, GBLinearHyperParams].",
    )

    final_hyperparams: StackingFinalLearnerHyperParams = Field(
        default=StackingFinalLearnerHyperParams(),
        description="Hyperparameters for the final learner.",
    )

    @field_validator("base_hyperparams", mode="after")
    @classmethod
    def _check_classes(cls, v: list[BaseLearnerHyperParams]) -> list[BaseLearnerHyperParams]:
        hp_classes = [type(hp) for hp in v]
        if not len(hp_classes) == len(set(hp_classes)):
            raise ValueError("Duplicate base learner hyperparameter classes are not allowed.")
        return v


class StackingForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: StackingHyperParams = StackingHyperParams()

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


class StackingForecaster(EnsembleForecaster):
    """Wrapper for sklearn's StackingRegressor to make it compatible with HorizonForecaster."""

    Config = StackingForecasterConfig
    HyperParams = StackingHyperParams

    def __init__(self, config: StackingForecasterConfig) -> None:
        """Initialize the Hybrid forecaster."""
        self._config = config

        self._base_learners: list[BaseLearner] = self._init_base_learners(
            config=config, base_hyperparams=config.hyperparams.base_hyperparams
        )

        self._final_learner = StackingFinalLearner(
            quantiles=config.quantiles, hyperparams=config.hyperparams.final_hyperparams, horizon=config.max_horizon
        )


__all__ = ["StackingFinalLearner", "StackingForecaster", "StackingForecasterConfig", "StackingHyperParams"]
