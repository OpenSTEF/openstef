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
from typing import override

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_core.types import Quantile
from openstef_meta.framework.base_learner import (
    BaseLearner,
    BaseLearnerHyperParams,
)
from openstef_meta.framework.final_learner import FinalLearner
from openstef_meta.framework.meta_forecaster import (
    EnsembleForecaster,
)
from openstef_models.models.forecasting.forecaster import (
    Forecaster,
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams

logger = logging.getLogger(__name__)


class StackingFinalLearner(FinalLearner):
    """Combines base learner predictions per quantile into final predictions using a regression approach."""

    def __init__(self, forecaster: Forecaster, feature_adders: None = None) -> None:
        """Initialize the Stacking final learner.

        Args:
            forecaster: The forecaster model to be used as the final learner.
            feature_adders: Placeholder for future feature adders (not yet implemented).
        """
        # Feature adders placeholder for future use
        if feature_adders is not None:
            raise NotImplementedError("Feature adders are not yet implemented.")

        # Split forecaster per quantile
        self.quantiles = forecaster.config.quantiles
        models: list[Forecaster] = []
        for q in self.quantiles:
            config = forecaster.config.model_copy(
                update={
                    "quantiles": [q],
                }
            )
            model = forecaster.__class__(config=config)
            models.append(model)
        self.models = models

    @override
    def fit(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> None:
        for i, q in enumerate(self.quantiles):
            self.models[i].fit(data=base_learner_predictions[q], data_val=None)

    @override
    def predict(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions = [
            self.models[i].predict(data=base_learner_predictions[q]).data for i, q in enumerate(self.quantiles)
        ]

        # Concatenate predictions along columns to form a DataFrame with quantile columns
        df = pd.concat(predictions, axis=1)

        return ForecastDataset(
            data=df,
            sample_interval=base_learner_predictions[self.quantiles[0]].sample_interval,
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

    final_hyperparams: BaseLearnerHyperParams = Field(
        default=GBLinearHyperParams(),
        description="Hyperparameters for the final learner. Defaults to GBLinearHyperParams.",
    )

    use_classifier: bool = Field(
        default=True,
        description="Whether to use sample weights when fitting base and final learners. Defaults to False.",
    )

    add_rolling_accuracy_features: bool = Field(
        default=False,
        description="Whether to add rolling accuracy features from base learners as additional features "
        "to the final learner. Defaults to False.",
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

        final_forecaster = self._init_base_learners(
            config=config, base_hyperparams=[config.hyperparams.final_hyperparams]
        )[0]
        self._final_learner = StackingFinalLearner(forecaster=final_forecaster)


__all__ = ["StackingFinalLearner", "StackingForecaster", "StackingForecasterConfig", "StackingHyperParams"]
