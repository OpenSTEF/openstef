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
from abc import abstractmethod
from typing import override

import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import Field
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

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
from openstef_meta.utils.pinball_errors import calculate_pinball_errors
from openstef_models.models.forecasting.forecaster import (
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams

logger = logging.getLogger(__name__)


Classifier = LGBMClassifier | XGBClassifier | LinearRegression


class LearnedWeightsFinalLearner(FinalLearner):
    """Combines base learner predictions with a classification approach to determine which base learner to use."""

    @abstractmethod
    def __init__(self, quantiles: list[Quantile]) -> None:
        self.quantiles = quantiles
        self.models: list[Classifier] = []
        self._is_fitted = False

    @override
    def fit(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> None:
        for i, q in enumerate(self.quantiles):
            pred = base_learner_predictions[q].data.drop(columns=[base_learner_predictions[q].target_column])
            labels = self._prepare_classification_data(
                quantile=q,
                target=base_learner_predictions[q].target_series,
                predictions=pred,
            )

            self.models[i].fit(X=pred, y=labels)  # type: ignore
        self._is_fitted = True

    @staticmethod
    def _prepare_classification_data(quantile: Quantile, target: pd.Series, predictions: pd.DataFrame) -> pd.Series:
        """Selects base learner with lowest error for each sample as target for classification.

        Returns:
            pd.Series: Series indicating the base learner with the lowest pinball loss for each sample.
        """

        # Calculate pinball loss for each base learner
        def column_pinball_losses(preds: pd.Series) -> pd.Series:
            return calculate_pinball_errors(y_true=target, y_pred=preds, alpha=quantile)

        pinball_losses = predictions.apply(column_pinball_losses)

        # For each sample, select the base learner with the lowest pinball loss
        return pinball_losses.idxmin(axis=1)

    def _calculate_sample_weights_quantile(self, base_predictions: pd.DataFrame, quantile: Quantile) -> pd.DataFrame:
        model = self.models[self.quantiles.index(quantile)]

        return model.predict_proba(X=base_predictions)  # type: ignore

    def _generate_predictions_quantile(self, base_predictions: ForecastInputDataset, quantile: Quantile) -> pd.Series:
        df = base_predictions.data.drop(columns=[base_predictions.target_column])
        weights = self._calculate_sample_weights_quantile(base_predictions=df, quantile=quantile)

        return df.mul(weights).sum(axis=1)

    @override
    def predict(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions = pd.DataFrame({
            Quantile(q).format(): self._generate_predictions_quantile(base_predictions=data, quantile=q)
            for q, data in base_learner_predictions.items()
        })

        return ForecastDataset(
            data=predictions,
            sample_interval=base_learner_predictions[self.quantiles[0]].sample_interval,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class LGBMFinalLearner(LearnedWeightsFinalLearner):
    """Final learner using only LGBM as base learners."""

    def __init__(self, quantiles: list[Quantile], n_estimators: int = 20) -> None:
        self.quantiles = quantiles
        self.models = [LGBMClassifier(class_weight="balanced", n_estimators=n_estimators) for _ in quantiles]
        self._is_fitted = False


class RandomForestFinalLearner(LearnedWeightsFinalLearner):
    def __init__(self, quantiles: list[Quantile], n_estimators: int = 20) -> None:
        self.quantiles = quantiles
        self.models = [
            LGBMClassifier(boosting_type="rf", class_weight="balanced", n_estimators=n_estimators) for _ in quantiles
        ]
        self._is_fitted = False


class XGBFinalLearner(LearnedWeightsFinalLearner):
    """Final learner using only XGBoost as base learners."""

    def __init__(self, quantiles: list[Quantile], n_estimators: int = 20) -> None:
        self.quantiles = quantiles
        self.models = [XGBClassifier(class_weight="balanced", n_estimators=n_estimators) for _ in quantiles]
        self._is_fitted = False


class LogisticRegressionFinalLearner(LearnedWeightsFinalLearner):
    """Final learner using only Logistic Regression as base learners."""

    def __init__(self, quantiles: list[Quantile]) -> None:
        self.quantiles = quantiles
        self.models = [LinearRegression() for _ in quantiles]
        self._is_fitted = False


class LearnedWeightsHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    base_hyperparams: list[BaseLearnerHyperParams] = Field(
        default=[LGBMHyperParams(), GBLinearHyperParams()],
        description="List of hyperparameter configurations for base learners. "
        "Defaults to [LGBMHyperParams, GBLinearHyperParams].",
    )

    final_learner: type[LearnedWeightsFinalLearner] = Field(
        default=LGBMFinalLearner,
        description="Type of final learner to use. Defaults to LearnedWeightsFinalLearner.",
    )


class LearnedWeightsForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: LearnedWeightsHyperParams = LearnedWeightsHyperParams()

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


class LearnedWeightsForecaster(EnsembleForecaster):
    """Wrapper for sklearn's StackingRegressor to make it compatible with HorizonForecaster."""

    Config = LearnedWeightsForecasterConfig
    HyperParams = LearnedWeightsHyperParams

    def __init__(self, config: LearnedWeightsForecasterConfig) -> None:
        """Initialize the LearnedWeightsForecaster."""
        self._config = config

        self._base_learners: list[BaseLearner] = self._init_base_learners(
            config=config, base_hyperparams=config.hyperparams.base_hyperparams
        )
        self._final_learner = config.hyperparams.final_learner(quantiles=config.quantiles)


__all__ = [
    "LGBMFinalLearner",
    "LearnedWeightsForecaster",
    "LearnedWeightsForecasterConfig",
    "LearnedWeightsHyperParams",
    "LogisticRegressionFinalLearner",
    "RandomForestFinalLearner",
    "XGBFinalLearner",
]
