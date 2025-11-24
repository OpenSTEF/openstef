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
from typing import override, Literal, Self

import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import Field
from sklearn.linear_model import LogisticRegression
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


# Base classes for Learned Weights Final Learner

Classifier = LGBMClassifier | XGBClassifier | LogisticRegression


class LWFLHyperParams(HyperParams):
    """Hyperparameters for Learned Weights Final Learner."""

    @property
    @abstractmethod
    def learner(self) -> type["WeightsLearner"]:
        """Returns the classifier to be used as final learner."""
        raise NotImplementedError("Subclasses must implement the 'estimator' property.")

    @classmethod
    def learner_from_params(cls, quantiles: list[Quantile], hyperparams: Self) -> "WeightsLearner":
        """Initialize the final learner from hyperparameters."""
        instance = cls()
        return instance.learner(quantiles=quantiles, hyperparams=hyperparams)


class WeightsLearner(FinalLearner):
    """Combines base learner predictions with a classification approach to determine which base learner to use."""

    @abstractmethod
    def __init__(self, quantiles: list[Quantile], hyperparams: LWFLHyperParams) -> None:
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


# Final learner implementations using different classifiers
# 1 LGBM Classifier


class LGBMLearnerHyperParams(LWFLHyperParams):
    """Hyperparameters for Learned Weights Final Learner with LGBM Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    n_leaves: int = Field(
        default=31,
        description="Number of leaves for the LGBM Classifier. Defaults to 31.",
    )

    @property
    @override
    def learner(self) -> type["LGBMLearner"]:
        """Returns the LGBMLearner"""
        return LGBMLearner


class LGBMLearner(WeightsLearner):
    """Final learner with LGBM Classifier."""

    HyperParams = LGBMLearnerHyperParams

    def __init__(
        self,
        quantiles: list[Quantile],
        hyperparams: LGBMLearnerHyperParams,
    ) -> None:
        self.quantiles = quantiles
        self.models = [
            LGBMClassifier(
                class_weight="balanced",
                n_estimators=hyperparams.n_estimators,
                num_leaves=hyperparams.n_leaves,
            )
            for _ in quantiles
        ]
        self._is_fitted = False


# 1 RandomForest Classifier
class RFLearnerHyperParams(LWFLHyperParams):
    """Hyperparameters for Learned Weights Final Learner with LGBM Random Forest Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    n_leaves: int = Field(
        default=31,
        description="Number of leaves for the LGBM Classifier. Defaults to 31.",
    )

    @property
    def learner(self) -> type["RandomForestLearner"]:
        """Returns the LGBMClassifier to be used as final learner."""
        return RandomForestLearner


class RandomForestLearner(WeightsLearner):
    """Final learner using only Random Forest as base learners."""

    def __init__(self, quantiles: list[Quantile], hyperparams: RFLearnerHyperParams) -> None:
        """Initialize RandomForestLearner."""
        self.quantiles = quantiles
        self.models = [
            LGBMClassifier(boosting_type="rf", class_weight="balanced", n_estimators=hyperparams.n_estimators)
            for _ in quantiles
        ]
        self._is_fitted = False


# 3 XGB Classifier
class XGBLearnerHyperParams(LWFLHyperParams):
    """Hyperparameters for Learned Weights Final Learner with LGBM Random Forest Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    @property
    def learner(self) -> type["XGBLearner"]:
        """Returns the LGBMClassifier to be used as final learner."""
        return XGBLearner


class XGBLearner(WeightsLearner):
    """Final learner using only XGBoost as base learners."""

    def __init__(self, quantiles: list[Quantile], hyperparams: XGBLearnerHyperParams) -> None:
        self.quantiles = quantiles
        self.models = [XGBClassifier(class_weight="balanced", n_estimators=hyperparams.n_estimators) for _ in quantiles]
        self._is_fitted = False


# 4 Logistic Regression Classifier
class LogisticLearnerHyperParams(LWFLHyperParams):
    """Hyperparameters for Learned Weights Final Learner with LGBM Random Forest Classifier."""

    fit_intercept: bool = Field(
        default=True,
        description="Whether to calculate the intercept for this model. Defaults to True.",
    )

    penalty: Literal["l1", "l2", "elasticnet"] = Field(
        default="l2",
        description="Specify the norm used in the penalization. Defaults to 'l2'.",
    )

    c: float = Field(
        default=1.0,
        description="Inverse of regularization strength; must be a positive float. Defaults to 1.0.",
    )

    @property
    def learner(self) -> type["LogisticLearner"]:
        """Returns the LGBMClassifier to be used as final learner."""
        return LogisticLearner


class LogisticLearner(WeightsLearner):
    """Final learner using only Logistic Regression as base learners."""

    def __init__(self, quantiles: list[Quantile], hyperparams: LogisticLearnerHyperParams) -> None:
        self.quantiles = quantiles
        self.models = [
            LogisticRegression(
                class_weight="balanced",
                fit_intercept=hyperparams.fit_intercept,
                penalty=hyperparams.penalty,
                C=hyperparams.c,
            )
            for _ in quantiles
        ]
        self._is_fitted = False


# Assembly classes
class LearnedWeightsHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    base_hyperparams: list[BaseLearnerHyperParams] = Field(
        default=[LGBMHyperParams(), GBLinearHyperParams()],
        description="List of hyperparameter configurations for base learners. "
        "Defaults to [LGBMHyperParams, GBLinearHyperParams].",
    )

    final_hyperparams: LWFLHyperParams = Field(
        default=LGBMLearnerHyperParams(),
        description="Hyperparameters for the final learner. Defaults to LGBMLearnerHyperParams.",
    )


class LearnedWeightsForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: LearnedWeightsHyperParams

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
        self._final_learner = config.hyperparams.final_hyperparams.learner_from_params(
            quantiles=config.quantiles,
            hyperparams=config.hyperparams.final_hyperparams,
        )


__all__ = [
    "LGBMLearner",
    "LGBMLearnerHyperParams",
    "LearnedWeightsForecaster",
    "LearnedWeightsForecasterConfig",
    "LearnedWeightsHyperParams",
    "LogisticLearner",
    "LogisticLearnerHyperParams",
    "RFLearnerHyperParams",
    "RandomForestLearner",
    "WeightsLearner",
    "XGBLearner",
    "XGBLearnerHyperParams",
]
