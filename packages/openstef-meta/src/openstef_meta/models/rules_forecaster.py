# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Rules-based Meta Forecaster Module."""

import logging
from typing import override

import pandas as pd
from pydantic import Field, field_validator
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import HyperParams
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import Quantile
from openstef_meta.framework.base_learner import (
    BaseLearner,
    BaseLearnerHyperParams,
)
from openstef_meta.framework.final_learner import FinalLearner, FinalLearnerHyperParams
from openstef_meta.framework.meta_forecaster import (
    EnsembleForecaster,
)
from openstef_meta.utils.decision_tree import Decision, DecisionTree, Rule
from openstef_models.models.forecasting.forecaster import (
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams
from openstef_models.transforms.time_domain import HolidayFeatureAdder

logger = logging.getLogger(__name__)


class RulesLearnerHyperParams(FinalLearnerHyperParams):
    """HyperParams for Stacking Final Learner."""

    feature_adders: list[TimeSeriesTransform] = Field(
        default=[],
        description="Additional features to add to the final learner.",
    )

    decision_tree: DecisionTree = Field(
        description="Decision tree defining the rules for the final learner.",
    )

    @field_validator("feature_adders", mode="after")
    @classmethod
    def _check_not_empty(cls, v: list[TimeSeriesTransform]) -> list[TimeSeriesTransform]:
        if v == []:
            raise ValueError("RulesForecaster requires at least one feature adder.")
        return v


class RulesLearner(FinalLearner):
    """Combines base learner predictions per quantile into final predictions using a regression approach."""

    def __init__(self, quantiles: list[Quantile], hyperparams: RulesLearnerHyperParams) -> None:
        """Initialize the Rules Learner.

        Args:
            quantiles: List of quantiles to predict.
            hyperparams: Hyperparameters for the final learner.
            horizon: Forecast horizon for which to create the final learner.
        """
        super().__init__(quantiles=quantiles, hyperparams=hyperparams)

        self.tree = hyperparams.decision_tree
        self.feature_adders = hyperparams.feature_adders

    @override
    def fit(
        self,
        base_learner_predictions: dict[Quantile, ForecastInputDataset],
        additional_features: ForecastInputDataset | None,
    ) -> None:
        # No fitting needed for rule-based final learner
        # Check that additional features are provided
        if additional_features is None:
            raise ValueError("Additional features must be provided for RulesFinalLearner prediction.")

    def _predict_tree(self, data: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
        """Predict using the decision tree rules.

        Args:
            data: DataFrame containing the additional features.
            columns: Expected columns for the output DataFrame.

        Returns:
            DataFrame with predictions for each quantile.
        """
        predictions = data.apply(self.tree.get_decision, axis=1)

        return pd.get_dummies(predictions).reindex(columns=columns)

    @override
    def predict(
        self,
        base_learner_predictions: dict[Quantile, ForecastInputDataset],
        additional_features: ForecastInputDataset | None,
    ) -> ForecastDataset:
        if additional_features is None:
            raise ValueError("Additional features must be provided for RulesFinalLearner prediction.")

        decisions = self._predict_tree(
            additional_features.data, columns=base_learner_predictions[self.quantiles[0]].data.columns
        )

        # Generate predictions
        predictions: list[pd.DataFrame] = []
        for q, data in base_learner_predictions.items():
            preds = data.data * decisions
            predictions.append(preds.sum(axis=1).to_frame(name=Quantile(q).format()))

        # Concatenate predictions along columns to form a DataFrame with quantile columns
        df = pd.concat(predictions, axis=1)

        return ForecastDataset(
            data=df,
            sample_interval=base_learner_predictions[self.quantiles[0]].sample_interval,
        )

    @property
    def is_fitted(self) -> bool:
        """Check the Rules Final Learner is fitted."""
        return True


class RulesForecasterHyperParams(HyperParams):
    """Hyperparameters for Rules Forecaster."""

    base_hyperparams: list[BaseLearnerHyperParams] = Field(
        default=[LGBMHyperParams(), GBLinearHyperParams()],
        description="List of hyperparameter configurations for base learners. "
        "Defaults to [LGBMHyperParams, GBLinearHyperParams].",
    )

    final_hyperparams: RulesLearnerHyperParams = Field(
        description="Hyperparameters for the final learner.",
        default=RulesLearnerHyperParams(
            decision_tree=DecisionTree(nodes=[Decision(idx=0, decision="LGBMForecaster")], outcomes={"LGBMForecaster"}),
            feature_adders=[HolidayFeatureAdder(country_code=CountryAlpha2("NL"))],
        ),
    )

    @field_validator("base_hyperparams", mode="after")
    @classmethod
    def _check_classes(cls, v: list[BaseLearnerHyperParams]) -> list[BaseLearnerHyperParams]:
        hp_classes = [type(hp) for hp in v]
        if not len(hp_classes) == len(set(hp_classes)):
            raise ValueError("Duplicate base learner hyperparameter classes are not allowed.")
        return v


class RulesForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: RulesForecasterHyperParams = Field(
        default=RulesForecasterHyperParams(),
        description="Hyperparameters for the Hybrid forecaster.",
    )

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


class RulesForecaster(EnsembleForecaster):
    """Wrapper for sklearn's StackingRegressor to make it compatible with HorizonForecaster."""

    Config = RulesForecasterConfig
    HyperParams = RulesForecasterHyperParams

    def __init__(self, config: RulesForecasterConfig) -> None:
        """Initialize the Hybrid forecaster."""
        self._config = config

        self._base_learners: list[BaseLearner] = self._init_base_learners(
            config=config, base_hyperparams=config.hyperparams.base_hyperparams
        )

        self._final_learner = RulesLearner(
            quantiles=config.quantiles,
            hyperparams=config.hyperparams.final_hyperparams,
        )


__all__ = [
    "RulesForecaster",
    "RulesForecasterConfig",
    "RulesForecasterHyperParams",
    "RulesLearner",
    "RulesLearnerHyperParams",
]
