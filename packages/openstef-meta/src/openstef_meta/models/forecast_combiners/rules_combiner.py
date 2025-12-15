# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Rules-based Meta Forecaster Module."""

import logging
from typing import cast, override

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner, ForecastCombinerConfig
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_meta.utils.decision_tree import Decision, DecisionTree

logger = logging.getLogger(__name__)


class RulesLearnerHyperParams(HyperParams):
    """HyperParams for Stacking Final Learner."""

    decision_tree: DecisionTree = Field(
        description="Decision tree defining the rules for the final learner.",
        default=DecisionTree(
            nodes=[Decision(idx=0, decision="LGBMForecaster")],
            outcomes={"LGBMForecaster"},
        ),
    )


class RulesCombinerConfig(ForecastCombinerConfig):
    """Configuration for Rules-based Forecast Combiner."""

    hyperparams: HyperParams = Field(
        description="Hyperparameters for the Rules-based final learner.",
        default=RulesLearnerHyperParams(),
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
    def _validate_hyperparams(v: HyperParams) -> HyperParams:
        if not isinstance(v, RulesLearnerHyperParams):
            raise TypeError("hyperparams must be an instance of RulesLearnerHyperParams.")
        return v


class RulesCombiner(ForecastCombiner):
    """Combines base learner predictions per quantile into final predictions using a regression approach."""

    Config = RulesCombinerConfig

    def __init__(self, config: RulesCombinerConfig) -> None:
        """Initialize the Rules Learner.

        Args:
            config: Configuration for the Rules Combiner.
        """
        hyperparams = cast(RulesLearnerHyperParams, config.hyperparams)
        self.tree = hyperparams.decision_tree
        self.quantiles = config.quantiles
        self.config = config

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        # No fitting needed for rule-based final learner
        # Check that additional features are provided
        if additional_features is None:
            raise ValueError("Additional features must be provided for RulesForecastCombiner prediction.")

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
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if additional_features is None:
            raise ValueError("Additional features must be provided for RulesForecastCombiner prediction.")

        decisions = self._predict_tree(
            additional_features.data, columns=data.select_quantile(quantile=self.quantiles[0]).data.columns
        )

        # Generate predictions
        predictions: list[pd.DataFrame] = []
        for q in self.quantiles:
            dataset = data.select_quantile(quantile=q)
            preds = dataset.input_data().multiply(decisions).sum(axis=1)

            predictions.append(preds.to_frame(name=Quantile(q).format()))

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
        if additional_features is None:
            raise ValueError("Additional features must be provided for RulesForecastCombiner prediction.")

        decisions = self._predict_tree(
            additional_features.data, columns=data.select_quantile(quantile=self.quantiles[0]).data.columns
        )

        # Generate predictions
        predictions: list[pd.DataFrame] = []
        for q in self.quantiles:
            dataset = data.select_quantile(quantile=q)
            preds = dataset.input_data().multiply(decisions).sum(axis=1)

            predictions.append(preds.to_frame(name=Quantile(q).format()))

        # Concatenate predictions along columns to form a DataFrame with quantile columns
        return pd.concat(predictions, axis=1)

    @property
    def is_fitted(self) -> bool:
        """Check the Rules Final Learner is fitted."""
        return True


__all__ = [
    "RulesCombiner",
    "RulesCombinerConfig",
    "RulesLearnerHyperParams",
]
