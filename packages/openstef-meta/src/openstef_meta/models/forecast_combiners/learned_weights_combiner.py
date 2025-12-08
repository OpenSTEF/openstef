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
from typing import Literal, override

import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import Field
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight  # type: ignore
from xgboost import XGBClassifier

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import (
    ForecastCombiner,
    ForecastCombinerConfig,
)
from openstef_meta.utils.datasets import EnsembleForecastDataset, combine_forecast_input_datasets

logger = logging.getLogger(__name__)


# Base classes for Learned Weights Final Learner

Classifier = LGBMClassifier | XGBClassifier | LogisticRegression | DummyClassifier
ClassifierNames = Literal["lgbm", "xgb", "logistic_regression", "dummy"]


class ClassifierParamsMixin:
    """Hyperparameters for the Final Learner."""

    @abstractmethod
    def get_classifier(self) -> Classifier:
        """Returns the classifier instance."""
        msg = "Subclasses must implement get_classifier method."
        raise NotImplementedError(msg)


class LGBMCombinerHyperParams(HyperParams, ClassifierParamsMixin):
    """Hyperparameters for Learned Weights Final Learner with LGBM Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    n_leaves: int = Field(
        default=31,
        description="Number of leaves for the LGBM Classifier. Defaults to 31.",
    )

    @override
    def get_classifier(self) -> LGBMClassifier:
        """Returns the LGBM Classifier."""
        return LGBMClassifier(
            class_weight="balanced",
            n_estimators=self.n_estimators,
            num_leaves=self.n_leaves,
            n_jobs=1,
        )


class RFCombinerHyperParams(HyperParams, ClassifierParamsMixin):
    """Hyperparameters for Learned Weights Final Learner with LGBM Random Forest Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    n_leaves: int = Field(
        default=31,
        description="Number of leaves for the LGBM Classifier. Defaults to 31.",
    )

    bagging_freq: int = Field(
        default=1,
        description="Frequency for bagging in the Random Forest. Defaults to 1.",
    )

    bagging_fraction: float = Field(
        default=0.8,
        description="Fraction of data to be used for each iteration of the Random Forest. Defaults to 0.8.",
    )

    feature_fraction: float = Field(
        default=1,
        description="Fraction of features to be used for each iteration of the Random Forest. Defaults to 1.",
    )

    @override
    def get_classifier(self) -> LGBMClassifier:
        """Returns the Random Forest LGBMClassifier."""
        return LGBMClassifier(
            boosting_type="rf",
            class_weight="balanced",
            n_estimators=self.n_estimators,
            bagging_freq=self.bagging_freq,
            bagging_fraction=self.bagging_fraction,
            feature_fraction=self.feature_fraction,
            num_leaves=self.n_leaves,
        )


# 3 XGB Classifier
class XGBCombinerHyperParams(HyperParams, ClassifierParamsMixin):
    """Hyperparameters for Learned Weights Final Learner with LGBM Random Forest Classifier."""

    n_estimators: int = Field(
        default=20,
        description="Number of estimators for the LGBM Classifier. Defaults to 20.",
    )

    @override
    def get_classifier(self) -> XGBClassifier:
        """Returns the XGBClassifier."""
        return XGBClassifier(n_estimators=self.n_estimators)


class LogisticCombinerHyperParams(HyperParams, ClassifierParamsMixin):
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

    @override
    def get_classifier(self) -> LogisticRegression:
        """Returns the LogisticRegression."""
        return LogisticRegression(
            class_weight="balanced",
            fit_intercept=self.fit_intercept,
            penalty=self.penalty,
            C=self.c,
        )


class WeightsCombinerConfig(ForecastCombinerConfig):
    """Configuration for WeightsCombiner."""

    hyperparams: HyperParams = Field(
        default=LGBMCombinerHyperParams(),
        description="Hyperparameters for the Weights Combiner.",
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

    @property
    def get_classifier(self) -> Classifier:
        """Returns the classifier instance from hyperparameters.

        Returns:
            Classifier instance.

        Raises:
            TypeError: If hyperparams do not implement ClassifierParamsMixin.
        """
        if not isinstance(self.hyperparams, ClassifierParamsMixin):
            msg = "hyperparams must implement ClassifierParamsMixin to get classifier."
            raise TypeError(msg)
        return self.hyperparams.get_classifier()


class WeightsCombiner(ForecastCombiner):
    """Combines base learner predictions with a classification approach to determine which base learner to use."""

    Config = WeightsCombinerConfig
    LGBMHyperParams = LGBMCombinerHyperParams
    RFHyperParams = RFCombinerHyperParams
    XGBHyperParams = XGBCombinerHyperParams
    LogisticHyperParams = LogisticCombinerHyperParams

    def __init__(self, config: WeightsCombinerConfig) -> None:
        """Initialize the Weigths Combiner."""
        self.quantiles = config.quantiles
        self.config = config
        self.hyperparams = config.hyperparams
        self._is_fitted: bool = False
        self._is_fitted = False
        self._label_encoder = LabelEncoder()

        # Initialize a classifier per quantile
        self.models: list[Classifier] = [config.get_classifier for _ in self.quantiles]

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
        sample_weights: pd.Series | None = None,
    ) -> None:

        self._label_encoder.fit(data.forecaster_names)

        for i, q in enumerate(self.quantiles):
            # Data preparation
            dataset = data.select_quantile_classification(quantile=q)
            combined_data = combine_forecast_input_datasets(
                dataset=dataset,
                other=additional_features,
            )
            input_data = combined_data.input_data()
            labels = combined_data.target_series
            self._validate_labels(labels=labels, model_index=i)
            labels = self._label_encoder.transform(labels)

            # Balance classes, adjust with sample weights
            weights = compute_sample_weight("balanced", labels) * combined_data.sample_weight_series

            self.models[i].fit(X=input_data, y=labels, sample_weight=weights)  # type: ignore
        self._is_fitted = True

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
                join="inner",
            )
        return df

    def _validate_labels(self, labels: pd.Series, model_index: int) -> None:
        if len(labels.unique()) == 1:
            msg = f"""Final learner for quantile {self.quantiles[model_index].format()} has
                     less than 2 classes in the target.
                    Switching to dummy classifier """
            logger.warning(msg=msg)
            self.models[model_index] = DummyClassifier(strategy="most_frequent")

    def _predict_model_weights_quantile(self, base_predictions: pd.DataFrame, model_index: int) -> pd.DataFrame:
        model = self.models[model_index]
        weights_array = model.predict_proba(base_predictions.to_numpy())  # type: ignore

        return pd.DataFrame(weights_array, index=base_predictions.index, columns=self._label_encoder.classes_)  # type: ignore

    def _generate_predictions_quantile(
        self,
        dataset: ForecastInputDataset,
        additional_features: ForecastInputDataset | None,
        model_index: int,
    ) -> pd.Series:

        input_data = self._prepare_input_data(
            dataset=dataset,
            additional_features=additional_features,
        )

        weights = self._predict_model_weights_quantile(base_predictions=input_data, model_index=model_index)

        return dataset.input_data().mul(weights).sum(axis=1)

    @override
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions = pd.DataFrame({
            Quantile(q).format(): self._generate_predictions_quantile(
                dataset=data.select_quantile(quantile=Quantile(q)),
                additional_features=additional_features,
                model_index=i,
            )
            for i, q in enumerate(self.quantiles)
        })
        target_series = data.target_series
        if target_series is not None:
            predictions[data.target_column] = target_series

        return ForecastDataset(
            data=predictions,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.forecast_start,
        )

    @override
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        contribution_list = [
            self._generate_contributions_quantile(
                dataset=data.select_quantile(quantile=Quantile(q)),
                additional_features=additional_features,
                model_index=i,
            )
            for i, q in enumerate(self.quantiles)
        ]

        contributions = pd.concat(contribution_list, axis=1)

        target_series = data.target_series
        if target_series is not None:
            contributions[data.target_column] = target_series

        return contributions

    def _generate_contributions_quantile(
        self,
        dataset: ForecastInputDataset,
        additional_features: ForecastInputDataset | None,
        model_index: int,
    ) -> pd.DataFrame:
        input_data = self._prepare_input_data(
            dataset=dataset,
            additional_features=additional_features,
        )
        weights = self._predict_model_weights_quantile(base_predictions=input_data, model_index=model_index)
        weights.columns = [f"{col}_{Quantile(self.quantiles[model_index]).format()}" for col in weights.columns]
        return weights

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted


__all__ = [
    "LGBMCombinerHyperParams",
    "LogisticCombinerHyperParams",
    "RFCombinerHyperParams",
    "WeightsCombiner",
    "XGBCombinerHyperParams",
]
