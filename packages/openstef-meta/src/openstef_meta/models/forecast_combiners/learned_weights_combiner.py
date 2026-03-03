# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Learned Weights Combiner.

Forecast combiner that uses a classification approach to learn weights for base forecasters.
It learns which forecaster is likely to perform best under different conditions.

The combiner can operate in two modes:
- Hard Selection: Selects the base forecaster with the highest predicted probability for each instance.
- Soft Selection: Uses the predicted probabilities as weights to combine base forecaster predictions.
"""

import logging
from typing import Literal, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight  # type: ignore[import-untyped]

from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ENSEMBLE_COLUMN_SEP, EnsembleForecastDataset
from openstef_core.exceptions import MissingExtraError, NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import (
    ForecastCombiner,
)
from openstef_meta.utils.datasets import combine_forecast_input_datasets

logger = logging.getLogger(__name__)


class LGBMCombinerHyperParams(HyperParams):
    """Hyperparameters for the LGBM gradient-boosted classifier."""

    n_estimators: int = Field(default=20, description="Number of boosting rounds.")
    n_leaves: int = Field(default=31, description="Maximum number of leaves per tree.")
    reg_alpha: float = Field(default=0.0, description="L1 regularization term on weights.")
    reg_lambda: float = Field(default=0.0, description="L2 regularization term on weights.")

    def get_classifier(self) -> ClassifierMixin:
        """Create an LGBM gradient-boosted classifier from these hyperparameters.

        Returns:
            Configured LGBMClassifier instance.

        Raises:
            MissingExtraError: If lightgbm is not installed.
        """
        try:
            from lightgbm import LGBMClassifier  # noqa: PLC0415
        except ImportError as e:
            raise MissingExtraError("lightgbm", "openstef-models") from e

        return cast(
            ClassifierMixin,
            LGBMClassifier(
                class_weight="balanced",
                n_estimators=self.n_estimators,
                num_leaves=self.n_leaves,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                n_jobs=1,
            ),
        )


class RFCombinerHyperParams(HyperParams):
    """Hyperparameters for the LGBM random-forest classifier."""

    n_estimators: int = Field(default=20, description="Number of trees.")
    n_leaves: int = Field(default=31, description="Maximum number of leaves per tree.")
    bagging_freq: int = Field(default=1, description="Frequency for bagging.")
    bagging_fraction: float = Field(default=0.8, description="Fraction of data per iteration.")
    feature_fraction: float = Field(default=1, description="Fraction of features per iteration.")

    def get_classifier(self) -> ClassifierMixin:
        """Create an LGBM random-forest classifier from these hyperparameters.

        Returns:
            Configured LGBMClassifier instance in random-forest mode.

        Raises:
            MissingExtraError: If lightgbm is not installed.
        """
        try:
            from lightgbm import LGBMClassifier  # noqa: PLC0415
        except ImportError as e:
            raise MissingExtraError("lightgbm", "openstef-models") from e

        return cast(
            ClassifierMixin,
            LGBMClassifier(
                boosting_type="rf",
                class_weight="balanced",
                n_estimators=self.n_estimators,
                num_leaves=self.n_leaves,
                bagging_freq=self.bagging_freq,
                bagging_fraction=self.bagging_fraction,
                feature_fraction=self.feature_fraction,
            ),
        )


class XGBCombinerHyperParams(HyperParams):
    """Hyperparameters for the XGBoost classifier."""

    n_estimators: int = Field(default=20, description="Number of boosting rounds.")

    def get_classifier(self) -> ClassifierMixin:
        """Create an XGBoost classifier from these hyperparameters.

        Returns:
            Configured XGBClassifier instance.

        Raises:
            MissingExtraError: If xgboost is not installed.
        """
        try:
            from xgboost import XGBClassifier  # noqa: PLC0415
        except ImportError as e:
            raise MissingExtraError("xgboost", "openstef-models") from e

        return cast(ClassifierMixin, XGBClassifier(n_estimators=self.n_estimators))


class LogisticCombinerHyperParams(HyperParams):
    """Hyperparameters for the logistic regression classifier."""

    fit_intercept: bool = Field(default=True, description="Whether to calculate the intercept.")
    penalty: Literal["l1", "l2", "elasticnet"] = Field(default="l2", description="Regularization norm.")
    c: float = Field(default=1.0, description="Inverse of regularization strength.")

    def get_classifier(self) -> ClassifierMixin:
        """Create a logistic regression classifier from these hyperparameters.

        Returns:
            Configured LogisticRegression instance.
        """
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        return LogisticRegression(
            class_weight="balanced",
            fit_intercept=self.fit_intercept,
            penalty=self.penalty,
            C=self.c,
        )


class WeightsCombiner(ForecastCombiner):
    """Combines base forecaster predictions with a classification approach.

    A classifier predicts per-timestep model weights.  Depending on ``hard_selection``,
    the combiner either picks the best forecaster (hard) or blends using predicted
    probabilities (soft).
    """

    hyperparams: HyperParams = Field(
        default_factory=LGBMCombinerHyperParams,
        description="Classifier hyperparameters. Must have a get_classifier() method.",
    )

    @property
    @override
    def hparams(self) -> HyperParams:
        return self.hyperparams

    hard_selection: bool = Field(
        default=False,
        description="If True, select the single best forecaster per timestep; otherwise blend.",
    )

    _label_encoder: LabelEncoder = PrivateAttr(default_factory=LabelEncoder)
    _is_fitted: bool = PrivateAttr(default=False)
    _feature_names: list[str] = PrivateAttr(default_factory=list[str])
    _models: dict[Quantile, ClassifierMixin] = PrivateAttr(default_factory=dict[Quantile, ClassifierMixin])

    def model_post_init(self, _context: object, /) -> None:
        """Validate hyperparams and initialize per-quantile classifiers.

        Raises:
            TypeError: If hyperparams does not have a ``get_classifier()`` method.
        """
        if not hasattr(self.hyperparams, "get_classifier"):
            msg = f"hyperparams ({type(self.hyperparams).__name__}) must have a get_classifier() method."
            raise TypeError(msg)

        self._models = {
            # One classifier per quantile — optimal forecaster may differ across quantile levels
            q: self.hyperparams.get_classifier()  # type: ignore[union-attr]
            for q in self.quantiles
        }

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        self._label_encoder.fit(data.forecaster_names)

        feature_names: list[str] = []
        for q in self.quantiles:
            base_data = data.get_base_predictions_for_quantile(quantile=q)
            labels = self._classify_best_forecaster(base_data, quantile=q)
            combined_data = combine_forecast_input_datasets(
                input_data=base_data,
                additional_features=additional_features,
            )
            input_data = combined_data.input_data()
            self._validate_labels(labels=labels, quantile=q)
            encoded_labels = self._label_encoder.transform(labels)

            weights = compute_sample_weight("balanced", encoded_labels) * combined_data.sample_weight_series
            self._models[q].fit(X=input_data, y=encoded_labels, sample_weight=weights)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            feature_names = list(input_data.columns)

        self._feature_names = feature_names
        self._is_fitted = True

    @staticmethod
    def _classify_best_forecaster(data: ForecastInputDataset, quantile: Quantile) -> pd.Series:
        """Compute best-forecaster labels via pinball loss.

        For each sample, returns the name of the forecaster with the lowest
        pinball loss at the given quantile.

        Returns:
            Series with the name of the best-performing forecaster per sample.
        """
        predictions = data.input_data()
        y_true = np.asarray(data.target_series)

        def _pinball_loss(preds: pd.Series) -> np.ndarray:
            y_pred = np.asarray(preds)
            errors = y_true - y_pred
            return np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)

        losses = predictions.apply(_pinball_loss)
        return losses.idxmin(axis=1)

    def _validate_labels(self, labels: pd.Series, quantile: Quantile) -> None:
        # Fall back to DummyClassifier when one forecaster dominates — sklearn classifiers need ≥2 classes
        if len(labels.unique()) == 1:
            logger.warning("Quantile %s has only 1 class — switching to DummyClassifier.", quantile.format())
            self._models[quantile] = DummyClassifier(strategy="most_frequent")

    def _predict_weights(self, base_predictions: pd.DataFrame, quantile: Quantile) -> pd.DataFrame:
        model = self._models[quantile]
        if isinstance(model, DummyClassifier):
            # DummyClassifier has no predict_proba — construct one-hot weights manually
            weights_array = pd.DataFrame(0, index=base_predictions.index, columns=self._label_encoder.classes_)
            weights_array[self._label_encoder.classes_[0]] = 1.0
        else:
            weights_array = model.predict_proba(base_predictions)  # type: ignore[union-attr]

        return pd.DataFrame(weights_array, index=base_predictions.index, columns=self._label_encoder.classes_)  # type: ignore[arg-type]

    @staticmethod
    def _prepare_input_data(
        dataset: ForecastInputDataset, additional_features: ForecastInputDataset | None
    ) -> pd.DataFrame:
        df = dataset.input_data(start=dataset.index[0])
        if additional_features is not None:
            df_a = additional_features.input_data(start=dataset.index[0])
            df = pd.concat([df, df_a], axis=1, join="inner")
        return df

    def _predict_quantile(
        self,
        dataset: ForecastInputDataset,
        additional_features: ForecastInputDataset | None,
        quantile: Quantile,
    ) -> pd.Series:
        input_data = self._prepare_input_data(dataset=dataset, additional_features=additional_features)
        weights = self._predict_weights(base_predictions=input_data, quantile=quantile)

        if self.hard_selection:
            # Convert soft probabilities to hard selection: max weight → 1.0, ties distributed equally
            weights = (weights == weights.max(axis=1).to_frame().to_numpy()) / weights.sum(axis=1).to_frame().to_numpy()

        # Weighted average: multiply each forecaster's prediction by its weight and sum
        return dataset.input_data().mul(weights).sum(axis=1)

    @override
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        predictions = pd.DataFrame({
            q.format(): self._predict_quantile(
                dataset=data.get_base_predictions_for_quantile(quantile=q),
                additional_features=additional_features,
                quantile=q,
            )
            for q in self.quantiles
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
    ) -> TimeSeriesDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        contribution_list = [
            self._contributions_for_quantile(
                dataset=data.get_base_predictions_for_quantile(quantile=q),
                additional_features=additional_features,
                quantile=q,
            )
            for q in self.quantiles
        ]

        contributions = pd.concat(contribution_list, axis=1)

        target_series = data.target_series
        if target_series is not None:
            contributions[data.target_column] = target_series

        return TimeSeriesDataset(data=contributions, sample_interval=data.sample_interval)

    def _contributions_for_quantile(
        self,
        dataset: ForecastInputDataset,
        additional_features: ForecastInputDataset | None,
        quantile: Quantile,
    ) -> pd.DataFrame:
        input_data = self._prepare_input_data(dataset=dataset, additional_features=additional_features)
        weights = self._predict_weights(base_predictions=input_data, quantile=quantile)
        weights.columns = [f"{col}{ENSEMBLE_COLUMN_SEP}{quantile.format()}" for col in weights.columns]
        return weights

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        """Feature importances from the internal classifiers, per quantile."""
        importances: dict[str, np.ndarray] = {}
        for q, model in self._models.items():
            if hasattr(model, "feature_importances_"):
                raw = np.array(model.feature_importances_, dtype=float)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            elif hasattr(model, "coef_"):
                raw = np.abs(np.array(model.coef_, dtype=float)).mean(axis=0)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            else:
                raw = np.ones(len(self._feature_names), dtype=float)

            total = raw.sum()
            importances[q.format()] = raw / total if total > 0 else raw

        return pd.DataFrame(importances, index=self._feature_names)


__all__ = [
    "LGBMCombinerHyperParams",
    "LogisticCombinerHyperParams",
    "RFCombinerHyperParams",
    "WeightsCombiner",
    "XGBCombinerHyperParams",
]
