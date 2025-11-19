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
from abc import abstractmethod

import pandas as pd
from pydantic import Field, field_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_core.types import Quantile
from openstef_models.estimators.hybrid import HybridQuantileRegressor
from openstef_models.models.forecasting.forecaster import (
    Forecaster,
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearForecasterConfig,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster, LGBMForecasterConfig, LGBMHyperParams
from openstef_models.models.forecasting.lgbmlinear_forecaster import (
    LGBMLinearForecaster,
    LGBMLinearForecasterConfig,
    LGBMLinearHyperParams,
)
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
    XGBoostHyperParams,
)

logger = logging.getLogger(__name__)


BaseLearner = LGBMForecaster | LGBMLinearForecaster | XGBoostForecaster | GBLinearForecaster
BaseLearnerHyperParams = LGBMHyperParams | LGBMLinearHyperParams | XGBoostHyperParams | GBLinearHyperParams
BaseLearnerConfig = (
    LGBMForecasterConfig | LGBMLinearForecasterConfig | XGBoostForecasterConfig | GBLinearForecasterConfig
)


class FinalLearner:
    """Combines base learner predictions for each quantile into final predictions."""

    @abstractmethod
    def fit(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> None:
        raise NotImplementedError("Subclasses must implement the fit method.")

    def predict(self, base_learner_predictions: dict[Quantile, ForecastInputDataset]) -> ForecastDataset:
        raise NotImplementedError("Subclasses must implement the predict method.")

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        raise NotImplementedError("Subclasses must implement the is_fitted property.")


class FinalForecaster(FinalLearner):
    """Combines base learner predictions for each quantile into final predictions."""

    def __init__(self, forecaster: Forecaster, feature_adders: None = None) -> None:
        # Feature adders placeholder for future use

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
        return all(x.is_fitted for x in self.models)


class HybridHyperParams(HyperParams):
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


class HybridForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: HybridHyperParams = HybridHyperParams()

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


class HybridForecaster(Forecaster):
    """Wrapper for sklearn's StackingRegressor to make it compatible with HorizonForecaster."""

    Config = HybridForecasterConfig
    HyperParams = HybridHyperParams

    _config: HybridForecasterConfig
    model: HybridQuantileRegressor

    def __init__(self, config: HybridForecasterConfig) -> None:
        """Initialize the Hybrid forecaster."""

        self._config = config

        self._base_learners: list[BaseLearner] = self._init_base_learners(
            base_hyperparams=config.hyperparams.base_hyperparams
        )
        final_forecaster = self._init_base_learners(base_hyperparams=[config.hyperparams.final_hyperparams])[0]
        self._final_learner = FinalForecaster(forecaster=final_forecaster)

    def _init_base_learners(self, base_hyperparams: list[BaseLearnerHyperParams]) -> list[BaseLearner]:
        """Initialize base learners based on provided hyperparameters.

        Returns:
            list[Forecaster]: List of initialized base learner forecasters.
        """
        base_learners: list[BaseLearner] = []
        horizons = self.config.horizons
        quantiles = self.config.quantiles

        for hyperparams in base_hyperparams:
            forecaster_cls = hyperparams.forecaster_class()
            config = forecaster_cls.Config(horizons=horizons, quantiles=quantiles)
            if "hyperparams" in forecaster_cls.Config.model_fields:
                config = config.model_copy(update={"hyperparams": hyperparams})

            base_learners.append(config.forecaster_from_config())

        return base_learners

    @property
    @override
    def is_fitted(self) -> bool:
        return all(x.is_fitted for x in self._base_learners)

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the Hybrid model to the training data.

        Args:
            data: Training data in the expected ForecastInputDataset format.
            data_val: Validation data for tuning the model (optional, not used in this implementation).

        """
        # Fit base learners
        [x.fit(data=data, data_val=data_val) for x in self._base_learners]

        # Reset forecast start date to ensure we predict on the full dataset
        full_dataset = ForecastInputDataset(
            data=data.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.index[0],
        )

        # Compute and store numeric feature min/max ranges from training data (exclude target column)
        features_df = full_dataset.data.drop(columns=[full_dataset.target_column], errors="ignore")
        numeric_features = features_df.select_dtypes(include="number")
        if not numeric_features.empty:
            mins = numeric_features.min()
            maxs = numeric_features.max()
            self._feature_ranges = {col: (float(mins[col]), float(maxs[col])) for col in numeric_features.columns}
        else:
            self._feature_ranges = {}

        base_predictions = self._predict_base_learners(data=full_dataset)

        # pass training features + stored ranges so the final input gains the inside_train_range flag
        quantile_datasets = self._prepare_input_final_learner(
            base_predictions=base_predictions,
            quantiles=self._config.quantiles,
            target_series=data.target_series,
            input_features=full_dataset.data,
            feature_ranges=self._feature_ranges,
        )

        self._final_learner.fit(
            base_learner_predictions=quantile_datasets,
        )

        self._is_fitted = True

    def _predict_base_learners(self, data: ForecastInputDataset) -> dict[type[BaseLearner], ForecastDataset]:
        """Generate predictions from base learners.

        Args:
            data: Input data for prediction.

        Returns:
            DataFrame containing base learner predictions.
        """
        base_predictions: dict[type[BaseLearner], ForecastDataset] = {}
        for learner in self._base_learners:
            preds = learner.predict(data=data)
            base_predictions[learner.__class__] = preds

        return base_predictions

    @staticmethod
    def _prepare_input_final_learner(
        quantiles: list[Quantile],
        base_predictions: dict[type[BaseLearner], ForecastDataset],
        target_series: pd.Series,
        input_features: pd.DataFrame | None = None,
        feature_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[Quantile, ForecastInputDataset]:
        """Prepare input data for the final learner based on base learner predictions.

        Args:
            base_predictions: Dictionary of base learner predictions.

        Returns:
            dictionary mapping quantile strings to DataFrames of base learner predictions.
        """
        predictions_quantiles: dict[Quantile, ForecastInputDataset] = {}
        sample_interval = base_predictions[next(iter(base_predictions))].sample_interval
        target_name = str(target_series.name)

        for q in quantiles:
            df = pd.DataFrame({
                learner.__name__: preds.data[Quantile(q).format()] for learner, preds in base_predictions.items()
            })
            df[target_name] = target_series

            # Add inside_train_range flag (1 if all numeric input features are within training min/max)
            inside_flag = pd.Series(False, index=df.index)
            if input_features is not None and feature_ranges:
                # align input_features to prediction index where possible
                features_aligned = input_features.reindex(df.index)
                numeric = features_aligned.select_dtypes(include="number")
                if not numeric.empty:
                    inside = pd.Series(True, index=numeric.index)
                    for col, (mn, mx) in feature_ranges.items():
                        if col in numeric.columns:
                            inside &= numeric[col].ge(mn) & numeric[col].le(mx)
                        else:
                            # missing feature in current input -> treat as inside range
                            inside &= True
                    inside_flag = inside.fillna(False) # type: ignore
            df["inside_train_range"] = inside_flag.astype(int)

            predictions_quantiles[q] = ForecastInputDataset(
                data=df,
                sample_interval=sample_interval,
                target_column=target_name,
                forecast_start=df.index[0],
            )

        return predictions_quantiles

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        base_predictions = self._predict_base_learners(data=data)

        final_learner_input = self._prepare_input_final_learner(
            quantiles=self._config.quantiles,
            base_predictions=base_predictions,
            target_series=data.target_series,
            input_features=data.data,
            feature_ranges=getattr(self, "_feature_ranges", None),
        )

        return self._final_learner.predict(base_learner_predictions=final_learner_input)

    # TODO(@Lars800): #745: Make forecaster Explainable


__all__ = ["HybridForecaster", "HybridForecasterConfig", "HybridHyperParams"]
