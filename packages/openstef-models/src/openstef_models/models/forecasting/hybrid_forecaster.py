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
from sklearn.linear_model import QuantileRegressor

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
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


class HybridHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    base_hyperparams: list[BaseLearnerHyperParams] = Field(
        default=[LGBMHyperParams(), GBLinearHyperParams()],
        description="List of hyperparameter configurations for base learners. "
        "Defaults to [LGBMHyperParams, GBLinearHyperParams].",
    )

    l1_penalty: float = Field(
        default=0.0,
        description="L1 regularization term for the quantile regression.",
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
        self._final_learner = [
            QuantileRegressor(quantile=float(q), alpha=config.hyperparams.l1_penalty) for q in config.quantiles
        ]

        self._is_fitted: bool = False

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def _hyperparams_forecast_map(hyperparams: type[BaseLearnerHyperParams]) -> type[BaseLearner]:
        """Map hyperparameters to forecast types.

        Args:
            hyperparams: Hyperparameters of the base learner.

        Returns:
            Corresponding Forecaster class.

        Raises:
            TypeError: If a nested HybridForecaster is attempted.
        """
        if isinstance(hyperparams, HybridHyperParams):
            raise TypeError("Nested HybridForecaster is not supported.")

        mapping: dict[type[BaseLearnerHyperParams], type[BaseLearner]] = {
            LGBMHyperParams: LGBMForecaster,
            LGBMLinearHyperParams: LGBMLinearForecaster,
            XGBoostHyperParams: XGBoostForecaster,
            GBLinearHyperParams: GBLinearForecaster,
        }
        return mapping[hyperparams]

    @staticmethod
    def _base_learner_config(base_learner_class: type[BaseLearner]) -> type[BaseLearnerConfig]:
        """Extract the configuration from a base learner.

        Args:
            base_learner_class: The base learner forecaster.

        Returns:
            The configuration of the base learner.
        """
        mapping: dict[type[BaseLearner], type[BaseLearnerConfig]] = {
            LGBMForecaster: LGBMForecasterConfig,
            LGBMLinearForecaster: LGBMLinearForecasterConfig,
            XGBoostForecaster: XGBoostForecasterConfig,
            GBLinearForecaster: GBLinearForecasterConfig,
        }
        return mapping[base_learner_class]

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

        full_dataset = ForecastInputDataset(
            data=data.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.index[0],
        )

        base_predictions = self._predict_base_learners(data=full_dataset)

        quantile_dataframes = self._prepare_input_final_learner(base_predictions=base_predictions)

        self._fit_final_learner(target=data.target_series, quantile_df=quantile_dataframes)

        self._is_fitted = True

    def _fit_final_learner(
        self,
        target: pd.Series,
        quantile_df: dict[str, pd.DataFrame],
    ) -> None:
        """Fit the final learner using base learner predictions.

        Args:
            target: Target values for training.
            quantile_df: Dictionary mapping quantile strings to DataFrames of base learner predictions.
        """
        for i, df in enumerate(quantile_df.values()):
            self._final_learner[i].fit(X=df, y=target)

    def _predict_base_learners(self, data: ForecastInputDataset) -> dict[str, ForecastDataset]:
        """Generate predictions from base learners.

        Args:
            data: Input data for prediction.

        Returns:
            DataFrame containing base learner predictions.
        """
        base_predictions: dict[str, ForecastDataset] = {}
        for learner in self._base_learners:
            preds = learner.predict(data=data)
            base_predictions[learner.__class__.__name__] = preds

        return base_predictions

    def _predict_final_learner(
        self, quantile_df: dict[str, pd.DataFrame], data: ForecastInputDataset
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate predictions
        predictions_dict = [
            pd.Series(self._final_learner[i].predict(X=quantile_df[q_str]), index=quantile_df[q_str].index, name=q_str)
            for i, q_str in enumerate(quantile_df.keys())
        ]

        # Construct DataFrame with appropriate quantile columns
        predictions = pd.DataFrame(
            data=predictions_dict,
        ).T

        return ForecastDataset(
            data=predictions,
            sample_interval=data.sample_interval,
        )

    @staticmethod
    def _prepare_input_final_learner(base_predictions: dict[str, ForecastDataset]) -> dict[str, pd.DataFrame]:
        """Prepare input data for the final learner based on base learner predictions.

        Args:
            base_predictions: Dictionary of base learner predictions.

        Returns:
            dictionary mapping quantile strings to DataFrames of base learner predictions.
        """
        predictions_quantiles: dict[str, pd.DataFrame] = {}
        first_key = next(iter(base_predictions))
        for quantile in base_predictions[first_key].quantiles:
            quantile_str = quantile.format()
            quantile_preds = pd.DataFrame({
                learner_name: preds.data[quantile_str] for learner_name, preds in base_predictions.items()
            })
            predictions_quantiles[quantile_str] = quantile_preds

        return predictions_quantiles

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        base_predictions = self._predict_base_learners(data=data)

        final_learner_input = self._prepare_input_final_learner(base_predictions=base_predictions)

        return self._predict_final_learner(
            quantile_df=final_learner_input,
            data=data,
        )

    # TODO(@Lars800): #745: Make forecaster Explainable


__all__ = ["HybridForecaster", "HybridForecasterConfig", "HybridHyperParams"]
