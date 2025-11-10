# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Hybrid Forecaster (Stacked LightGBM + Linear Model Gradient Boosting).

Provides method that attempts to combine the advantages of a linear model (Extraplolation)
and tree-based model (Non-linear patterns). This is acieved by training two base learners,
followed by a small linear model that regresses on the baselearners' predictions.
The implementation is based on sklearn's StackingRegressor.
"""

from typing import TYPE_CHECKING, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_models.estimators.hybrid import HybridQuantileRegressor
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearHyperParams
from openstef_models.models.forecasting.lightgbm_forecaster import LightGBMHyperParams

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class HybridHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    lightgbm_params: LightGBMHyperParams = LightGBMHyperParams()
    gb_linear_params: GBLinearHyperParams = GBLinearHyperParams()

    l1_penalty: float = Field(
        default=0.0,
        description="L1 regularization term for the quantile regression.",
    )


class HybridForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: HybridHyperParams = HybridHyperParams()

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


MODEL_CODE_VERSION = 2


class HybridForecasterState(BaseConfig):
    """Serializable state for Hybrid forecaster persistence."""

    version: int = Field(default=MODEL_CODE_VERSION, description="Version of the model code.")
    config: HybridForecasterConfig = Field(..., description="Forecaster configuration.")
    model: str = Field(..., description="Base64-encoded serialized Hybrid model.")


class HybridForecaster(Forecaster):
    """Wrapper for sklearn's StackingRegressor to make it compatible with HorizonForecaster."""

    Config = HybridForecasterConfig
    HyperParams = HybridHyperParams

    _config: HybridForecasterConfig
    model: HybridQuantileRegressor

    def __init__(self, config: HybridForecasterConfig) -> None:
        """Initialize the Hybrid forecaster."""
        self._config = config

        self._model = HybridQuantileRegressor(
            quantiles=[float(q) for q in config.quantiles],
            lightgbm_n_estimators=config.hyperparams.lightgbm_params.n_estimators,
            lightgbm_learning_rate=config.hyperparams.lightgbm_params.learning_rate,
            lightgbm_max_depth=config.hyperparams.lightgbm_params.max_depth,
            lightgbm_min_child_weight=config.hyperparams.lightgbm_params.min_child_weight,
            lightgbm_min_data_in_leaf=config.hyperparams.lightgbm_params.min_data_in_leaf,
            lightgbm_min_data_in_bin=config.hyperparams.lightgbm_params.min_data_in_bin,
            lightgbm_reg_alpha=config.hyperparams.lightgbm_params.reg_alpha,
            lightgbm_reg_lambda=config.hyperparams.lightgbm_params.reg_lambda,
            lightgbm_num_leaves=config.hyperparams.lightgbm_params.num_leaves,
            lightgbm_max_bin=config.hyperparams.lightgbm_params.max_bin,
            lightgbm_colsample_by_tree=config.hyperparams.lightgbm_params.colsample_bytree,
            gblinear_n_steps=config.hyperparams.gb_linear_params.n_steps,
            gblinear_learning_rate=config.hyperparams.gb_linear_params.learning_rate,
            gblinear_reg_alpha=config.hyperparams.gb_linear_params.reg_alpha,
            gblinear_reg_lambda=config.hyperparams.gb_linear_params.reg_lambda,
        )

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return self._model.is_fitted

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the Hybrid model to the training data.

        Args:
            data: Training data in the expected ForecastInputDataset format.
            data_val: Validation data for tuning the model (optional, not used in this implementation).

        """
        input_data: pd.DataFrame = data.input_data()
        target: npt.NDArray[np.floating] = data.target_series.to_numpy()  # type: ignore
        sample_weights: pd.Series = data.sample_weight_series

        self._model.fit(X=input_data, y=target, sample_weight=sample_weights)

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self._model.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)
        prediction: npt.NDArray[np.floating] = self._model.predict(X=input_data)

        return ForecastDataset(
            data=pd.DataFrame(
                data=prediction,
                index=input_data.index,
                columns=[quantile.format() for quantile in self.config.quantiles],
            ),
            sample_interval=data.sample_interval,
        )


__all__ = ["HybridForecaster", "HybridForecasterConfig", "HybridHyperParams"]
