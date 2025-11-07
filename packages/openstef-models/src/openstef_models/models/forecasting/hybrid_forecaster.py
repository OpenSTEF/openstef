from __future__ import annotations

import base64
import logging
from typing import Any, Literal, Self, Union, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field


from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    ModelLoadingError,
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_models.estimators.hybrid import HybridQuantileRegressor
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.models.forecasting.lightgbm_forecaster import LightGBMHyperParams


class HybridHyperParams(HyperParams):
    """Hyperparameters for Support Vector Regression (Hybrid)."""

    lightgbm_params: LightGBMHyperParams = LightGBMHyperParams()

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
    """Wrapper for sklearn's Hybrid to make it compatible with HorizonForecaster."""

    Config = HybridForecasterConfig
    HyperParams = HybridHyperParams

    _config: HybridForecasterConfig
    model: HybridQuantileRegressor

    def __init__(self, config: HybridForecasterConfig) -> None:
        """Initialize the Hybrid forecaster.

        Args:
            kernel: Kernel type for Hybrid. Must be one of "linear", "poly", "rbf", "sigmoid", or "precomputed".
            C: Regularization parameter.
            epsilon: Epsilon in the epsilon-Hybrid model.
        """
        self._config = config

        self._model = HybridQuantileRegressor(
            quantiles=config.quantiles,
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
            lightgbm_subsample=config.hyperparams.lightgbm_params.subsample,
            lightgbm_colsample_by_tree=config.hyperparams.lightgbm_params.colsample_bytree,
            lightgbm_colsample_by_node=config.hyperparams.lightgbm_params.colsample_bynode,
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

        self._model.fit(X=input_data, y=target)

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
