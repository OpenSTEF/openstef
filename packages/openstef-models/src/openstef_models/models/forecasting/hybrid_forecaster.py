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
from typing import TYPE_CHECKING, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_models.estimators.hybrid import HybridQuantileRegressor
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearHyperParams
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt


class HybridHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    lgbm_params: LGBMHyperParams = LGBMHyperParams()
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
            lgbm_n_estimators=config.hyperparams.lgbm_params.n_estimators,
            lgbm_learning_rate=config.hyperparams.lgbm_params.learning_rate,
            lgbm_max_depth=config.hyperparams.lgbm_params.max_depth,
            lgbm_min_child_weight=config.hyperparams.lgbm_params.min_child_weight,
            lgbm_min_data_in_leaf=config.hyperparams.lgbm_params.min_data_in_leaf,
            lgbm_min_data_in_bin=config.hyperparams.lgbm_params.min_data_in_bin,
            lgbm_reg_alpha=config.hyperparams.lgbm_params.reg_alpha,
            lgbm_reg_lambda=config.hyperparams.lgbm_params.reg_lambda,
            lgbm_num_leaves=config.hyperparams.lgbm_params.num_leaves,
            lgbm_max_bin=config.hyperparams.lgbm_params.max_bin,
            lgbm_colsample_by_tree=config.hyperparams.lgbm_params.colsample_bytree,
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

    @staticmethod
    def _prepare_fit_input(data: ForecastInputDataset) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        input_data: pd.DataFrame = data.input_data()

        # Scale the target variable
        target: np.ndarray = np.asarray(data.target_series.values)

        # Prepare sample weights
        sample_weight: pd.Series = data.sample_weight_series

        return input_data, target, sample_weight

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the Hybrid model to the training data.

        Args:
            data: Training data in the expected ForecastInputDataset format.
            data_val: Validation data for tuning the model (optional, not used in this implementation).

        """
        # Prepare training data
        input_data, target, sample_weight = self._prepare_fit_input(data)

        if data_val is not None:
            logger.warning(
                "Validation data provided, but HybridForecaster does not currently support validation during fitting."
            )

        self._model.fit(
            X=input_data,
            y=target,
            sample_weight=sample_weight,
        )

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

    # TODO(@Lars800): #745: Make forecaster Explainable


__all__ = ["HybridForecaster", "HybridForecasterConfig", "HybridHyperParams"]
