# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""LightGBM-based forecasting models for probabilistic energy forecasting.

Provides gradient boosting tree models using LightGBM for multi-quantile energy
forecasting. Optimized for time series data with specialized loss functions and
comprehensive hyperparameter control for production forecasting workflows.
"""

from typing import Literal, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_models.estimators.lgbm import LGBMQuantileRegressor
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig


class LGBMLinearHyperParams(HyperParams):
    """LgbLinear hyperparameters for gradient boosting tree models.

    Example:
        Creating custom hyperparameters for deep trees with regularization:

        >>> hyperparams = LGBMLinearHyperParams(
        ...     n_estimators=200,
        ...     max_depth=8,
        ...     learning_rate=0.1,
        ...     reg_alpha=0.1,
        ...     reg_lambda=1.0,
        ... )

    Note:
        These parameters are optimized for probabilistic forecasting with
        quantile regression. The default objective function is specialized
        for magnitude-weighted pinball loss.
    """

    # Core Tree Boosting Parameters

    n_estimators: int = Field(
        default=10,
        description="Number of boosting rounds/trees to fit. Higher values may improve performance but "
        "increase training time and risk overfitting.",
    )
    learning_rate: float = Field(
        default=0.3,
        alias="eta",
        description="Step size shrinkage used to prevent overfitting. Range: [0,1]. Lower values require "
        "more boosting rounds.",
    )
    max_depth: int = Field(
        default=4,
        description="Maximum depth of trees. Higher values capture more complex patterns but risk "
        "overfitting. Range: [1,∞]",
    )
    min_child_weight: float = Field(
        default=1,
        description="Minimum sum of instance weight (hessian) needed in a child. Higher values prevent "
        "overfitting. Range: [0,∞]",
    )

    min_data_in_leaf: int = Field(
        default=5,
        description="Minimum number of data points in a leaf. Higher values prevent overfitting. Range: [1,∞]",
    )
    min_data_in_bin: int = Field(
        default=5,
        description="Minimum number of data points in a bin. Higher values prevent overfitting. Range: [1,∞]",
    )

    # Regularization
    reg_alpha: float = Field(
        default=0,
        description="L1 regularization on leaf weights. Higher values increase regularization. Range: [0,∞]",
    )
    reg_lambda: float = Field(
        default=1,
        description="L2 regularization on leaf weights. Higher values increase regularization. Range: [0,∞]",
    )

    # Tree Structure Control
    num_leaves: int = Field(
        default=31,
        description="Maximum number of leaves. 0 means no limit. Only relevant when grow_policy='lossguide'.",
    )

    max_bin: int = Field(
        default=256,
        description="Maximum number of discrete bins for continuous features. Higher values may improve accuracy but "
        "increase memory. Only for hist tree_method.",
    )

    # Subsampling Parameters
    colsample_bytree: float = Field(
        default=0.5,
        description="Fraction of features used when constructing each tree. Range: (0,1]",
    )

    # General Parameters
    random_state: int | None = Field(
        default=None,
        alias="seed",
        description="Random seed for reproducibility. Controls tree structure randomness.",
    )

    early_stopping_rounds: int | None = Field(
        default=10,
        description="Training will stop if performance doesn't improve for this many rounds. Requires validation data.",
    )


class LGBMLinearForecasterConfig(ForecasterConfig):
    """Configuration for LgbLinear-based forecaster.
    Extends HorizonForecasterConfig with LgbLinear-specific hyperparameters
    and execution settings.

    Example:
    Creating a LgbLinear forecaster configuration with custom hyperparameters:
    >>> from datetime import timedelta
    >>> from openstef_core.types import LeadTime, Quantile
    >>> config = LGBMLinearForecasterConfig(
    ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
    ...     horizons=[LeadTime(timedelta(hours=1))],
    ...     hyperparams=LGBMLinearHyperParams(n_estimators=100, max_depth=6)
    ... )
    """  # noqa: D205

    hyperparams: LGBMLinearHyperParams = LGBMLinearHyperParams()

    # General Parameters
    device: str = Field(
        default="cpu",
        description="Device for LgbLinear computation. Options: 'cpu', 'cuda', 'cuda:<ordinal>', 'gpu'",
    )
    n_jobs: int = Field(
        default=1,
        description="Number of parallel threads for tree construction. -1 uses all available cores.",
    )
    verbosity: Literal[-1, 0, 1, 2, 3] = Field(
        default=-1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )


MODEL_CODE_VERSION = 1


class LGBMLinearForecasterState(BaseConfig):
    """Serializable state for LgbLinear forecaster persistence.

    Contains all information needed to restore a trained LgbLinear model,
    including configuration and the serialized model weights. Used for
    model saving, loading, and version management in production systems.
    """

    version: int = Field(default=MODEL_CODE_VERSION, description="Version of the model code.")
    config: LGBMLinearForecasterConfig = Field(..., description="Forecaster configuration.")
    model: str = Field(..., description="Base64-encoded serialized LgbLinear model.")


class LGBMLinearForecaster(Forecaster, ExplainableForecaster):
    """LgbLinear-based forecaster for probabilistic energy forecasting.

    Implements gradient boosting trees using LgbLinear for multi-quantile forecasting.
    Optimized for time series prediction with specialized loss functions and
    comprehensive hyperparameter control suitable for production energy forecasting.

    The forecaster uses a multi-output strategy where each quantile is predicted
    by separate trees within the same boosting ensemble. This approach provides
    well-calibrated uncertainty estimates while maintaining computational efficiency.

    Invariants:
        - fit() must be called before predict() to train the model
        - Configuration quantiles determine the number of prediction outputs
        - Model state is preserved across predict() calls after fitting
        - Input features must match training data structure during prediction

    Example:
        Basic forecasting workflow:

        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime, Quantile
        >>> config = LGBMLinearForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=LGBMLinearHyperParams(n_estimators=100, max_depth=6)
        ... )
        >>> forecaster = LGBMLinearForecaster(config)
        >>> # forecaster.fit(training_data)
        >>> # predictions = forecaster.predict(test_data)

    Note:
        LgbLinear dependency is optional and must be installed separately.
        The model automatically handles multi-quantile output and uses
        magnitude-weighted pinball loss by default for better forecasting performance.

    See Also:
        LGBMLinearHyperParams: Detailed hyperparameter configuration options.
        HorizonForecaster: Base interface for all forecasting models.
        GBLinearForecaster: Alternative linear model using LgbLinear.
    """

    Config = LGBMLinearForecasterConfig
    HyperParams = LGBMLinearHyperParams

    _config: LGBMLinearForecasterConfig
    _lgbmlinear_model: LGBMQuantileRegressor

    def __init__(self, config: LGBMLinearForecasterConfig) -> None:
        """Initialize LgbLinear forecaster with configuration.

        Creates an untrained LgbLinear regressor with the specified configuration.
        The underlying LgbLinear model is configured for multi-output quantile
        regression using the provided hyperparameters and execution settings.

        Args:
            config: Complete configuration including hyperparameters, quantiles,
                and execution settings for the LgbLinear model.
        """
        self._config = config

        self._lgbmlinear_model = LGBMQuantileRegressor(
            quantiles=[float(q) for q in config.quantiles],
            linear_tree=True,
            n_estimators=config.hyperparams.n_estimators,
            learning_rate=config.hyperparams.learning_rate,
            max_depth=config.hyperparams.max_depth,
            min_child_weight=config.hyperparams.min_child_weight,
            min_data_in_leaf=config.hyperparams.min_data_in_leaf,
            min_data_in_bin=config.hyperparams.min_data_in_bin,
            reg_alpha=config.hyperparams.reg_alpha,
            reg_lambda=config.hyperparams.reg_lambda,
            num_leaves=config.hyperparams.num_leaves,
            max_bin=config.hyperparams.max_bin,
            colsample_bytree=config.hyperparams.colsample_bytree,
            random_state=config.hyperparams.random_state,
            early_stopping_rounds=config.hyperparams.early_stopping_rounds,
            verbosity=config.verbosity,
        )

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> LGBMLinearHyperParams:
        return self._config.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self._lgbmlinear_model.__sklearn_is_fitted__()

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        input_data: pd.DataFrame = data.input_data()
        target: npt.NDArray[np.floating] = data.target_series.to_numpy()  # type: ignore
        sample_weight = data.sample_weight_series

        # Prepare validation data if provided
        eval_set = None
        eval_sample_weight = None
        if data_val is not None:
            val_input_data: pd.DataFrame = data_val.input_data()
            val_target: npt.NDArray[np.floating] = data_val.target_series.to_numpy()  # type: ignore
            val_sample_weight = cast(npt.NDArray[np.floating], data_val.sample_weight_series.to_numpy())  # type: ignore
            eval_set = [(val_input_data, val_target)]

            eval_sample_weight = [val_sample_weight]

        self._lgbmlinear_model.fit(  # type: ignore
            X=input_data,
            y=target,
            feature_name=input_data.columns.tolist(),
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=eval_sample_weight,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)
        prediction: npt.NDArray[np.floating] = self._lgbmlinear_model.predict(X=input_data)

        return ForecastDataset(
            data=pd.DataFrame(
                data=prediction,
                index=input_data.index,
                columns=[quantile.format() for quantile in self.config.quantiles],
            ),
            sample_interval=data.sample_interval,
        )

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        models = self._lgbmlinear_model._models  # noqa: SLF001
        weights_df = pd.DataFrame(
            [models[i].feature_importances_ for i in range(len(models))],
            index=[quantile.format() for quantile in self.config.quantiles],
            columns=models[0].feature_name_,
        ).transpose()

        weights_df.index.name = "feature_name"
        weights_df.columns.name = "quantiles"

        weights_abs = weights_df.abs()
        total = weights_abs.sum(axis=0).replace(to_replace=0, value=1.0)  # pyright: ignore[reportUnknownMemberType]

        return weights_abs / total


__all__ = ["LGBMLinearForecaster", "LGBMLinearForecasterConfig", "LGBMLinearHyperParams"]
