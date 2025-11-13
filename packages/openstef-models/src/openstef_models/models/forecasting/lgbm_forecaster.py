# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""LightGBM-based forecasting models for probabilistic energy forecasting.

Provides gradient boosting tree models using LightGBM for multi-quantile energy
forecasting. Optimized for time series data with specialized loss functions and
comprehensive hyperparameter control for production forecasting workflows.
"""

from typing import TYPE_CHECKING, Literal, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_models.estimators.lgbm import LGBMQuantileRegressor
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig

if TYPE_CHECKING:
    import numpy.typing as npt


class LGBMHyperParams(HyperParams):
    """LightGBM hyperparameters for gradient boosting tree models.

    Example:
        Creating custom hyperparameters for deep trees with regularization:

        >>> hyperparams = LGBMHyperParams(
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
        default=100,
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
        default=5,
        description="Maximum depth of trees. Higher values capture more complex patterns but risk "
        "overfitting. Range: [1,∞]",
    )
    min_child_weight: float = Field(
        default=1,
        description="Minimum sum of instance weight (hessian) needed in a child. Higher values prevent "
        "overfitting. Range: [0,∞]",
    )

    min_data_in_leaf: int = Field(
        default=10,
        description="Minimum number of data points in a leaf. Higher values prevent overfitting. Range: [1,∞]",
    )
    min_data_in_bin: int = Field(
        default=10,
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
        default=1.0,
        description="Fraction of features used when constructing each tree. Range: (0,1]",
    )


class LGBMForecasterConfig(ForecasterConfig):
    """Configuration for LightGBM-based forecaster.
    Extends HorizonForecasterConfig with LightGBM-specific hyperparameters
    and execution settings.

    Example:
    Creating a LightGBM forecaster configuration with custom hyperparameters:
    >>> from datetime import timedelta
    >>> from openstef_core.types import LeadTime, Quantile
    >>> config = LGBMForecasterConfig(
    ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
    ...     horizons=[LeadTime(timedelta(hours=1))],
    ...     hyperparams=LGBMHyperParams(n_estimators=100, max_depth=6))
    """  # noqa: D205

    hyperparams: LGBMHyperParams = LGBMHyperParams()

    # General Parameters
    device: str = Field(
        default="cpu",
        description="Device for LightGBM computation. Options: 'cpu', 'cuda', 'cuda:<ordinal>', 'gpu'",
    )
    n_jobs: int = Field(
        default=1,
        description="Number of parallel threads for tree construction. -1 uses all available cores.",
    )
    verbosity: Literal[-1, 0, 1, 2, 3] = Field(
        default=-1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )

    random_state: int | None = Field(
        default=None,
        alias="seed",
        description="Random seed for reproducibility. Controls tree structure randomness.",
    )

    early_stopping_rounds: int | None = Field(
        default=None,
        description="Training will stop if performance doesn't improve for this many rounds. Requires validation data.",
    )


MODEL_CODE_VERSION = 1


class LGBMForecaster(Forecaster, ExplainableForecaster):
    """LightGBM-based forecaster for probabilistic energy forecasting.

    Implements gradient boosting trees using LightGBM for multi-quantile forecasting.
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
        >>> config = LGBMForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=LGBMHyperParams(n_estimators=100, max_depth=6)
        ... )
        >>> forecaster = LGBMForecaster(config)
        >>> # forecaster.fit(training_data)
        >>> # predictions = forecaster.predict(test_data)

    Note:
        LightGBM dependency is optional and must be installed separately.
        The model automatically handles multi-quantile output and uses
        magnitude-weighted pinball loss by default for better forecasting performance.

    See Also:
        LGBMHyperParams: Detailed hyperparameter configuration options.
        HorizonForecaster: Base interface for all forecasting models.
        GBLinearForecaster: Alternative linear model using LightGBM.
    """

    Config = LGBMForecasterConfig
    HyperParams = LGBMHyperParams

    _config: LGBMForecasterConfig
    _lgbm_model: LGBMQuantileRegressor

    def __init__(self, config: LGBMForecasterConfig) -> None:
        """Initialize LightGBM forecaster with configuration.

        Creates an untrained LightGBM regressor with the specified configuration.
        The underlying LightGBM model is configured for multi-output quantile
        regression using the provided hyperparameters and execution settings.

        Args:
            config: Complete configuration including hyperparameters, quantiles,
                and execution settings for the LightGBM model.
        """
        self._config = config

        self._lgbm_model = LGBMQuantileRegressor(
            quantiles=[float(q) for q in config.quantiles],
            linear_tree=False,
            random_state=config.random_state,
            early_stopping_rounds=config.early_stopping_rounds,
            verbosity=config.verbosity,
            **config.hyperparams.model_dump(),
        )

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> LGBMHyperParams:
        return self._config.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self._lgbm_model.__sklearn_is_fitted__()

    @staticmethod
    def _prepare_fit_input(data: ForecastInputDataset) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        input_data: pd.DataFrame = data.input_data()
        target: np.ndarray = np.asarray(data.target_series.values)
        sample_weight: pd.Series = data.sample_weight_series

        return input_data, target, sample_weight

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        # Prepare training data
        input_data, target, sample_weight = self._prepare_fit_input(data)

        # Evaluation sets
        eval_set = [(input_data, target)]
        sample_weight_eval_set = [sample_weight]

        if data_val is not None:
            input_data_val, target_val, sample_weight_val = self._prepare_fit_input(data_val)
            eval_set.append((input_data_val, target_val))
            sample_weight_eval_set.append(sample_weight_val)

        self._lgbm_model.fit(
            X=input_data,
            y=target,
            feature_name=input_data.columns.tolist(),
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=sample_weight_eval_set,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)
        prediction: npt.NDArray[np.floating] = self._lgbm_model.predict(X=input_data)

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
        models = self._lgbm_model.models
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


__all__ = ["LGBMForecaster", "LGBMForecasterConfig", "LGBMHyperParams"]
