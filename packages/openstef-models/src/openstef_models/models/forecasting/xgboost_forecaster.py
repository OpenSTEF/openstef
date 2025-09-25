# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""XGBoost-based forecasting models for probabilistic energy forecasting.

Provides gradient boosting tree models using XGBoost for multi-quantile energy
forecasting. Optimized for time series data with specialized loss functions and
comprehensive hyperparameter control for production forecasting workflows.
"""

import base64
import json
from functools import partial
from typing import Any, Literal, Self, cast, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingExtraError, ModelLoadingError, NotFittedError
from openstef_core.mixins import HyperParams, State
from openstef_models.models.forecasting import ForecasterConfig, HorizonForecaster, HorizonForecasterConfig
from openstef_models.utils.loss_functions import OBJECTIVE_MAP, ObjectiveFunctionType

try:
    import xgboost as xgb
except ImportError as e:
    raise MissingExtraError("xgboost", "openstef-models") from e


class XGBoostHyperParams(HyperParams):
    """XGBoost hyperparameters for gradient boosting tree models.

    Configures tree-specific parameters for XGBoost gbtree booster. Provides
    comprehensive control over model complexity, regularization, and training
    behavior for energy forecasting tasks.

    These parameters control tree structure, learning rates, regularization,
    and sampling strategies. Proper tuning is essential for balancing model
    performance and overfitting prevention in time series forecasting.

    Example:
        Creating custom hyperparameters for deep trees with regularization:

        >>> hyperparams = XGBoostHyperParams(
        ...     n_estimators=200,
        ...     max_depth=8,
        ...     learning_rate=0.1,
        ...     reg_alpha=0.1,
        ...     reg_lambda=1.0,
        ...     subsample=0.8
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
        default=6,
        description="Maximum depth of trees. Higher values capture more complex patterns but risk "
        "overfitting. Range: [1,∞]",
    )
    min_child_weight: float = Field(
        default=1,
        description="Minimum sum of instance weight (hessian) needed in a child. Higher values prevent "
        "overfitting. Range: [0,∞]",
    )
    gamma: float = Field(
        default=0,
        alias="min_split_loss",
        description="Minimum loss reduction required to make a split. Higher values make algorithm more "
        "conservative. Range: [0,∞]",
    )
    objective: ObjectiveFunctionType = Field(
        default="pinball_loss_magnitude_weighted",
        description="Objective function for training. 'pinball_loss_magnitude_weighted' is recommended for "
        "probabilistic forecasting.",
    )

    # Regularization
    reg_alpha: float = Field(
        default=0, description="L1 regularization on leaf weights. Higher values increase regularization. Range: [0,∞]"
    )
    reg_lambda: float = Field(
        default=1, description="L2 regularization on leaf weights. Higher values increase regularization. Range: [0,∞]"
    )
    max_delta_step: float = Field(
        default=0,
        description="Maximum delta step allowed for leaf weight estimation. Useful for logistic regression with "
        "imbalanced classes. Range: [0,∞]",
    )

    # Tree Structure Control
    max_leaves: int = Field(
        default=0, description="Maximum number of leaves. 0 means no limit. Only relevant when grow_policy='lossguide'."
    )
    grow_policy: Literal["depthwise", "lossguide"] = Field(
        default="depthwise",
        description="Controls how new nodes are added. 'depthwise' grows level by level, 'lossguide' adds leaves "
        "with highest loss reduction.",
    )
    max_bin: int = Field(
        default=256,
        description="Maximum number of discrete bins for continuous features. Higher values may improve accuracy but "
        "increase memory. Only for hist tree_method.",
    )
    num_parallel_trees: int = Field(
        default=1,
        description="Number of trees to grow per round. Higher values increase model complexity and training time. "
        "Range: [1,∞]",
    )

    # Subsampling Parameters
    subsample: float = Field(
        default=1.0,
        description="Fraction of training samples used for each tree. Lower values prevent overfitting. Range: (0,1]",
    )
    colsample_bytree: float = Field(
        default=1.0, description="Fraction of features used when constructing each tree. Range: (0,1]"
    )
    colsample_bylevel: float = Field(
        default=1.0, description="Fraction of features used for each level within a tree. Range: (0,1]"
    )
    colsample_bynode: float = Field(
        default=1.0, description="Fraction of features used for each split/node. Range: (0,1]"
    )

    # Tree Construction Method
    tree_method: Literal["auto", "exact", "hist", "approx", "gpu_hist"] = Field(
        default="auto",
        description="Tree construction algorithm. 'hist' is fastest for large datasets, 'exact' for small "
        "datasets, 'approx' is deprecated.",
    )

    # General Parameters
    random_state: int | None = Field(
        default=None, alias="seed", description="Random seed for reproducibility. Controls tree structure randomness."
    )

    early_stopping_rounds: int | None = Field(
        default=None,
        description="Training will stop if performance doesn't improve for this many rounds. Requires validation data.",
    )


class XGBoostForecasterConfig(HorizonForecasterConfig):
    """Configuration for XGBoost-based forecasting models.

    Combines hyperparameters with execution settings for XGBoost models.
    Controls both model training behavior and computational resources for
    efficient forecasting in production environments.

    Example:
        Creating a GPU-accelerated configuration:

        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime, Quantile
        >>> config = XGBoostForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=6))],
        ...     hyperparams=XGBoostHyperParams(n_estimators=500, max_depth=8),
        ...     device="cuda",
        ...     n_jobs=-1,
        ...     verbosity=2
        ... )
    """

    hyperparams: XGBoostHyperParams = XGBoostHyperParams()

    # General Parameters
    device: str = Field(
        default="cpu", description="Device for XGBoost computation. Options: 'cpu', 'cuda', 'cuda:<ordinal>', 'gpu'"
    )
    n_jobs: int = Field(
        default=1, description="Number of parallel threads for tree construction. -1 uses all available cores."
    )
    verbosity: Literal[0, 1, 2, 3] = Field(
        default=1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )


MODEL_CODE_VERSION = 1


class XGBoostForecasterState(BaseConfig):
    """Serializable state for XGBoost forecaster persistence.

    Contains all information needed to restore a trained XGBoost model,
    including configuration and the serialized model weights. Used for
    model saving, loading, and version management in production systems.
    """

    version: int = Field(default=MODEL_CODE_VERSION, description="Version of the model code.")
    config: XGBoostForecasterConfig = Field(..., description="Forecaster configuration.")
    model: str = Field(..., description="Base64-encoded serialized XGBoost model.")


class XGBoostForecaster(HorizonForecaster):
    """XGBoost-based forecaster for probabilistic energy forecasting.

    Implements gradient boosting trees using XGBoost for multi-quantile forecasting.
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
        >>> config = XGBoostForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=XGBoostHyperParams(n_estimators=100, max_depth=6)
        ... )
        >>> forecaster = XGBoostForecaster(config)
        >>> # forecaster.fit(training_data)
        >>> # predictions = forecaster.predict(test_data)

    Note:
        XGBoost dependency is optional and must be installed separately.
        The model automatically handles multi-quantile output and uses
        magnitude-weighted pinball loss by default for better forecasting performance.

    See Also:
        XGBoostHyperParams: Detailed hyperparameter configuration options.
        HorizonForecaster: Base interface for all forecasting models.
        GBLinearForecaster: Alternative linear model using XGBoost.
    """

    _config: XGBoostForecasterConfig
    _xgboost_model: xgb.XGBRegressor

    def __init__(self, config: XGBoostForecasterConfig) -> None:
        """Initialize XGBoost forecaster with configuration.

        Creates an untrained XGBoost regressor with the specified configuration.
        The underlying XGBoost model is configured for multi-output quantile
        regression using the provided hyperparameters and execution settings.

        Args:
            config: Complete configuration including hyperparameters, quantiles,
                and execution settings for the XGBoost model.
        """
        self._config = config

        objective = partial(OBJECTIVE_MAP[self._config.hyperparams.objective], quantiles=self._config.quantiles)

        self._xgboost_model = xgb.XGBRegressor(
            # Multi-output configuration
            multi_strategy="one_output_per_tree",
            # Core parameters for forecasting
            n_estimators=self._config.hyperparams.n_estimators,
            learning_rate=self._config.hyperparams.learning_rate,
            max_depth=self._config.hyperparams.max_depth,
            min_child_weight=self._config.hyperparams.min_child_weight,
            gamma=self._config.hyperparams.gamma,
            # Regularization parameters
            reg_alpha=self._config.hyperparams.reg_alpha,
            reg_lambda=self._config.hyperparams.reg_lambda,
            max_delta_step=self._config.hyperparams.max_delta_step,
            # Tree structure control
            max_leaves=self._config.hyperparams.max_leaves,
            grow_policy=self._config.hyperparams.grow_policy,
            max_bin=self._config.hyperparams.max_bin,
            num_parallel_trees=self._config.hyperparams.num_parallel_trees,
            # Subsampling parameters
            subsample=self._config.hyperparams.subsample,
            colsample_bytree=self._config.hyperparams.colsample_bytree,
            colsample_bylevel=self._config.hyperparams.colsample_bylevel,
            colsample_bynode=self._config.hyperparams.colsample_bynode,
            # Tree construction method
            tree_method=self._config.hyperparams.tree_method,
            # General parameters
            random_state=self._config.hyperparams.random_state,
            device=self._config.device,
            n_jobs=self._config.n_jobs,
            verbosity=self._config.verbosity,
            # Early stopping handled in fit method
            early_stopping_rounds=self._config.hyperparams.early_stopping_rounds,
            # Objective
            objective=objective,
        )

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> XGBoostHyperParams:
        return self._config.hyperparams

    @override
    def to_state(self) -> State:
        meta: dict[str, Any] = {}
        meta["_estimator_type"] = self._xgboost_model._get_type()  # noqa: SLF001
        meta_str = json.dumps(meta)
        self._xgboost_model.get_booster().set_attr(scikit_learn=meta_str)
        model_raw = self._xgboost_model.get_booster().save_raw()
        self._xgboost_model.get_booster().set_attr(scikit_learn=None)

        return XGBoostForecasterState(
            config=self._config,
            version=MODEL_CODE_VERSION,
            model=base64.b64encode(model_raw).decode("utf-8"),
        ).model_dump(mode="json")

    @override
    def from_state(self, state: State) -> Self:
        if not isinstance(state, dict) or "version" not in state:
            raise ModelLoadingError("Invalid state format")

        state = cast(dict[str, Any], state)
        if state["version"] > MODEL_CODE_VERSION:
            msg = f"Unsupported model version: {state['version']}"
            raise ModelLoadingError(msg)

        state_parsed: XGBoostForecasterState = XGBoostForecasterState.model_validate(state)
        instance = self.__class__(config=state_parsed.config)
        model_raw = bytearray(base64.b64decode(state_parsed.model))
        instance._xgboost_model.load_model(model_raw)  # pyright: ignore[reportUnknownMemberType]  # noqa: SLF001
        return instance

    @property
    @override
    def is_fitted(self) -> bool:
        return self._xgboost_model.__sklearn_is_fitted__()

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        input_data: pd.DataFrame = data.input_data()
        target: npt.NDArray[np.floating] = data.target_series().to_numpy()
        # Multi output setting requires a target per output (quantile)
        target_per_quantile: npt.NDArray[np.floating] = np.repeat(
            target[:, np.newaxis], repeats=len(self.config.quantiles), axis=1
        )
        sample_weight = data.sample_weight_series()

        # Prepare validation data if provided
        eval_set = None
        eval_set_sample_weight = None
        if data_val is not None:
            val_input_data: pd.DataFrame = data_val.input_data()
            val_target: npt.NDArray[np.floating] = data_val.target_series().to_numpy()
            val_target_per_quantile: npt.NDArray[np.floating] = np.repeat(
                val_target[:, np.newaxis], repeats=len(self.config.quantiles), axis=1
            )
            val_sample_weight = data_val.sample_weight_series()
            eval_set = [(val_input_data, val_target_per_quantile)]
            eval_set_sample_weight = [val_sample_weight]

        self._xgboost_model.fit(
            X=input_data,
            y=target_per_quantile,
            sample_weight=sample_weight,
            eval_set=eval_set,
            sample_weight_eval_set=eval_set_sample_weight,
            verbose=self._config.verbosity,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)
        prediction: npt.NDArray[np.floating] = self._xgboost_model.predict(X=input_data)

        return ForecastDataset(
            data=pd.DataFrame(
                data=prediction,
                index=input_data.index,
                columns=[quantile.format() for quantile in self.config.quantiles],
            ),
            sample_interval=data.sample_interval,
        )


__all__ = ["XGBoostForecaster", "XGBoostForecasterConfig", "XGBoostHyperParams"]
