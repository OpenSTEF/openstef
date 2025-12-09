# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""XGBoost-based forecasting models for probabilistic energy forecasting.

Provides gradient boosting tree models using XGBoost for multi-quantile energy
forecasting. Optimized for time series data with specialized loss functions and
comprehensive hyperparameter control for production forecasting workflows.
"""

from typing import Literal, override

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.preprocessing import StandardScaler

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingExtraError, NotFittedError
from openstef_core.mixins import HyperParams
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.utils.evaluation_functions import EvaluationFunctionType, get_evaluation_function
from openstef_models.utils.loss_functions import (
    ObjectiveFunctionType,
    get_objective_function,
    xgb_prepare_target_for_objective,
)

try:
    import xgboost as xgb
except ImportError as e:
    raise MissingExtraError("xgboost", "openstef-models") from e


class XGBoostHyperParams(HyperParams):
    """XGBoost hyperparameters for gradient boosting tree models.

    Configures tree-specific parameters for XGBoost gbtree booster. Provides
    control over model complexity, regularization, and training
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
        default="pinball_loss",
        description="Objective function for training. 'pinball_loss' is recommended for probabilistic forecasting.",
    )
    evaluation_metric: EvaluationFunctionType = Field(
        default="mean_pinball_loss",
        description="Metric used for evaluation during training. Defaults to 'mean_pinball_loss' "
        "for quantile regression.",
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
        default=42, description="Random seed for reproducibility. Controls tree structure randomness."
    )
    early_stopping_rounds: int | None = Field(
        default=None,
        description="Training will stop if performance doesn't improve for this many rounds. Requires validation data.",
    )
    use_target_scaling: bool = Field(
        default=True,
        description="Whether to apply standard scaling to the target variable before training. Improves convergence.",
    )

    @classmethod
    def forecaster_class(cls) -> "type[XGBoostForecaster]":
        """Get the forecaster class for these hyperparams.

        Returns:
            Forecaster class associated with this configuration.
        """
        return XGBoostForecaster


class XGBoostForecasterConfig(ForecasterConfig):
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
    verbosity: Literal[0, 1, 2, 3, True] = Field(
        default=1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )

    def forecaster_from_config(self) -> "XGBoostForecaster":
        """Create a XGBoost forecaster instance from this configuration.

        Returns:
            Forecaster instance associated with this configuration.
        """
        return XGBoostForecaster(config=self)


MODEL_CODE_VERSION = 1


class XGBoostForecaster(Forecaster, ExplainableForecaster):
    """XGBoost-based forecaster for probabilistic energy forecasting.

    Implements gradient boosting trees using XGBoost for multi-quantile forecasting.
    Optimized for time series prediction with specialized loss functions and
    hyperparameter control suitable for production energy forecasting.

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

    Config = XGBoostForecasterConfig
    HyperParams = XGBoostHyperParams

    _config: XGBoostForecasterConfig
    _xgboost_model: xgb.XGBRegressor
    _target_scaler: StandardScaler | None

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
            objective=get_objective_function(
                function_type=self._config.hyperparams.objective, quantiles=self._config.quantiles
            ),
            eval_metric=get_evaluation_function(
                function_type=self._config.hyperparams.evaluation_metric, quantiles=self._config.quantiles
            ),
            disable_default_eval_metric=True,
        )
        self._target_scaler = StandardScaler() if self._config.hyperparams.use_target_scaling else None

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> XGBoostHyperParams:
        return self._config.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self._xgboost_model.__sklearn_is_fitted__()

    def _prepare_fit_input(self, data: ForecastInputDataset) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        input_data: pd.DataFrame = data.input_data()

        # Scale the target variable
        target: np.ndarray = np.asarray(data.target_series.values)
        if self._target_scaler is not None:
            target = self._target_scaler.transform(target.reshape(-1, 1)).flatten()
        # Reshape target for multi-quantile objectives
        target = xgb_prepare_target_for_objective(
            target=target,
            quantiles=self.config.quantiles,
            objective=self._config.hyperparams.objective,
        )

        # Prepare sample weights
        sample_weight: pd.Series = data.sample_weight_series

        return input_data, target, sample_weight

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        # Fit the target scaler
        target: np.ndarray = np.asarray(data.target_series.values)
        if self._target_scaler is not None:
            self._target_scaler.fit(target.reshape(-1, 1))

        # Prepare training data
        input_data, target, sample_weight = self._prepare_fit_input(data)

        # Evaluation sets
        eval_set = [(input_data, target)]
        sample_weight_eval_set = [sample_weight]

        if data_val is not None:
            input_data_val, target_val, sample_weight_val = self._prepare_fit_input(data_val)
            eval_set.append((input_data_val, target_val))
            sample_weight_eval_set.append(sample_weight_val)

        self._xgboost_model.fit(
            X=input_data,
            y=target,
            sample_weight=sample_weight,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            verbose=self._config.verbosity,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Get input features for prediction
        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)

        # Generate predictions
        predictions_array: np.ndarray = self._xgboost_model.predict(input_data).reshape(-1, len(self.config.quantiles))

        # Inverse transform the scaled predictions
        if self._target_scaler is not None and len(predictions_array) > 0:
            predictions_array = self._target_scaler.inverse_transform(predictions_array)

        # Construct DataFrame with appropriate quantile columns
        predictions = pd.DataFrame(
            data=predictions_array,
            index=input_data.index,
            columns=[quantile.format() for quantile in self.config.quantiles],
        )

        return ForecastDataset(
            data=predictions,
            sample_interval=data.sample_interval,
        )

    def predict_contributions(self, data: ForecastInputDataset, *, scale: bool) -> pd.DataFrame:
        """Get feature contributions for each prediction.

        Args:
            data: Input dataset for which to compute feature contributions.
            scale: If True, scale contributions to sum to 1.0 per quantile.

        Returns:
            DataFrame with contributions per base learner.
        """
        # Get input features for prediction
        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)
        xgb_input: xgb.DMatrix = xgb.DMatrix(data=input_data)

        # Generate predictions
        booster = self._xgboost_model.get_booster()
        predictions_array: np.ndarray = booster.predict(xgb_input, pred_contribs=True, strict_shape=True)[:, :, :-1]

        # Remove last column
        contribs = predictions_array / np.sum(predictions_array, axis=-1, keepdims=True)

        # Flatten to 2D array, name columns accordingly
        contribs = contribs.reshape(contribs.shape[0], -1)

        df = pd.DataFrame(
            data=contribs,
            index=input_data.index,
            columns=[
                f"{feature}_{quantile.format()}" for feature in input_data.columns for quantile in self.config.quantiles
            ],
        )

        if scale:
            # Scale contributions so that they sum to 1.0 per quantile and are positive
            for q in self.config.quantiles:
                quantile_cols = [col for col in df.columns if col.endswith(f"_{q.format()}")]
                row_sums = df[quantile_cols].abs().sum(axis=1)
                df[quantile_cols] = df[quantile_cols].abs().div(row_sums, axis=0)

        # Construct DataFrame with appropriate quantile columns
        return df

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        booster = self._xgboost_model.get_booster()
        weights_df = pd.DataFrame(
            data=booster.get_score(importance_type="gain"),
            index=[quantile.format() for quantile in self.config.quantiles],
        ).transpose()
        weights_df.index.name = "feature_name"
        weights_df.columns.name = "quantiles"

        weights_abs = weights_df.abs()
        total = weights_abs.sum(axis=0).replace(to_replace=0, value=1.0)  # pyright: ignore[reportUnknownMemberType]

        return weights_abs / total


__all__ = ["XGBoostForecaster", "XGBoostForecasterConfig", "XGBoostHyperParams"]
