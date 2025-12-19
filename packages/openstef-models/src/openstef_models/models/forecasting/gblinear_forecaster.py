# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


"""Gradient Boosted linear forecaster.

Provides a model that uses a linear booster (`gb_linear`) from the Gradient Boosting framework
for forecasting. This model does not suffer from the extrapolation issues of tree-based models
and can be more suitable for certain types of time series data where it is important
to predict values outside the range of the training data.
"""

from typing import Literal, override

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import Field
from sklearn.preprocessing import StandardScaler

from openstef_core.datasets.mixins import LeadTime
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import InputValidationError, MissingExtraError, NotFittedError
from openstef_core.mixins.predictor import HyperParams
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


class GBLinearHyperParams(HyperParams):
    """Hyperparameter configuration for GBLinear forecaster."""

    # Learning Parameters
    n_steps: int = Field(
        default=500,
        description="Number for steps (boosting rounds) to train the GBLinear model.",
    )
    updater: str = Field(
        default="shotgun",
        description="The updater to use for the GBLinear booster.",
    )
    learning_rate: float = Field(
        default=0.15,
        description="Step size shrinkage used to prevent overfitting. Range: [0,1]. Lower values require more boosting "
        "rounds.",
    )
    objective: ObjectiveFunctionType | Literal["reg:quantileerror"] = Field(
        default="reg:quantileerror",
        description="Objective function for training. 'reg:quantileerror' is recommended "
        "for probabilistic forecasting.",
    )
    evaluation_metric: EvaluationFunctionType = Field(
        default="mean_pinball_loss",
        description="Metric used for evaluation during training. Defaults to 'mean_pinball_loss' "
        "for quantile regression.",
    )

    # Regularization
    reg_alpha: float = Field(
        default=0.0001, description="L1 regularization on weights. Higher values increase regularization. Range: [0,∞]"
    )
    reg_lambda: float = Field(
        default=0.1, description="L2 regularization on weights. Higher values increase regularization. Range: [0,∞]"
    )

    # Feature selection
    feature_selector: str = Field(
        default="shuffle",
        description="Feature selection method.",
    )
    top_k: int = Field(
        default=0,
        description="Number of top features to select. 0 means using all features.",
    )

    # General Parameters
    random_state: int | None = Field(
        default=None, description="Random seed for reproducibility. Controls tree structure randomness."
    )
    early_stopping_rounds: int | None = Field(
        default=10,
        description="Training will stop if performance doesn't improve for this many rounds. Requires validation data.",
    )


class GBLinearForecasterConfig(ForecasterConfig):
    """Configuration for GBLinear forecaster."""

    horizons: list[LeadTime] = Field(
        default=...,
        description=(
            "Lead times for predictions, accounting for data availability and versioning cutoffs. "
            "Each horizon defines how far ahead the model should predict."
        ),
        min_length=1,
        max_length=1,
    )

    hyperparams: GBLinearHyperParams = Field(
        default=GBLinearHyperParams(),
    )
    device: str = Field(
        default="cpu", description="Device for XGBoost computation. Options: 'cpu', 'cuda', 'cuda:<ordinal>', 'gpu'"
    )
    verbosity: Literal[0, 1, 2, 3, True] = Field(
        default=1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )


MODEL_CODE_VERSION = 1


class GBLinearForecaster(Forecaster, ExplainableForecaster):
    """GBLinear-based forecaster for probabilistic energy forecasting.

    Implements gradient boosted linear models using XGBoost's `gblinear` booster for
    multi-quantile forecasting. Unlike tree-based models, this linear approach does not
    suffer from extrapolation issues and provides better performance for time series data
    where predictions outside the training range are required.

    The forecaster uses linear models with gradient boosting optimization, making it
    particularly suitable for forecasting scenarios where the underlying relationships
    are approximately linear or when avoiding extrapolation artifacts is critical.
    This approach provides well-calibrated uncertainty estimates while maintaining
    computational efficiency and interpretability.

    Invariants:
        - fit() must be called before predict() to train the model
        - Configuration quantiles determine the number of prediction outputs
        - Model state is preserved across predict() calls after fitting
        - Input features must match training data structure during prediction

    Example:
        Basic forecasting workflow:

        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime, Quantile
        >>> config = GBLinearForecasterConfig(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=GBLinearHyperParams(
        ...         learning_rate=0.1,
        ...         reg_alpha=0.1,
        ...         reg_lambda=1.0
        ...     )
        ... )
        >>> forecaster = GBLinearForecaster(config)
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP

    Note:
        XGBoost dependency is optional and must be installed separately.
        The model automatically handles multi-quantile output using quantile regression
        and is optimized for energy forecasting applications where linear relationships
        dominate and extrapolation beyond training data is required.

    See Also:
        GBLinearHyperParams: Detailed hyperparameter configuration options.
        HorizonForecaster: Base interface for all forecasting models.
        XGBoostForecaster: Tree-based alternative for non-linear patterns.
    """

    Config = GBLinearForecasterConfig
    HyperParams = GBLinearHyperParams

    _config: GBLinearForecasterConfig
    _gblinear_model: xgb.XGBRegressor
    _target_scaler: StandardScaler

    def __init__(self, config: GBLinearForecasterConfig) -> None:
        """Initialize GBLinear forecaster with configuration.

        Args:
            config: Configuration for the forecaster.
        """
        self._config = config or GBLinearForecasterConfig()

        self._gblinear_model = xgb.XGBRegressor(
            booster="gblinear",
            # Core parameters for forecasting
            n_estimators=self._config.hyperparams.n_steps,
            learning_rate=self._config.hyperparams.learning_rate,
            early_stopping_rounds=self._config.hyperparams.early_stopping_rounds,
            # Regularization parameters
            reg_alpha=self._config.hyperparams.reg_alpha,
            reg_lambda=self._config.hyperparams.reg_lambda,
            # Boosting structure control
            feature_selector=self._config.hyperparams.feature_selector,
            updater=self._config.hyperparams.updater,
            quantile_alpha=[float(q) for q in self._config.quantiles],
            top_k=self._config.hyperparams.top_k if self._config.hyperparams.feature_selector == "thrifty" else None,
            # Objective
            objective=get_objective_function(
                function_type=self._config.hyperparams.objective, quantiles=self._config.quantiles
            )
            if self._config.hyperparams.objective != "reg:quantileerror"
            else "reg:quantileerror",
            eval_metric=get_evaluation_function(
                function_type=self._config.hyperparams.evaluation_metric, quantiles=self._config.quantiles
            ),
            disable_default_eval_metric=True,
        )
        self._target_scaler = StandardScaler()

    @property
    @override
    def config(self) -> GBLinearForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> GBLinearHyperParams:
        return self._config.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self._gblinear_model.__sklearn_is_fitted__()

    def _prepare_fit_input(self, data: ForecastInputDataset) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        input_data: pd.DataFrame = data.input_data()
        # Scale the target variable
        target: np.ndarray = np.asarray(data.target_series.values)
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
        # Data checks
        if data.data.isna().any().any():
            raise InputValidationError("There are nan values in the input data. Use imputation transform to fix them.")

        if len(data.data) == 0:
            raise InputValidationError("The input data is empty after dropping NaN values.")

        # Fit the scalers
        self._target_scaler.fit(data.target_series.to_frame())

        # Prepare training data
        input_data, target, sample_weight = self._prepare_fit_input(data)

        # Evaluation sets
        eval_set = [(input_data, target)]
        sample_weight_eval_set = [sample_weight]

        if data_val is not None:
            input_data_val, target_val, sample_weight_val = self._prepare_fit_input(data_val)
            eval_set.append((input_data_val, target_val))
            sample_weight_eval_set.append(sample_weight_val)

        self._gblinear_model.fit(
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

        # Data checks
        if data.input_data().isna().any().any():
            raise InputValidationError("There are nan values in the input data. Use imputation transform to fix them.")

        # Get input features for prediction
        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)

        # Generate predictions
        predictions_array: np.ndarray = self._gblinear_model.predict(input_data).reshape(-1, len(self.config.quantiles))

        # Inverse transform the scaled predictions
        if len(predictions_array) > 0:
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

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        booster = self._gblinear_model.get_booster()
        weights_df = pd.DataFrame(
            data=booster.get_score(importance_type="weight"),
            index=[quantile.format() for quantile in self.config.quantiles],
        ).transpose()
        weights_df.index.name = "feature_name"
        weights_df.columns.name = "quantiles"

        weights_abs = weights_df.abs()
        total = weights_abs.sum(axis=0).replace(to_replace=0, value=1.0)  # pyright: ignore[reportUnknownMemberType]

        return weights_abs / total
