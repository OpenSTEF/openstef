# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


"""Gradient Boosted linear forecaster.

Provides a model that uses a linear booster (`gb_linear`) from the Gradient Boosting framework
for forecasting. This model does not suffer from the extrapolation issues of tree-based models
and can be more suitable for certain types of time series data where it is important
to predict values outside the range of the training data.
"""

import base64
import json
from functools import partial
from typing import Any, Literal, Self, cast, override

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import Field
from sklearn.preprocessing import StandardScaler

from openstef_core.datasets.mixins import LeadTime
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingExtraError, ModelLoadingError, NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.mixins.stateful import State
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.utils.loss_functions import OBJECTIVE_MAP, ObjectiveFunctionType, xgb_prepare_target_for_objective

try:
    import xgboost as xgb
except ImportError as e:
    raise MissingExtraError("xgboost", "openstef-models") from e


class GBLinearHyperParams(HyperParams):
    """Hyperparameter configuration for GBLinear forecaster."""

    # Learning Parameters
    n_steps: int = Field(
        default=500,
        description="Nomber for steps (boosting rounds) to train the GBLinear model.",
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
        default="pinball_loss",
        description="Objective function for training. 'pinball_loss' is recommended for probabilistic forecasting.",
    )

    # Regularization
    reg_alpha: float = Field(
        default=0.0001, description="L1 regularization on weights. Higher values increase regularization. Range: [0,∞]"
    )
    reg_lambda: float = Field(
        default=0.0, description="L2 regularization on weights. Higher values increase regularization. Range: [0,∞]"
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

        if self.config.hyperparams.objective == "reg:quantileerror":
            objective = "reg:quantileerror"
        else:
            objective = partial(OBJECTIVE_MAP[self._config.hyperparams.objective], quantiles=self._config.quantiles)

        self._gblinear_model = xgb.XGBRegressor(
            booster="gblinear",
            # Core parameters for forecasting
            objective=objective,
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

    @override
    def to_state(self) -> State:
        model_raw = self._gblinear_model.get_booster().save_raw()

        return {
            "config": self._config,
            "version": MODEL_CODE_VERSION,
            "model": base64.b64encode(model_raw).decode("utf-8"),
            "scaler": self._target_scaler,  # pyright: ignore[reportUnknownMemberType]
        }

    @override
    def from_state(self, state: State) -> Self:
        if not isinstance(state, dict) or "version" not in state:
            raise ModelLoadingError("Invalid state format")

        state = cast(dict[str, Any], state)
        if state["version"] > MODEL_CODE_VERSION:
            msg = f"Unsupported model version: {state['version']}"
            raise ModelLoadingError(msg)

        instance = self.__class__(config=state["config"])
        model_raw = bytearray(base64.b64decode(state["model"]))
        instance._gblinear_model.load_model(model_raw)  # pyright: ignore[reportUnknownMemberType]  # noqa: SLF001

        booster = instance._gblinear_model.get_booster()  # noqa: SLF001
        booster_config = json.loads(booster.save_config())
        loaded_booster_type = booster_config.get("learner", {}).get("gradient_booster", {}).get("name", "")
        if loaded_booster_type != "gblinear":
            msg = f"Invalid booster type in state: expected 'gblinear', got '{loaded_booster_type}'"
            raise ModelLoadingError(msg)

        instance._target_scaler = state["scaler"]  # noqa: SLF001

        return instance

    def _prepare_fit_input(self, data: ForecastInputDataset) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        input_data: pd.DataFrame = data.input_data()

        # Scale the target variable
        target: np.ndarray = np.asarray(data.target_series.values)
        target = self._target_scaler.transform(target.reshape(-1, 1)).flatten()  # pyright: ignore[reportUnknownMemberType]
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
        self._target_scaler.fit(target.reshape(-1, 1))  # pyright: ignore[reportUnknownMemberType]

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

        # Get input features for prediction
        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)

        # Generate predictions
        predictions_array: np.ndarray = self._gblinear_model.predict(input_data)

        # Inverse transform the scaled predictions
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
