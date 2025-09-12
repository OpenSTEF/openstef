# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant median forecasting models for educational and baseline purposes.

Provides basic forecasting models that predict constant values based on historical
medians. These models serve as educational examples and performance baselines for
more sophisticated forecasting approaches.
"""

from typing import Self, override

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError, ModelNotFittedError
from openstef_core.types import Quantile
from openstef_models.models.forecasting.mixins import (
    BaseHorizonForecaster,
    ForecasterHyperParams,
    HorizonForecasterConfig,
    ModelState,
)
from openstef_models.models.forecasting.multi_horizon_adapter import MultiHorizonForecasterAdapter


class ConstantMedianForecasterHyperParams(ForecasterHyperParams):
    """Hyperparameter configuration for constant median forecaster."""

    constant: float = Field(
        default=0.01,
        description="Constant to add to the forecasts.",
    )


class ConstantMedianForecasterConfig(HorizonForecasterConfig):
    """Configuration for constant median forecaster."""

    hyperparams: ConstantMedianForecasterHyperParams = Field(
        default=...,
    )


MODEL_CODE_VERSION = 2


class ConstantMedianState(BaseConfig):
    """Serializable state for constant median forecaster."""

    version: int = Field(default=MODEL_CODE_VERSION, description="State version for compatibility checks.")
    config: ConstantMedianForecasterConfig = Field(default=...)
    quantile_values: dict[Quantile, float] = Field(default={})


class ConstantMedianForecaster(BaseHorizonForecaster):
    """Constant median-based forecaster for single horizon predictions.

    Predicts constant values based on historical quantiles from training data.
    Useful as a baseline model and for educational purposes.

    The forecaster computes quantile values during training and returns these
    constant values for all future predictions. Performance is typically poor
    but provides a simple baseline for comparison with more sophisticated models.

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> config = ConstantMedianForecasterConfig(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=ConstantMedianForecasterHyperParams()
        ... )
        >>> forecaster = ConstantMedianForecaster(config)
        >>> # forecaster.fit_horizon(training_data)
        >>> # predictions = forecaster.predict_horizon(test_data)
    """

    def __init__(
        self,
        config: ConstantMedianForecasterConfig,
        state: ConstantMedianState | None = None,
    ) -> None:
        """Initialize the constant median forecaster.

        Args:
            config: Configuration specifying quantiles and hyperparameters.
            state: Optional pre-trained state for restored models.
        """
        self._state: ConstantMedianState = state if state is not None else ConstantMedianState(config=config)

    @property
    @override
    def config(self) -> ConstantMedianForecasterConfig:
        return self._state.config

    @property
    @override
    def hyperparams(self) -> ConstantMedianForecasterHyperParams:
        return self._state.config.hyperparams

    @override
    def get_state(self) -> ModelState:
        return self._state.model_dump(mode="json")

    @classmethod
    @override
    def from_state(cls, state: ModelState) -> Self:
        if not isinstance(state, dict) or "version" not in state or state["version"] > MODEL_CODE_VERSION:
            raise ModelLoadingError("Invalid state for ConstantMedianForecaster")

        try:
            # Gracefully migrate state from older model versions
            if state["version"] == 1:
                state["quantile_values"] = state["quantile_values_v1"]

            state = ConstantMedianState.model_validate(state)
        except Exception as e:
            raise ModelLoadingError("Failed to validate state") from e

        return cls(config=state.config, state=state)

    @property
    @override
    def is_fitted(self) -> bool:
        return len(self._state.quantile_values) > 0

    @override
    def fit_horizon(self, input_data: ForecastInputDataset) -> None:
        self._state = ConstantMedianState(
            config=self.config,
            quantile_values={
                quantile: input_data.target_series().quantile(quantile) for quantile in self.config.quantiles
            },
        )

    @override
    def predict_horizon(self, input_data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise ModelNotFittedError(self.__class__.__name__)

        return ForecastDataset(
            data=pd.DataFrame(
                data={
                    quantile.format(): self._state.quantile_values[quantile] + self.hyperparams.constant
                    for quantile in self.config.quantiles
                },
                index=(
                    input_data.index[input_data.index > pd.Timestamp(input_data.forecast_start)]
                    if input_data.forecast_start is not None
                    else input_data.index
                ),
            ),
            sample_interval=input_data.sample_interval,
        )


class ConstantMedianHorizonForecaster(
    MultiHorizonForecasterAdapter[ConstantMedianForecasterConfig, ConstantMedianForecaster]
):
    """Multi-horizon adapter for constant median forecasting.

    Creates separate ConstantMedianForecaster models for each prediction horizon
    and combines their outputs. Each horizon-specific model learns its own median
    values from training data, then the adapter stitches results together by using
    each model's predictions for its designated time range.
    """

    @classmethod
    @override
    def get_forecaster_type(cls) -> type[ConstantMedianForecaster]:
        return ConstantMedianForecaster

    @classmethod
    @override
    def create_forecaster(cls, config: ConstantMedianForecasterConfig) -> ConstantMedianForecaster:
        return ConstantMedianForecaster(config=config)
