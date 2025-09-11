# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

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
    constant: float = Field(
        default=0.01,
        description="Constant to add to the forecasts.",
    )


class ConstantMedianForecasterConfig(HorizonForecasterConfig):
    hyperparams: ConstantMedianForecasterHyperParams = Field(
        default=...,
    )


MODEL_CODE_VERSION = 2


class ConstantMedianState(BaseConfig):
    version: int = Field(default=MODEL_CODE_VERSION, description="State version for compatibility checks.")
    config: ConstantMedianForecasterConfig = Field(default=...)
    quantile_values: dict[Quantile, float] = Field(default={})


class ConstantMedianForecaster(BaseHorizonForecaster):
    def __init__(
        self,
        config: ConstantMedianForecasterConfig,
        state: ConstantMedianState | None = None,
    ) -> None:
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
                    str(quantile): self._state.quantile_values[quantile] + self.hyperparams.constant
                    for quantile in self.config.quantiles
                },
                index=(
                    input_data.index[input_data.index > pd.Timestamp(input_data.forecast_start)]
                    if input_data.forecast_start is not None
                    else input_data.index,
                ),
            ),
            sample_interval=input_data.sample_interval,
        )


class DummyMedianHorizonForecaster(
    MultiHorizonForecasterAdapter[ConstantMedianForecasterConfig, ConstantMedianForecaster]
):
    @classmethod
    @override
    def get_forecaster_type(cls) -> type[ConstantMedianForecaster]:
        return ConstantMedianForecaster

    @classmethod
    @override
    def create_forecaster(cls, config: ConstantMedianForecasterConfig) -> ConstantMedianForecaster:
        return ConstantMedianForecaster(config=config)
