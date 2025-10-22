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

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError, NotFittedError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig


class ConstantMedianForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for constant median forecaster."""

    constant: float = Field(
        default=0.01,
        description="Constant to add to the forecasts.",
    )


class ConstantMedianForecasterConfig(ForecasterConfig):
    """Configuration for constant median forecaster."""

    horizons: list[LeadTime] = Field(
        default=...,
        description=(
            "Lead times for predictions, accounting for data availability and versioning cutoffs. "
            "Each horizon defines how far ahead the model should predict."
        ),
        min_length=1,
        max_length=1,
    )

    hyperparams: ConstantMedianForecasterHyperParams = Field(
        default=ConstantMedianForecasterHyperParams(),
    )


MODEL_CODE_VERSION = 2


class ConstantMedianForecaster(Forecaster):
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

    _config: ConstantMedianForecasterConfig
    _quantile_values: dict[Quantile, float]

    def __init__(
        self,
        config: ConstantMedianForecasterConfig | None = None,
    ) -> None:
        """Initialize the constant median forecaster.

        Args:
            config: Configuration specifying quantiles and hyperparameters.
            state: Optional pre-trained state for restored models.
        """
        self._config = config or ConstantMedianForecasterConfig()
        self._quantile_values: dict[Quantile, float] = {}

    @property
    @override
    def config(self) -> ConstantMedianForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> ConstantMedianForecasterHyperParams:
        return self._config.hyperparams

    @override
    def to_state(self) -> State:
        return {
            "version": MODEL_CODE_VERSION,
            "config": self.config.model_dump(mode="json"),
            "quantile_values": self._quantile_values,
        }

    @override
    def from_state(self, state: State) -> Self:
        if not isinstance(state, dict) or "version" not in state or state["version"] > MODEL_CODE_VERSION:
            raise ModelLoadingError("Invalid state for ConstantMedianForecaster")

        # Gracefully migrate state from older model versions
        if state["version"] == 1:
            state["quantile_values"] = state["quantile_values_v1"]

        instance = self.__class__(config=ConstantMedianForecasterConfig.model_validate(state["config"]))
        instance._quantile_values = state["quantile_values"]  # noqa: SLF001
        return instance

    @property
    @override
    def is_fitted(self) -> bool:
        return len(self._quantile_values) > 0

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        self._quantile_values = {quantile: data.target_series.quantile(quantile) for quantile in self.config.quantiles}

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        forecast_index = data.create_forecast_range(horizon=self.config.max_horizon)
        return ForecastDataset(
            data=pd.DataFrame(
                data={
                    quantile.format(): self._quantile_values[quantile] + self.hyperparams.constant
                    for quantile in self.config.quantiles
                },
                index=forecast_index,
            ),
            sample_interval=data.sample_interval,
        )
