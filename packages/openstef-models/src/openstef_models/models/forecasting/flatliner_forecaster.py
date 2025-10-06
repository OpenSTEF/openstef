# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant zero forecasting model.

Provides basic forecasting model that predict constant flatliner zero values. It can be used
when a flatline (non-)zero measurement is observed in the past and expected in the future.
"""

from typing import Self, override

import pandas as pd

from openstef_core.datasets import MultiHorizon
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError
from openstef_models.models.forecasting import Forecaster, ForecasterConfig


class FlatlinerForecasterConfig(ForecasterConfig):
    """Configuration for flatliner forecaster."""


MODEL_CODE_VERSION = 1


class FlatlinerForecaster(Forecaster):
    """Flatliner forecaster that predicts a flatline of zeros.

    A simple forecasting model that always predicts zero for all horizons and quantiles.

    Invariants:
        - Configuration quantiles determine the number of prediction outputs
        - Zeros are predicted for all horizons and quantiles

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> config = FlatlinerForecasterConfig(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=2))],
        ... )
        >>> forecaster = FlatlinerForecaster(config)
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP

    See Also:
        FlatlinerCheckTransform: Transform to detect flatliner patterns in time series data.
        Forecaster: Base class for forecasting models that predict multiple horizons.
    """

    _config: FlatlinerForecasterConfig

    def __init__(
        self,
        config: FlatlinerForecasterConfig | None = None,
    ) -> None:
        """Initialize the flatliner forecaster.

        Args:
            config: Configuration specifying quantiles and horizons.
        """
        self._config = config or FlatlinerForecasterConfig()

    @property
    @override
    def config(self) -> FlatlinerForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def to_state(self) -> object:
        return {
            "version": MODEL_CODE_VERSION,
            "config": self.config.model_dump(mode="json"),
        }

    @override
    def from_state(self, state: object) -> Self:
        if not isinstance(state, dict) or "version" not in state or state["version"] > MODEL_CODE_VERSION:
            raise ModelLoadingError("Invalid state for FlatlinerForecaster")

        return self.__class__(config=FlatlinerForecasterConfig.model_validate(state["config"]))

    @override
    def fit(
        self,
        data: MultiHorizon[ForecastInputDataset],
        data_val: MultiHorizon[ForecastInputDataset] | None = None,
    ) -> None:
        pass

    @override
    def predict(self, data: MultiHorizon[ForecastInputDataset]) -> ForecastDataset:
        input_data_list = [horizon_data.input_data(start=horizon_data.forecast_start) for horizon_data in data.values()]
        return ForecastDataset(
            data=pd.concat([
                pd.DataFrame(
                    data={quantile.format(): 0.0 for quantile in self.config.quantiles},
                    index=input_data.index,
                )
                for input_data in input_data_list
            ]),
            sample_interval=next(iter(data.values())).sample_interval,
        )
