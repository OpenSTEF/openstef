# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant zero forecasting model.

Provides basic forecasting model that predict constant flatliner zero values. It can be used
when a flatline (non-)zero measurement is observed in the past and expected in the future.
"""

from typing import override

import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig


class FlatlinerForecasterConfig(ForecasterConfig):
    """Configuration for flatliner forecaster."""

    predict_median: bool = Field(
        default=False,
        description="If True, predict the median of load measurements instead of zero.",
    )


MODEL_CODE_VERSION = 1


class FlatlinerForecaster(Forecaster, ExplainableForecaster):
    """Flatliner forecaster that predicts a flatline of zeros or median.

    A simple forecasting model that always predicts zero (or the median of historical
    load measurements if configured) for all horizons and quantiles.

    Invariants:
        - Configuration quantiles determine the number of prediction outputs
        - Zeros (or median values) are predicted for all horizons and quantiles

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
        FlatlineChecker: Transform to detect flatliner patterns in time series data.
        Forecaster: Base class for forecasting models that predict multiple horizons.
    """

    Config = FlatlinerForecasterConfig

    _config: FlatlinerForecasterConfig
    _median_value: float | None

    def __init__(
        self,
        config: FlatlinerForecasterConfig | None = None,
    ) -> None:
        """Initialize the flatliner forecaster.

        Args:
            config: Configuration specifying quantiles and horizons.
        """
        self._config = config or FlatlinerForecasterConfig()
        self._median_value = None

    @property
    @override
    def config(self) -> FlatlinerForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        # When predict_median is True, the model needs to be fitted to compute the median
        if self._config.predict_median:
            return self._median_value is not None
        return True

    @override
    def fit(
        self,
        data: ForecastInputDataset,
        data_val: ForecastInputDataset | None = None,
    ) -> None:
        if self._config.predict_median:
            self._median_value = float(data.target_series.median())

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        forecast_index = data.create_forecast_range(horizon=self.config.max_horizon)

        prediction_value = self._median_value if self._config.predict_median else 0.0

        return ForecastDataset(
            data=pd.DataFrame(
                data={quantile.format(): prediction_value for quantile in self.config.quantiles},
                index=forecast_index,
            ),
            sample_interval=data.sample_interval,
        )

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[0.0],
            index=["load"],
            columns=[quantile.format() for quantile in self.config.quantiles],
        )
