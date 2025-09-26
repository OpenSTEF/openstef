# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Base case forecasting model that predicts load from 1 week ago.

Provides a simple baseline forecasting model that predicts the load value from
exactly 1 week ago when available. This model serves as a naive
baseline for comparison with more sophisticated forecasting approaches, implementing
the common assumption that energy load patterns exhibit weekly periodicity.

The forecaster looks back 7 days to find historical load values and uses them
as predictions for future time periods. When historical data is not available
for a specific time point, the model returns NaN for that prediction.
"""

from datetime import timedelta
from typing import Self, override, cast

import pandas as pd
from pydantic import Field
from scipy.stats import norm

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError, NotFittedError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.forecaster import HorizonForecaster, HorizonForecasterConfig


class BaseCaseForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for base case forecaster."""

    lookback_days: int = Field(
        default=7,
        description="Number of days to look back for base case predictions.",
        ge=1,
    )


class BaseCaseForecasterConfig(HorizonForecasterConfig):
    """Configuration for base case forecaster."""

    hyperparams: BaseCaseForecasterHyperParams = Field(
        default=BaseCaseForecasterHyperParams(),
    )


MODEL_CODE_VERSION = 1


class BaseCaseForecaster(HorizonForecaster):
    """Base case forecaster that predicts load from 1 week ago.

    A simple baseline forecasting model that leverages weekly periodicity in energy
    load patterns. For each prediction time point, it looks back exactly 1 week
    (168 hours) to find the historical load value and uses it as the prediction.

    This forecaster is particularly useful for:
    - Establishing performance baselines for more complex models
    - Quick estimates when sophisticated models are unavailable
    - Educational purposes to demonstrate weekly load patterns
    - Initial model validation and sanity checking

    The model assigns the same predicted value to all quantiles, as it doesn't
    provide uncertainty estimates beyond point predictions.

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> config = BaseCaseForecasterConfig(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=BaseCaseForecasterHyperParams(lookback_weeks=1)
        ... )
        >>> forecaster = BaseCaseForecaster(config)
        >>> # forecaster.fit(training_data)
        >>> # predictions = forecaster.predict(test_data)
    """

    _config: BaseCaseForecasterConfig

    def __init__(
        self,
        config: BaseCaseForecasterConfig | None = None,
    ) -> None:
        """Initialize the base case forecaster.

        Args:
            config: Configuration specifying quantiles and hyperparameters.
        """
        self._config = config or BaseCaseForecasterConfig(
            quantiles=[Quantile(0.5)],  # Default median quantile
            horizons=[LeadTime(timedelta(hours=1))],  # Default 1-hour horizon
        )

    @property
    @override
    def config(self) -> BaseCaseForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> BaseCaseForecasterHyperParams:
        return self._config.hyperparams

    @override
    def to_state(self) -> State:
        return {
            "version": MODEL_CODE_VERSION,
            "config": self.config.model_dump(mode="json"),
        }

    @override
    def from_state(self, state: State) -> Self:
        if not isinstance(state, dict) or "version" not in state or state["version"] > MODEL_CODE_VERSION:
            raise ModelLoadingError("Invalid state for BaseCaseForecaster")

        return self.__class__(config=BaseCaseForecasterConfig.model_validate(state["config"]))  # noqa: SLF001

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        pass

    def _create_quantile_predictions(
        self,
        forecast_index: pd.DatetimeIndex,
        historical_values: pd.Series,
        hourly_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Create predictions for each quantile.

        Args:
            forecast_index: The forecast period index.
            historical_values: Historical values for each forecast timestamp.
            hourly_stats: Hourly statistics for confidence intervals.

        Returns:
            Dictionary mapping quantile strings to prediction series.
        """
        predictions_data = {}

        for quantile in self.config.quantiles:
            # Calculate z-score for the quantile
            z_score = norm.ppf(float(quantile))

            # Apply confidence interval: base_value + z_score * std
            quantile_values = historical_values + z_score * hourly_stats["std"]
            predictions_data[quantile.format()] = quantile_values

        return pd.DataFrame(
            data=predictions_data,
            index=forecast_index,
        )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Generate predictions using historical values from lookback period with confidence intervals.

        Args:
            data: The forecast input dataset containing historical data and target column.

        Returns:
            ForecastDataset containing quantile predictions for the forecast period.
        """
        # Get forecast timestamps
        forecast_index = data.input_data().index

        historical_values_shifted = data.target_series().shift(periods=self._config.hyperparams.lookback_days, freq="D")
        if data.forecast_start is not None:
            historical_values_shifted = (
                historical_values_shifted[historical_values_shifted.index > pd.Timestamp(data.forecast_start)]
            )

        historical_with_hour = historical_values_shifted.to_frame()
        historical_with_hour["hour"] = cast(pd.DatetimeIndex, historical_with_hour.index).hour

        # Compute mean and std for each hour
        historical_with_hour["std"] = historical_with_hour.groupby([
            historical_with_hour.index.date,
            historical_with_hour.index.hour,
        ])[data.target_column].transform("std").fillna(0)

        # Create quantile predictions
        predictions_data = self._create_quantile_predictions(
            forecast_index, historical_values_shifted, historical_with_hour
        )

        return ForecastDataset(
            data=predictions_data,
            sample_interval=data.sample_interval,
        )
