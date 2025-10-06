# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Base case forecasting model that uses lag features for predictions.

Provides a simple baseline forecasting model that predicts using historical load values
from lag features created by LagTransform. This model serves as a naive baseline for
comparison with more sophisticated forecasting approaches, implementing the common
assumption that energy load patterns exhibit weekly periodicity.

The forecaster constructs lag column names based on hyperparameter configuration and
uses them to make predictions. It prioritizes the primary lag (default: 7-day) but falls
back to the fallback lag (default: 14-day) when primary lag data is not available.
"""

from datetime import timedelta
from typing import Self, cast, override

import pandas as pd
from pydantic import Field
from scipy.stats import norm

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_models.models.forecasting.forecaster import HorizonForecaster, HorizonForecasterConfig


class BaseCaseForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for base case forecaster."""

    primary_lag: timedelta = Field(
        default=timedelta(days=7),
        description="Primary lag to use for predictions (default: 7 days for weekly patterns)",
    )
    fallback_lag: timedelta = Field(
        default=timedelta(days=14),
        description="Fallback lag to use when primary lag data is unavailable (default: 14 days)",
    )


class BaseCaseForecasterConfig(HorizonForecasterConfig):
    """Configuration for base case forecaster."""

    hyperparams: BaseCaseForecasterHyperParams = Field(
        default_factory=BaseCaseForecasterHyperParams,
    )


MODEL_CODE_VERSION = 1


class BaseCaseForecaster(HorizonForecaster):
    """Base case forecaster that repeats weekly patterns for predictions.

    A simple baseline forecasting model that uses pandas-native operations to repeat
    the last week of historical target data for forecasting. This model serves as a
    naive baseline for comparison with more sophisticated forecasting approaches,
    implementing the common assumption that energy load patterns exhibit weekly periodicity.

    The forecaster takes the last week of historical data (based on primary_lag, default: 7 days)
    and uses pandas reindex with forward fill to repeat this weekly pattern until the end
    of the forecast period. Missing values are filled using the fallback lag period
    (default: 14 days, representing 2 weeks ago).

    The confidence intervals are calculated using hourly standard deviations computed from
    the repeated base case data, providing uncertainty estimates for each prediction.

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>>
        >>> # Default configuration (7-day primary, 14-day fallback)
        >>> config = BaseCaseForecasterConfig(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ... )
        >>> forecaster = BaseCaseForecaster(config)
        >>>
        >>> # Custom lag configuration
        >>> custom_config = BaseCaseForecasterConfig(
        ...     quantiles=[Quantile(0.5)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=BaseCaseForecasterHyperParams(
        ...         primary_lag=timedelta(days=7),
        ...         fallback_lag=timedelta(days=21)
        ...     )
        ... )
        >>> custom_forecaster = BaseCaseForecaster(custom_config)
        >>>
        >>> # Works directly with target variable in the dataset
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(forecast_data)  # doctest: +SKIP

    Note:
        The forecaster works directly with the target variable in the input dataset,
        automatically detecting the forecast period from the forecast_start parameter
        and repeating the appropriate historical weekly pattern.
    """

    _config: BaseCaseForecasterConfig

    def __init__(
        self,
        config: BaseCaseForecasterConfig,
    ) -> None:
        """Initialize the base case forecaster.

        Args:
            config: Configuration specifying quantiles, horizons, and lag hyperparameters.
                   If None, uses default configuration with 7-day primary and 14-day fallback lags.
        """
        self._config = config

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

        return self.__class__(config=BaseCaseForecasterConfig.model_validate(state["config"]))

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Base case forecaster is always "fitted" - no training required

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        # Base case forecaster requires no training - just validate data has target column
        pass

    def _get_basecase_values(self, data: ForecastInputDataset) -> pd.Series:
        """Get basecase values using pandas-native lag approach.

        Creates lagged target data by shifting timestamps, similar to LagTransform.
        Falls back to secondary lag if primary lag has missing values.

        Args:
            data: Input dataset containing target variable history and forecast range

        Returns:
            Series with basecase values for forecast timestamps
        """
        if data.forecast_start is None:
            # No forecast period specified - return empty series with DatetimeIndex
            empty_index = pd.DatetimeIndex([], freq=data.sample_interval)  # pyright: ignore[reportArgumentType] - bad type stubs
            return pd.Series(dtype=float, index=empty_index, name=data.target_column)

        # Get target series from historical data only (before forecast_start)
        # Use all available data but only up to forecast_start for lag calculations
        all_data = data.data
        historical_data = all_data[all_data.index < pd.Timestamp(data.forecast_start)]
        target_series = historical_data[data.target_column]

        # Create primary lag series (shift timestamps forward by primary_lag)
        # Following LagTransform approach: subtract negative lag from timestamps
        primary_lag_series = target_series.copy()
        primary_lag_series.index = cast(pd.DatetimeIndex, primary_lag_series.index) + self.hyperparams.primary_lag

        # Get forecast period data
        forecast_index = data.index[data.index >= pd.Timestamp(data.forecast_start)]
        primary_forecast = primary_lag_series.reindex(forecast_index)

        # Fill missing values with fallback lag if needed
        if primary_forecast.isna().any():
            # Create fallback lag series
            fallback_lag_series = target_series.copy()
            fallback_lag_series.index = (
                cast(pd.DatetimeIndex, fallback_lag_series.index) + self.hyperparams.fallback_lag
            )

            fallback_forecast = fallback_lag_series.reindex(forecast_index)
            primary_forecast = primary_forecast.fillna(fallback_forecast)  # pyright: ignore[reportUnknownMemberType]

        return primary_forecast

    @staticmethod
    def _calculate_hourly_std(basecase_values: pd.Series) -> pd.Series:
        """Calculate standard deviation for each hour using basecase values.

        Mimics the behavior of generate_basecase_confidence_interval from the old version.

        Args:
            basecase_values: Series containing the basecase values for calculating std.

        Returns:
            Series with hourly standard deviation values mapped to forecast timestamps.
        """
        if len(basecase_values.dropna()) == 0:
            # If no valid basecase values, return zero std
            return pd.Series(0.0, index=basecase_values.index)

        # Create DataFrame with values and hours for grouping
        basecase_value_with_hour = pd.DataFrame({
            "values": basecase_values,
            "hour": cast(pd.DatetimeIndex, basecase_values.index).hour,
        }).dropna()  # pyright: ignore[reportUnknownMemberType]

        if len(basecase_value_with_hour) == 0:
            return pd.Series(0.0, index=basecase_values.index)

        # Group by hour and calculate std
        hourly_std = basecase_value_with_hour.groupby("hour")["values"].std().fillna(0.0)  # pyright: ignore[reportUnknownMemberType]

        # Map std values to forecast timestamps
        forecast_hours = cast(pd.DatetimeIndex, basecase_values.index).hour
        return forecast_hours.map(hourly_std).fillna(0.0)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Generate predictions using repeated weekly patterns with confidence intervals.

        Args:
            data: The forecast input dataset containing target variable history.

        Returns:
            ForecastDataset containing quantile predictions for the forecast period.
        """
        # Get basecase values by repeating last week of target data
        basecase_values = self._get_basecase_values(data)

        # Calculate hourly standard deviation for confidence intervals
        hourly_std = self._calculate_hourly_std(basecase_values)

        # Create predictions for each quantile
        predictions_data = {}
        for quantile in self.config.quantiles:
            # Calculate z-score for the quantile
            z_score = norm.ppf(float(quantile))  # pyright: ignore[reportUnknownMemberType]

            # Apply confidence interval: base_value + z_score * std
            quantile_values = basecase_values + z_score * hourly_std
            predictions_data[quantile.format()] = quantile_values

        return ForecastDataset(
            data=pd.DataFrame(
                data=predictions_data,
                index=basecase_values.index,
            ),
            sample_interval=data.sample_interval,
        )
