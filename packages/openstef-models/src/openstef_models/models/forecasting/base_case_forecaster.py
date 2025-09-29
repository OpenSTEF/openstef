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
back to the fallback lag (default: 14-day) when primary lag data is not available,
matching the behavior of the legacy create_basecase_forecast_pipeline.
"""

from datetime import timedelta
from typing import Self, cast, override

import pandas as pd
from pydantic import Field
from scipy.stats import norm

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingColumnsError, ModelLoadingError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_core.utils import timedelta_to_isoformat
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
    """Base case forecaster that uses lag features for predictions.

    A simple baseline forecasting model that leverages lag features created by LagTransform
    to make predictions. This model serves as a naive baseline for comparison with more
    sophisticated forecasting approaches, implementing the common assumption that energy
    load patterns exhibit weekly periodicity.

    The forecaster constructs lag column names based on the configured lags (primary_lag and
    fallback_lag hyperparameters) and the target column name. It prioritizes the primary lag
    features (default: 7-day lag, e.g., 'load_lag_-P7D') but falls back to fallback lag
    features (default: 14-day lag, e.g., 'load_lag_-P14D') when primary lag data is not available.

    The confidence intervals are calculated using hourly standard deviations computed from
    the lag data, providing uncertainty estimates for each prediction.

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
        >>> # Assumes data has lag columns like 'load_lag_-P7D', 'load_lag_-P14D'
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP

    Note:
        The forecaster expects lag columns to be present in the input data, typically
        created by applying LagTransform during preprocessing. The column names are
        constructed as: {target_column}_lag_{ISO8601_duration}

    See Also:
        LagTransform: A transformation that creates lag features for time series forecasting.
    """

    _config: BaseCaseForecasterConfig

    def __init__(
        self,
        config: BaseCaseForecasterConfig | None = None,
    ) -> None:
        """Initialize the base case forecaster.

        Args:
            config: Configuration specifying quantiles, horizons, and lag hyperparameters.
                   If None, uses default configuration with 7-day primary and 14-day fallback lags.
        """
        self._config = config or BaseCaseForecasterConfig()
        self._lag_columns_detected: bool = False
        self._primary_lag_target_column_name: str | None = None
        self._fallback_lag_target_column_name: str | None = None

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

    @staticmethod
    def _construct_lag_column_name(target_column: str, lag: timedelta) -> str:
        """Constructs the column name for the lag target feature.

        Args:
            target_column: The name of the target column
            lag: The timedelta lag

        Returns:
            The lag feature column name
        """
        # Convert timedelta to ISO format (e.g., P7D becomes -P7D for negative lag)
        iso_lag = timedelta_to_isoformat(-lag)
        return f"{target_column}_lag_{iso_lag}"

    @property
    @override
    def is_fitted(self) -> bool:
        return self._lag_columns_detected

    def _detect_lag_columns(self, data: ForecastInputDataset) -> None:
        """Detect lag columns in the input data.

        Looks for columns matching patterns like 'load_lag_-P7D', 'load_lag_-P14D', etc.

        Args:
            data: The input dataset containing features and target variable.

        Raises:
            MissingColumnsError: If both primary and fallback lag columns are not available.
        """
        self._primary_lag_target_column_name = self._construct_lag_column_name(
            data.target_column, self._config.hyperparams.primary_lag
        )

        self._fallback_lag_target_column_name = self._construct_lag_column_name(
            data.target_column, self._config.hyperparams.fallback_lag
        )

        # Check if we have at least one lag column available
        has_primary = self._primary_lag_target_column_name in data.feature_names
        has_fallback = self._fallback_lag_target_column_name in data.feature_names

        if not has_primary and not has_fallback:
            raise MissingColumnsError([self._primary_lag_target_column_name, self._fallback_lag_target_column_name])

        # Mark that we have detected lag columns
        self._lag_columns_detected = True

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        # Detect lag columns during fitting
        self._detect_lag_columns(data)

    def _get_basecase_values(self, data: ForecastInputDataset) -> pd.Series:
        """Get basecase values from available lag columns.

        Prioritizes primary lag column (7-day), falls back to fallback lag column (14-day).

        Args:
            data: Input dataset containing lag features

        Returns:
            Series with basecase values for prediction timestamps, or NaN series if no lag columns
        """
        forecast_data = data.input_data(start=data.forecast_start)

        # If no lag columns were detected, return NaN series
        if not self._lag_columns_detected:
            return pd.Series(float("nan"), index=forecast_data.index)

        # Try primary lag column first
        if self._primary_lag_target_column_name in data.feature_names:
            primary_values = forecast_data[self._primary_lag_target_column_name]

            # If primary has gaps, fill with fallback where available
            if self._fallback_lag_target_column_name and self._fallback_lag_target_column_name in forecast_data.columns:
                fallback_values = forecast_data[self._fallback_lag_target_column_name]
                return primary_values.fillna(fallback_values)  # pyright: ignore[reportUnknownMemberType]

            return primary_values

        # Fall back to fallback lag column if available
        if self._fallback_lag_target_column_name in data.feature_names:
            return forecast_data[self._fallback_lag_target_column_name]

        # If we reach here, no lag columns are available
        return pd.Series(float("nan"), index=forecast_data.index)

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
        """Generate predictions using lag features with confidence intervals.

        Args:
            data: The forecast input dataset containing lag features.

        Returns:
            ForecastDataset containing quantile predictions for the forecast period.
        """
        # Ensure lag columns are available
        self._detect_lag_columns(data)

        # Get basecase values from lag columns
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
