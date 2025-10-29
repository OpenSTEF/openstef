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
from typing import override

import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig


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


class BaseCaseForecasterConfig(ForecasterConfig):
    """Configuration for base case forecaster."""

    quantiles: list[Quantile] = Field(
        default=[Quantile(0.5)],
        description=(
            "Probability levels for uncertainty estimation. Each quantile represents a confidence level "
            "(e.g., 0.1 = 10th percentile, 0.5 = median, 0.9 = 90th percentile). "
            "Models must generate predictions for all specified quantiles."
        ),
        min_length=1,
        max_length=1,
    )
    horizons: list[LeadTime] = Field(
        default=...,
        description=(
            "Lead times for predictions, accounting for data availability and versioning cutoffs. "
            "Each horizon defines how far ahead the model should predict."
        ),
        min_length=1,
        max_length=1,
    )

    hyperparams: BaseCaseForecasterHyperParams = Field(
        default_factory=BaseCaseForecasterHyperParams,
    )


MODEL_CODE_VERSION = 1


class BaseCaseForecaster(Forecaster, ExplainableForecaster):
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
        ...     quantiles=[Quantile(0.1)],
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

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Base case forecaster is always "fitted" - no training required

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        pass  # Base case forecaster requires no training - just validate data has target column

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Generate predictions using repeated weekly patterns with confidence intervals.

        Args:
            data: The forecast input dataset containing target variable history.

        Returns:
            ForecastDataset containing quantile predictions for the forecast period.
        """
        # The range to forecast
        forecast_index = data.create_forecast_range(horizon=self.config.max_horizon)

        # Get target series from historical data only (before forecast_start)
        # Use all available data but only up to forecast_start for lag calculations
        target_series = data.target_series[data.index < pd.Timestamp(data.forecast_start)]

        # Create primary lag series (shift timestamps forward by primary_lag)
        # Following LagTransform approach: subtract negative lag from timestamps
        prediction = target_series.shift(freq=self.hyperparams.primary_lag).reindex(forecast_index)

        # Fill missing values with fallback lag if needed
        if prediction.isna().any():
            # Create fallback lag series
            prediction_fallback = target_series.shift(freq=self.hyperparams.fallback_lag).reindex(forecast_index)
            prediction = prediction.fillna(prediction_fallback)  # pyright: ignore[reportUnknownMemberType]

        return ForecastDataset(
            data=pd.DataFrame(
                {
                    self.config.quantiles[0].format(): prediction,
                },
                index=forecast_index,
            ),
            sample_interval=data.sample_interval,
        )

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[1.0],
            index=["load"],
            columns=[quantile.format() for quantile in self.config.quantiles],
        )
