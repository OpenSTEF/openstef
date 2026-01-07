# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Median regressor based forecasting models for energy forecasting.

Provides median regression models for multi-quantile energy forecasting.
Optimized for time series data with specialized loss functions and
comprehensive hyperparameter control for production forecasting workflows.

Note that this is a autoregressive model, meaning that it uses the previous
    predictions to predict the next value.

    This regressor is good for predicting two types of signals:
    - Signals with very slow dynamics compared to the sampling rate, possibly
      with a lot of noise.
    - Signals that switch between two or more states, which random in nature or
    depend on unknown features, but tend to be stable in each state. An example of
    this may be waste heat delivered from an industrial process. Using a median
    over the last few timesteps adds some hysterisis to avoid triggering on noise.

    Tips for using this regressor:
    - Set the lags to be evenly spaced and at a frequency mathching the
      frequency of the input data. For example, if the input data is at 15
      minute intervals, set the lags to be at 15 minute intervals as well.
    - Use a small training dataset, since there are no actual parameters to train.
    - Set the frequency of the input data index to avoid inferring it. Inference might be
    a problem if we get very small chunks of data in training or validation sets.
    - Use only one training horizon, since the regressor will use the same lags for all
      training horizons.
    - Allow for missing data by setting completeness_threshold to 0. If the prediction horizon
      is larger than the context window there will be a lot of nans in the input data, but
      the autoregression solves that.
"""

from datetime import timedelta
from typing import override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_core.utils.pydantic import timedelta_from_isoformat
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig


class MedianForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for base case forecaster."""

    primary_lag: timedelta = Field(
        default=timedelta(days=7),
        description="Primary lag to use for predictions (default: 7 days for weekly patterns)",
    )
    fallback_lag: timedelta = Field(
        default=timedelta(days=14),
        description="Fallback lag to use when primary lag data is unavailable (default: 14 days)",
    )


class MedianForecasterConfig(ForecasterConfig):
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

    hyperparams: MedianForecasterHyperParams = Field(
        default_factory=MedianForecasterHyperParams,
    )


class MedianForecaster(Forecaster, ExplainableForecaster):
    """Median forecaster using lag features for predictions.

    This forecaster predicts the median value based on specified lag features.
    It is particularly useful for signals with slow dynamics or state-switching behavior.

    Hyperparameters:
        lags: List of time deltas representing the lag features to use for prediction.
              These should be aligned with the data sampling frequency.
        context_window: Time delta representing the context window size for input data.
                        This defines how much historical data is considered for making predictions.

                _config: MedianForecasterConfig
    """

    Config = MedianForecasterConfig
    HyperParams = MedianForecasterHyperParams

    def __init__(
        self,
        config: MedianForecasterConfig,
    ) -> None:
        """Initialize the base case forecaster.

        Args:
            config: Configuration specifying quantiles, horizons, and lag hyperparameters.
                   If None, uses default configuration with 7-day primary and 14-day fallback lags.
        """
        self._config = config
        self.is_fitted_ = False
        self.feature_names_: list[str] = []

    @property
    @override
    def config(self) -> MedianForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> MedianForecasterHyperParams:
        return self._config.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:

        return pd.DataFrame(
            data=self.feature_importances_, columns=[self.config.quantiles[0].format()], index=self.feature_names_
        )

    @property
    def frequency(self) -> int:
        """Retrieve the model input frequency.

        Returns:
            The frequency of the model input

        """
        return self.frequency_

    @staticmethod
    def _fill_diagonal_with_median(lag_array: np.ndarray, start: int, end: int, median: float) -> np.ndarray | None:
        # Use the calculated median to fill in future lag values where this prediction would be used as input.

        # If the start index is beyond the array bounds, no future updates are needed from this step.
        if start >= lag_array.shape[0]:
            return lag_array

        # Ensure the end index does not exceed the array bounds.
        end = min(end, lag_array.shape[0])

        # Get a view of the sub-array where the diagonal needs to be filled.
        # The slice represents future time steps (rows) and corresponding lag features (columns).
        # Rows: from 'start' up to (but not including) 'end'
        # Columns: from 0 up to (but not including) 'end - start'
        # This selects the part of the array where lag_array[start + k, k] resides for k in range(end - start).
        view = lag_array[start:end, 0 : (end - start)]

        # Create a mask for NaNs on the diagonal
        diagonal_nan_mask = np.isnan(np.diag(view))

        # Only update if there are NaNs on the diagonal
        if np.any(diagonal_nan_mask):
            # Create a temporary array to hold the new diagonal
            updated_diagonal = np.diag(view).copy()
            updated_diagonal[diagonal_nan_mask] = median
            np.fill_diagonal(view, updated_diagonal)
        return None

    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> pd.Timedelta:
        """Infer the frequency of a pandas DatetimeIndex if the freq attribute is not set.
        This method calculates the most common time difference between consecutive timestamps,
        which is more permissive of missing chunks of data than the pandas infer_freq method.

        Args:
            index (pd.DatetimeIndex): The datetime index to infer the frequency from.

        Returns:
            pd.Timedelta: The inferred frequency as a pandas Timedelta.
        """
        if len(index) < 2:
            raise ValueError(
                "Cannot infer frequency from an index with fewer than 2 timestamps."
            )

        # Calculate the differences between consecutive timestamps
        deltas = index.to_series().diff().dropna()

        # Find the most common difference
        inferred_freq = deltas.mode().iloc[0]
        return inferred_freq

    def _frequency_matches(self, index: pd.DatetimeIndex) -> bool:
        """Check if the frequency of the input data matches the model frequency.

        Args:
            index (pd.DatetimeIndex): The input data to check.

        Returns:
            bool: True if the frequencies match, False otherwise.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(
                "The index of the input data must be a pandas DatetimeIndex."
            )

        if index.freq is None:
            input_frequency = self._infer_frequency(index)
        else:
            input_frequency = index.freq

        return input_frequency == self.frequency

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Predict the median of the lag features for each time step in the context window.

        Args:
            data (ForecastInputDataset): The input data for prediction.
            This should be a pandas dataframe with lag features.

        Returns:
            np.array: The predicted median for each time step in the context window.
            If any lag feature is NaN, this will be ignored.
            If all lag features are NaN, the regressor will return NaN.

        Raises:
            ValueError: If the input data is missing any of the required lag features.
        """
        if not self.is_fitted:
            msg = "This MedianForecaster instance is not fitted yet"
            raise AttributeError(msg)

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)

        # Check that the input data contains the required lag features
        missing_features = set(self.feature_names_) - set(data.feature_names)
        if missing_features:
            msg = f"The input data is missing the following lag features: {missing_features}"
            raise ValueError(msg)

        if not self._frequency_matches(data.input_data().index):
            msg = (
                f"The frequency of the input data does not match the model frequency. "
                f"Input data frequency: {data.input_data().index.freq}, "
                f"Model frequency: {pd.Timedelta(minutes=self.frequency_)}"
            )
            raise ValueError(msg)

        # Reindex the input data to ensure there are no gaps in the time series.
        # This is important for the autoregressive logic that follows.
        # Store the original index to return predictions aligned with the input.
        old_index = input_data.index
        # Create a new date range with the expected frequency.
        new_index = pd.date_range(input_data.index[0], input_data.index[-1], freq=self.frequency_)
        # Reindex the input DataFrame, filling any new timestamps with NaN.
        input_data = input_data.reindex(new_index, fill_value=np.nan)

        # Select only the lag feature columns in the specified order.
        lag_df = input_data[self.feature_names_]

        # Convert the lag DataFrame and its index to NumPy arrays for faster processing.
        lag_array = lag_df.to_numpy()
        # Initialize the prediction array with NaNs.
        prediction = np.full(lag_array.shape[0], np.nan)

        # Calculate the time step size based on the model frequency.
        step_size = self.frequency_
        # Determine the number of steps corresponding to the smallest and largest lags.
        smallest_lag_steps = int(self.lags_to_time_deltas_[self.feature_names_[0]] / step_size)
        largest_lag_steps = int(self.lags_to_time_deltas_[self.feature_names_[-1]] / step_size)

        # Iterate through each time step in the reindexed data.
        for time_step in range(lag_array.shape[0]):
            # Get the lag features for the current time step.
            current_lags = lag_array[time_step]
            # Calculate the median of the available lag features, ignoring NaNs.
            median = np.nanmedian(current_lags)
            # If the median calculation resulted in NaN (e.g., all lags were NaN), skip the autoregression step.
            if not np.isnan(median):
                median = float(median)
            else:
                continue

            # Store the calculated median in the prediction array.
            prediction[time_step] = median

            # Auto-regressive step: update the lag array for future time steps.
            # Calculate the start and end indices in the future time steps that will be affected.
            start, end = (
                time_step + smallest_lag_steps,
                time_step + largest_lag_steps + 1,
            )
            self._fill_diagonal_with_median(lag_array, start, end, median)

        # Convert the prediction array back to a pandas DataFrame using the reindexed time index.
        prediction_df = pd.DataFrame(prediction, index=lag_df.index.to_numpy(), columns=["median"])

        # Reindex the prediction DataFrame back to the original input data index.
        prediction_df = prediction_df.reindex(old_index)

        return ForecastDataset(
            data=prediction_df.dropna().rename(columns={"median": self.config.quantiles[0].format()}),
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Take car of fitting the median forecaster.

        This regressor does not need any fitting,
        but it does need to know the feature names of the lag features and the order of these.

        Lag features are expected to be evently spaced and match the frequency of the input data.
        The lag features are expected to be named in the format T-<lag_in_minutes> or T-<lag_in_days>d.
        For example, T-1min, T-2min, T-3min or T-1d, T-2d.

        Which lag features are used is determined by the feature engineering step.
        """
        self.frequency_ = data.sample_interval
        # Check that the frequency of the input data matches frequency of the lags
        if not self._frequency_matches(
            data.data.index.drop_duplicates()
        ):  # Several training horizons give duplicates
            raise ValueError(
                f"The input data frequency ({data.data.index.freq}) does not match the model frequency ({self.frequency_})."
            )

        lag_perfix = f"{data.target_column}_lag_"
        self.feature_names_ = [
            feature_name for feature_name in data.feature_names if feature_name.startswith(lag_perfix)
        ]

        if not self.feature_names_:
            msg = f"No lag features found in the input data with prefix '{lag_perfix}'."
            raise ValueError(msg)

        self.lags_to_time_deltas_ = {
            feature_name: timedelta_from_isoformat(feature_name.replace(lag_perfix, ""))
            for feature_name in self.feature_names_
        }

        # Check if lags are evenly spaced
        lag_deltas = sorted(self.lags_to_time_deltas_.values())
        lag_intervals = [
            (lag_deltas[i] - lag_deltas[i - 1]).total_seconds() for i in range(1, len(lag_deltas))
        ]
        if not all(interval == lag_intervals[0] for interval in lag_intervals):
            msg = "Lag features are not evenly spaced. Please ensure lag features are evenly spaced and match the data frequency."
            raise ValueError(msg)

        self.feature_names_ = sorted(self.feature_names_, key=lambda f: self.lags_to_time_deltas_[f])

        self.feature_importances_ = np.ones(len(self.feature_names_)) / (len(self.feature_names_) or 1.0)
        self.is_fitted_ = True
