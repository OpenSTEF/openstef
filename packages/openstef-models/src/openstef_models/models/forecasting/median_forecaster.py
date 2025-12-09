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
from typing import Self, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
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

                _config: BaseCaseForecasterConfig
    """

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

    @property
    @override
    def config(self) -> MedianForecasterConfig:
        return self._config

    @property
    @override
    def hyperparams(self) -> MedianForecasterHyperParams:
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

        return self.__class__(config=MedianForecasterConfig.model_validate(state["config"]))

    @property
    @override
    def is_fitted(self) -> bool:
        return self._xgboost_model.__sklearn_is_fitted__() 



    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        booster = self._xgboost_model.get_booster()
        weights_df = pd.DataFrame(
            data=booster.get_score(importance_type="gain"),
            index=[quantile.format() for quantile in self.config.quantiles],
        ).transpose()
        weights_df.index.name = "feature_name"
        weights_df.columns.name = "quantiles"

        weights_abs = weights_df.abs()
        total = weights_abs.sum(axis=0).replace(to_replace=0, value=1.0)  # pyright: ignore[reportUnknownMemberType]

        return weights_abs / total



    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> pd.Timedelta:
        """
        Infer the frequency of a pandas DatetimeIndex if the freq attribute is not set.
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
        """
        Check if the frequency of the input data matches the model frequency.

        Args:
            x (pd.DataFrame): The input data to check.

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

        return input_frequency == pd.Timedelta(minutes=self.frequency)

    @staticmethod
    def _extract_and_validate_lags(
        x: pd.DataFrame,
    ) -> tuple[tuple[str], int, list[tuple[str, int]]]:
        """Extract and validate the lag features from the input data.

        This method checks that the lag features are evenly spaced and match the frequency of the input data.
        It also extracts the lag features and their corresponding time deltas.
        Args:
            x (pd.DataFrame): The input data containing lag features.
        Returns:
            tuple: A tuple containing:
                - A list of feature names, sorted by their lag in minutes.
                - The frequency of the lag features in minutes.
                - A list of tuples containing the lag feature names and their corresponding time deltas in minutes.
        """
        # Check that the input data contains the required lag features
        feature_names = list(x.columns[x.columns.str.startswith("T-")])
        if len(feature_names) == 0:
            raise ValueError("No lag features found in the input data.")

        # Convert all lags to minutes to make comparable
        feature_to_lags_in_min = []
        for feature in feature_names:
            if feature.endswith("min"):
                lag_in_min = int(feature.split("-")[1].split("min")[0])
            elif feature.endswith("d"):
                lag_in_min = int(feature.split("-")[1].split("d")[0]) * 60 * 24
            else:
                raise ValueError(
                    f"Feature name '{feature}' does not follow the expected format."
                    " Expected format is 'T-<lag_in_minutes>' or 'T-<lag_in_days>d'."
                )
            feature_to_lags_in_min.append((feature, lag_in_min))

        # Sort the features by lag in minutes
        feature_to_lags_in_min.sort(key=lambda x: x[1])
        sorted_features, sorted_lags_in_min = zip(*feature_to_lags_in_min)

        # Check that the lags are evenly spaced
        diffs = np.diff(sorted_lags_in_min)
        unique_diffs = np.unique(diffs)
        if len(unique_diffs) > 1:
            raise ValueError(
                "Lag features are not evenly spaced. "
                f"Got lags with differences: {unique_diffs} min. "
                "Please ensure that the lag features are generated correctly."
            )
        frequency = unique_diffs[0]

        return sorted_features, frequency, feature_to_lags_in_min

    @staticmethod
    def _fill_diagonal_with_median(
        lag_array: np.ndarray, start: int, end: int, median: float
    ) -> np.ndarray | None:
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

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """
        Predict the median of the lag features for each time step in the context window.

        Args:
            x (pd.DataFrame): The input data for prediction. This should be a pandas dataframe with lag features.

        Returns:
            np.array: The predicted median for each time step in the context window.
            If any lag feature is NaN, this will be ignored.
            If all lag features are NaN, the regressor will return NaN.
        """

        input_data: pd.DataFrame = data.input_data(start=data.forecast_start)

        if not self._frequency_matches(input_data.index):
            raise ValueError(
                f"The input data frequency ({input_data.index.freq}) does not match the model frequency ({self.frequency})."
            )

        # Check that the input data contains the required lag features
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            raise ValueError(
                f"The input data is missing the following lag features: {missing_features}"
            )

        # Reindex the input data to ensure there are no gaps in the time series.
        # This is important for the autoregressive logic that follows.
        # Store the original index to return predictions aligned with the input.
        original_index = input_data.index.copy()
        first_index = input_data.index[0]
        last_index = input_data.index[-1]
        freq = pd.Timedelta(minutes=self.frequency)
        # Create a new date range with the expected frequency.
        new_index = pd.date_range(first_index, last_index, freq=freq)
        # Reindex the input DataFrame, filling any new timestamps with NaN.
        input_data = input_data.reindex(new_index, fill_value=np.nan)

        # Select only the lag feature columns in the specified order.
        lag_df = input_data[self.feature_names]

        # Convert the lag DataFrame and its index to NumPy arrays for faster processing.
        lag_array = lag_df.to_numpy()
        time_index = lag_df.index.to_numpy()
        # Initialize the prediction array with NaNs.
        prediction = np.full(lag_array.shape[0], np.nan)

        # Calculate the time step size based on the model frequency.
        step_size = pd.Timedelta(minutes=self.frequency)
        # Determine the number of steps corresponding to the smallest and largest lags.
        smallest_lag_steps = int(
            self.lags_to_time_deltas_[self.feature_names[0]] / step_size
        )
        largest_lag_steps = int(
            self.lags_to_time_deltas_[self.feature_names[-1]] / step_size
        )

        # Iterate through each time step in the reindexed data.
        for time_step in range(lag_array.shape[0]):
            # Get the lag features for the current time step.
            current_lags = lag_array[time_step]
            # Calculate the median of the available lag features, ignoring NaNs.
            median = np.nanmedian(current_lags)
            # Store the calculated median in the prediction array.
            prediction[time_step] = median

            # If the median calculation resulted in NaN (e.g., all lags were NaN), skip the autoregression step.
            if np.isnan(median):
                continue

            # Auto-regressive step: update the lag array for future time steps.
            # Calculate the start and end indices in the future time steps that will be affected.
            start, end = (
                time_step + smallest_lag_steps,
                time_step + largest_lag_steps + 1,
            )
            self._fill_diagonal_with_median(lag_array, start, end, median)

        # Convert the prediction array back to a pandas DataFrame using the reindexed time index.
        prediction_df = pd.DataFrame(prediction, index=time_index, columns=["median"])
        # Select only the predictions corresponding to the original input index.
        prediction = prediction_df.loc[original_index].to_numpy().flatten()

        # Return the final predictions as a ForecastDataset.
        predictions = pd.DataFrame(
            data=prediction,
            index=input_data.index,
            columns=[quantile.format() for quantile in self.config.quantiles],
        )

        return ForecastDataset(
            data=predictions,
            sample_interval=data.sample_interval,
        )

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
                """This model does not have any hyperparameters to fit,
        but it does need to know the feature names of the lag features and the order of these.

        Lag features are expected to be evently spaced and match the frequency of the input data.
        The lag features are expected to be named in the format T-<lag_in_minutes> or T-<lag_in_days>d.
        For example, T-1min, T-2min, T-3min or T-1d, T-2d.

        Which lag features are used is determined by the feature engineering step.
        """                
        input_data_predictors = data.input_data()
    
        target_series = data.target_series
    
        (
            feature_names,
            frequency,
            feature_to_lags_in_min,
        ) = self._extract_and_validate_lags(input_data_predictors)

        self.feature_names_ = list(feature_names)
        self.frequency_ = frequency
        self.lags_to_time_deltas_ = {
            key: pd.Timedelta(minutes=val) for key, val in feature_to_lags_in_min
        }

        # Check that the frequency of the input data matches frequency of the lags
        if not self._frequency_matches(
            input_data_predictors.index.drop_duplicates()
        ):  # Several training horizons give duplicates
            raise ValueError(
                f"The input data frequency ({input_data_predictors.index.freq}) does not match the model frequency ({self.frequency})."
            )

        self.feature_importances = np.ones(len(self.feature_names)) / (
            len(self.feature_names) or 1.0
        )
