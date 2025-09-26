# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Median forecasting model with autoregressive capabilities.

Provides an autoregressive forecasting model that uses the median of lag features
to predict future values. This model is particularly suitable for signals with
slow dynamics or those that switch between stable states with noise.

The model uses historical lag features (T-1min, T-2min, etc.) to compute median
values, with autoregressive updating for future predictions where the model's
own predictions become input features for subsequent time steps.
"""

from typing import Self, override

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.utils.validation import check_is_fitted

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ModelLoadingError, NotFittedError
from openstef_core.mixins import State
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.forecaster import HorizonForecaster, HorizonForecasterConfig


class MedianForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for median forecaster.
    
    Currently no hyperparameters are needed as the model computes
    the median directly from lag features without training.
    """

    pass


class MedianForecasterConfig(HorizonForecasterConfig):
    """Configuration for median forecaster.
    
    The median forecaster only supports quantile 0.5 (median) prediction
    and is designed for single-horizon forecasting scenarios.
    """

    quantiles: list[Quantile] = Field(
        default=[Quantile(0.5)],
        description="Only median (0.5) quantile is supported by this forecaster.",
    )

    horizons: list[LeadTime] = Field(
        default=[LeadTime(pd.Timedelta(hours=1))],
        max_length=1,
        description="Single horizon for prediction. Must contain exactly one lead time.",
    )

    hyperparams: MedianForecasterHyperParams = Field(
        default=MedianForecasterHyperParams(),
    )


MODEL_CODE_VERSION = 1


class MedianForecaster(HorizonForecaster):
    """Median-based autoregressive forecaster for single horizon predictions.

    This forecaster computes the median of lag features (T-1min, T-2min, etc.)
    to predict future values. It includes autoregressive capabilities where
    the model's own predictions are used as lag features for subsequent predictions.

    The model is particularly effective for:
    - Signals with very slow dynamics compared to the sampling rate, possibly with noise
    - Signals that switch between stable states but tend to remain stable in each state
    - Scenarios where waste heat or similar processes provide predictable baseline values

    Key characteristics:
    - Only supports median (quantile 0.5) predictions
    - Autoregressive: uses previous predictions as inputs for future predictions
    - No training required: directly computes median from available lag features
    - Handles missing data gracefully by ignoring NaN values in median calculation

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> config = MedianForecasterConfig(
        ...     quantiles=[Quantile(0.5)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ... )
        >>> forecaster = MedianForecaster(config)
        >>> # forecaster.fit(training_data)  # No actual training needed
        >>> # predictions = forecaster.predict(test_data)
    """

    _config: MedianForecasterConfig
    _feature_names: list[str]
    _frequency_minutes: int
    _lags_to_time_deltas: dict[str, pd.Timedelta]

    def __init__(
        self,
        config: MedianForecasterConfig | None = None,
    ) -> None:
        """Initialize the median forecaster.

        Args:
            config: Configuration specifying quantiles and hyperparameters.
        """
        self._config = config or MedianForecasterConfig()
        self._feature_names: list[str] = []
        self._frequency_minutes: int = 0
        self._lags_to_time_deltas: dict[str, pd.Timedelta] = {}

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
            "feature_names": self._feature_names,
            "frequency_minutes": self._frequency_minutes,
            "lags_to_time_deltas": {k: v.total_seconds() for k, v in self._lags_to_time_deltas.items()},
        }

    @override
    def from_state(self, state: State) -> Self:
        if not isinstance(state, dict) or "version" not in state or state["version"] > MODEL_CODE_VERSION:
            raise ModelLoadingError("Invalid state for MedianForecaster")

        instance = self.__class__(config=MedianForecasterConfig.model_validate(state["config"]))
        instance._feature_names = state["feature_names"]  # noqa: SLF001
        instance._frequency_minutes = state["frequency_minutes"]  # noqa: SLF001
        instance._lags_to_time_deltas = {  # noqa: SLF001
            k: pd.Timedelta(seconds=v) for k, v in state["lags_to_time_deltas"].items()
        }
        return instance

    @property
    @override
    def is_fitted(self) -> bool:
        return len(self._feature_names) > 0

    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> pd.Timedelta:
        """Infer the frequency of a pandas DatetimeIndex if the freq attribute is not set.
        
        This method calculates the most common time difference between consecutive timestamps,
        which is more permissive of missing chunks of data than the pandas infer_freq method.

        Args:
            index: The datetime index to infer the frequency from.

        Returns:
            The inferred frequency as a pandas Timedelta.
        """
        if len(index) < 2:
            raise ValueError("Cannot infer frequency from an index with fewer than 2 timestamps.")

        # Calculate the differences between consecutive timestamps
        deltas = index.to_series().diff().dropna()

        # Find the most common difference
        inferred_freq = deltas.mode().iloc[0]
        return inferred_freq

    def _frequency_matches(self, index: pd.DatetimeIndex) -> bool:
        """Check if the frequency of the input data matches the model frequency.

        Args:
            index: The datetime index to check.

        Returns:
            True if the frequencies match, False otherwise.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("The index of the input data must be a pandas DatetimeIndex.")

        if index.freq is None:
            input_frequency = self._infer_frequency(index)
        else:
            input_frequency = index.freq

        return input_frequency == pd.Timedelta(minutes=self._frequency_minutes)

    @staticmethod
    def _extract_and_validate_lags(
        data: pd.DataFrame,
    ) -> tuple[list[str], int, dict[str, int]]:
        """Extract and validate the lag features from the input data.

        This method checks that the lag features are evenly spaced and match the frequency 
        of the input data. It also extracts the lag features and their corresponding time deltas.
        
        Args:
            data: The input data containing lag features.
            
        Returns:
            A tuple containing:
                - A list of feature names, sorted by their lag in minutes.
                - The frequency of the lag features in minutes.
                - A dict mapping feature names to their corresponding time deltas in minutes.
        """
        # Check that the input data contains the required lag features
        feature_names = [col for col in data.columns if col.startswith("T-")]
        if len(feature_names) == 0:
            raise ValueError("No lag features found in the input data.")

        # Convert all lags to minutes to make comparable
        feature_to_lags_in_min = {}
        for feature in feature_names:
            try:
                if feature.endswith("min"):
                    lag_str = feature.split("-")[1].split("min")[0]
                    lag_in_min = int(lag_str)
                elif feature.endswith("d"):
                    lag_str = feature.split("-")[1].split("d")[0]
                    lag_in_min = int(lag_str) * 60 * 24
                else:
                    raise ValueError(
                        f"Feature name '{feature}' does not follow the expected format. "
                        "Expected format is 'T-<lag_in_minutes>' or 'T-<lag_in_days>d'."
                    )
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Feature name '{feature}' does not follow the expected format. "
                    "Expected format is 'T-<lag_in_minutes>' or 'T-<lag_in_days>d'."
                ) from e
            feature_to_lags_in_min[feature] = lag_in_min

        # Sort the features by lag in minutes
        sorted_features = sorted(feature_names, key=lambda x: feature_to_lags_in_min[x])
        sorted_lags_in_min = [feature_to_lags_in_min[f] for f in sorted_features]

        # Check that the lags are evenly spaced
        if len(sorted_lags_in_min) > 1:
            diffs = np.diff(sorted_lags_in_min)
            unique_diffs = np.unique(diffs)
            if len(unique_diffs) > 1:
                raise ValueError(
                    "Lag features are not evenly spaced. "
                    f"Got lags with differences: {unique_diffs} min. "
                    "Please ensure that the lag features are generated correctly."
                )
            frequency = unique_diffs[0]
        else:
            # If only one lag feature, assume frequency is the lag itself
            frequency = sorted_lags_in_min[0]

        return sorted_features, frequency, feature_to_lags_in_min

    @staticmethod
    def _fill_diagonal_with_median(
        lag_array: np.ndarray, start: int, end: int, median: float
    ) -> None:
        """Fill future lag values where this prediction would be used as input.

        Args:
            lag_array: Array of lag features with time steps as rows and lag features as columns.
            start: Starting index in the future time steps.
            end: Ending index in the future time steps.
            median: The median value to fill.
        """
        # If the start index is beyond the array bounds, no future updates are needed from this step.
        if start >= lag_array.shape[0]:
            return

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
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the median forecaster by extracting and validating lag features.

        This model does not have any hyperparameters to fit, but it does need to know
        the feature names of the lag features and their frequency. Lag features are expected
        to be evenly spaced and match the frequency of the input data.

        Args:
            data: Training dataset containing lag features.
            data_val: Validation dataset (unused by this model).
        """
        # Extract lag features from the input data
        (
            feature_names,
            frequency,
            feature_to_lags_in_min,
        ) = self._extract_and_validate_lags(data.data)

        self._feature_names = feature_names
        self._frequency_minutes = frequency
        self._lags_to_time_deltas = {
            key: pd.Timedelta(minutes=val) for key, val in feature_to_lags_in_min.items()
        }

        # Check that the frequency of the input data matches frequency of the lags
        if not self._frequency_matches(data.index.drop_duplicates()):
            raise ValueError(
                f"The input data frequency does not match the model frequency ({self._frequency_minutes} min)."
            )

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Predict the median of the lag features for each time step.

        Args:
            data: Input dataset for prediction containing lag features.

        Returns:
            Forecast dataset with median predictions.

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If input data frequency doesn't match model frequency or missing features.
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Check frequency compatibility
        if not self._frequency_matches(data.index):
            raise ValueError(
                f"The input data frequency does not match the model frequency ({self._frequency_minutes} min)."
            )

        # Check that the input data contains the required lag features
        missing_features = set(self._feature_names) - set(data.data.columns)
        if missing_features:
            raise ValueError(
                f"The input data is missing the following lag features: {missing_features}"
            )

        # Reindex the input data to ensure there are no gaps in the time series.
        # This is important for the autoregressive logic that follows.
        # Store the original index to return predictions aligned with the input.
        original_index = data.index.copy()
        first_index = data.index[0]
        last_index = data.index[-1]
        freq = pd.Timedelta(minutes=self._frequency_minutes)
        
        # Create a new date range with the expected frequency.
        new_index = pd.date_range(first_index, last_index, freq=freq)
        
        # Reindex the input DataFrame, filling any new timestamps with NaN.
        reindexed_data = data.data.reindex(new_index, fill_value=np.nan)

        # Select only the lag feature columns in the specified order.
        lag_df = reindexed_data[self._feature_names]

        # Convert the lag DataFrame and its index to NumPy arrays for faster processing.
        lag_array = lag_df.to_numpy()
        time_index = lag_df.index.to_numpy()
        
        # Initialize the prediction array with NaNs.
        prediction = np.full(lag_array.shape[0], np.nan)

        # Calculate the time step size based on the model frequency.
        step_size = pd.Timedelta(minutes=self._frequency_minutes)
        
        # Determine the number of steps corresponding to the smallest and largest lags.
        smallest_lag_steps = int(self._lags_to_time_deltas[self._feature_names[0]] / step_size)
        largest_lag_steps = int(self._lags_to_time_deltas[self._feature_names[-1]] / step_size)

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
        final_predictions = prediction_df.loc[original_index].to_numpy().flatten()

        # Create result DataFrame with quantile column names according to the config
        result_data = pd.DataFrame(
            data={quantile.format(): final_predictions for quantile in self.config.quantiles},
            index=original_index,
        )

        # Create forecast dataset, ensuring we only predict after forecast_start if specified
        if data.forecast_start is not None:
            result_data = result_data[result_data.index > pd.Timestamp(data.forecast_start)]

        return ForecastDataset(
            data=result_data,
            sample_interval=data.sample_interval,
        )