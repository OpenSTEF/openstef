"""This module contains the median regressor."""

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from openstef.model.regressors.regressor import OpenstfRegressor


class MedianRegressor(OpenstfRegressor, RegressorMixin):
    """
    Median regressor implementing the OpenSTEF regressor API.
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

    def __init__(self):
        """Initialize MedianRegressor."""
        super().__init__()

    @property
    def feature_names(self) -> list:
        """Retrieve the model input feature names.

        Returns:
            The list of feature names

        """
        check_is_fitted(self, "feature_names_")
        return self.feature_names_
    
    @property
    def frequency(self) -> int:
        """Retrieve the model input frequency.

        Returns:
            The frequency of the model input

        """
        check_is_fitted(self, "frequency_")
        return self.frequency_
    

    @property
    def can_predict_quantiles(self) -> bool:
        return False
    
    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> pd.Timedelta:
        """
        Infer the frequency of a pandas DatetimeIndex if the freq attribute is not set.

        Args:
            index (pd.DatetimeIndex): The datetime index to infer the frequency from.

        Returns:
            pd.Timedelta: The inferred frequency as a pandas Timedelta.
        """
        if len(index) < 2:
            raise ValueError("Cannot infer frequency from an index with fewer than 2 timestamps.")
        
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
            raise ValueError("The index of the input data must be a pandas DatetimeIndex.")
        
        if index.freq is None:
            input_frequency = self._infer_frequency(index)
        else:
            input_frequency = index.freq

        return input_frequency == pd.Timedelta(minutes=self.frequency)
    

    def predict(self, x: pd.DataFrame, **kwargs) -> np.array:
        """
        Predict the median of the lag features for each time step in the context window.

        Args:
            x (pd.DataFrame): The input data for prediction. This should be a pandas dataframe with lag features.

        Returns:
            np.array: The predicted median for each time step in the context window.
            If any lag feature is NaN, this will be ignored.
            If all lag features are NaN, the regressor will return NaN.
        """
        if not self._frequency_matches(x.index):
            raise ValueError(
                f"The input data frequency ({x.index.freq}) does not match the model frequency ({self.frequency})."
            )

        # Check that the input data contains the required lag features
        missing_features = set(self.feature_names) - set(x.columns)
        if missing_features:
            raise ValueError(
                f"The input data is missing the following lag features: {missing_features}"
            )
        largest_lag = self.lags_to_time_deltas_[self.feature_names[-1]]
        smallest_lag = self.lags_to_time_deltas_[self.feature_names[0]]

        # Reindex the input data to ensure there are no gaps
        # The returned values need to be the original index
        original_index = x.index.copy()
        first_index = x.index[0]
        last_index = x.index[-1]
        freq = pd.Timedelta(minutes=self.frequency)
        new_index = pd.date_range(first_index, last_index, freq=freq)
        x = x.reindex(new_index, fill_value=np.nan)

        # Put the lag features in the right column order
        lag_df = x[self.feature_names]

        prediction = []
        for index, time_step in enumerate(lag_df.index):
            current_lags = lag_df.iloc[index]
            median = np.nanmedian(current_lags, axis=0)            
            prediction.append(median)

            # Fill the future lags with the median value, if this is not nan
            if median is np.nan:
                continue
            slice_start = time_step + smallest_lag
            slice_end = time_step + largest_lag
            diagonal_matrix = lag_df.loc[slice_start:slice_end].copy().to_numpy()
            np.fill_diagonal(diagonal_matrix, median)
            df_to_fill_from = pd.DataFrame(
                diagonal_matrix,
                index=lag_df.loc[slice_start:slice_end].index,
                columns=lag_df.columns,
            )

            # Fill in any nan values in lag_df with the values from the diagonal matrix
            lag_df.loc[slice_start:slice_end] = lag_df.loc[
                slice_start:slice_end
            ].combine_first(df_to_fill_from)

        prediction_df = pd.DataFrame(
            prediction,
            index=lag_df.index,
            columns=["median"],
        )
        prediction = prediction_df.loc[original_index].to_numpy().flatten()

        return prediction

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> RegressorMixin:
        """This model does not have any hyperparameters to fit,
        but it does need to know the feature names of the lag features and the order of these.
        Lag features are expected to be evently spaced and match the frequency of the input data.
        The lag features are expected to be named in the format T-<lag_in_minutes> or T-<lag_in_days>d.
        For example, T-1min, T-2min, T-3min or T-1d, T-2d.

        Which lag features are used is determined by the feature engineering step.
        """
        feature_names = list(x.columns[x.columns.str.startswith("T-")])
        if len(feature_names) == 0:
            raise ValueError("No lag features found in the input data.")
        
        # Convert all lags to minutes to make comparable
        feature_to_lags_in_min = []
        for feature in feature_names:
            if feature.endswith("min"):
                lag_in_min = int(feature.split("-")[1].split("min")[0])
            elif feature.endswith("d"):
                lag_in_min = int(feature.split("-")[1].split("d")[0]) * 60 * 2
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

        self.feature_names_ = list(sorted_features)
        self.frequency_ = unique_diffs[0]
        self.lags_to_time_deltas_ = {key: pd.Timedelta(minutes=val) for key, val in zip(self.feature_names_, sorted_lags_in_min)}

        # Check that the frequency of the input data matches frequency of the lags
        if not self._frequency_matches(x.index.drop_duplicates()): # Several training horizons give duplicates
            raise ValueError(
                f"The input data frequency ({x.index.freq}) does not match the model frequency ({self.frequency})."
            )

        self.feature_importances_ = np.ones(len(self.feature_names_)) / (
            len(self.feature_names_) or 1.0
        )
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True
