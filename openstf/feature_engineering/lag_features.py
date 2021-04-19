import re

import numpy as np
import pandas as pd
import scipy.signal


def generate_lag_feature_functions(data, feature_set_names=None, horizon=24):
    """Creates functions to generate lag features in a dataset.

    Args:
        data (pd.DataFrame): input data for an xgboost prediction or model training.
        feature_set_names (list of strings): minute lagtimes that where used during training
            of the model. If empty a new set will be automatically generated.
        horizon (int): Forecast horizon limit in hours.


    Returns:
        dict: dictionary with lag functions

    Example:
        lag_functions = generate_lag_functions(data,minute_list,h_ahead)
    """

    # Generate lag_times if no features are provided
    if feature_set_names is None:
        lag_times_minutes, lag_time_days_list = generate_trivial_lag_features(horizon)

    # Or extract lag features if provided
    else:
        lag_times_minutes, lag_time_days_list = extract_lag_features(feature_set_names)

    # Empty dict to store all generated lag functions
    lag_functions = {}

    # Add intraday-lag functions (lags in minutes)
    for minutes in lag_times_minutes:

        def func(x, shift=minutes):
            return x.shift(freq="T", periods=1 * shift)

        new = {"T-" + str(int(minutes)) + "min": func}
        lag_functions.update(new)

    # Add day lag functions:
    for day in lag_time_days_list:

        def func(x, shift=day):
            return x.shift(freq="1d", periods=1 * shift)

        new = {"T-" + str(int(day)) + "d": func}
        lag_functions.update(new)
    return lag_functions


def generate_trivial_lag_features(horizon):
    """Function that generates relevant lag times for lag feature function creation.
    This function is mostly used during training of models and not during predicting
    Args:
        data: (pd.DataFrame): input data for an xgboost prediction or model training.
        horizon: Forecast horizon limit in hours.

    Returns:
        minutes_list (list[int]): list of minute lags that were used as features during training
        days_list (list[int]): list of minute lags that were used as features during training

    """
    mindays = int(np.ceil(horizon / 24))
    lag_time_days_list = list(np.linspace(mindays, 14, 15 - mindays))

    # Make list of trivial lag times
    trivial_lag_minutes_list = np.linspace(60, 24 * 60, 24).tolist() + [15, 30, 45]

    # Discard lag times that are not available for the specified horizon
    trivial_lag_times_minutes = set(
        [i for i in trivial_lag_minutes_list if i >= horizon * 60]
    )

    return trivial_lag_times_minutes, lag_time_days_list


def extract_lag_features(lag_features):
    """Creates a list of lag minutes and a list of lag days that were used during
    the training of the input model

    Args:
        lag_features (list[str]): All requested lag features

    Returns:
        minutes_list (list[int]): list of minute lags that were used as features during training
        days_list (list[int]): list of minute lags that were used as features during training
    """

    # Prepare empty lists to append on
    minutes_list = []
    days_list = []

    for lag_feature in lag_features:

        # Select the number of days or the number of minutes by matching with a regular expression
        number_of_minutes = re.search(r"T-(\d+)min", lag_feature)
        number_of_days = re.search(r"T-(\d+)d", lag_feature)

        # Append to the appropriate list
        if number_of_minutes is not None:
            minutes_list.append(int(number_of_minutes[1]))
        elif number_of_days is not None:
            days_list.append(int(number_of_days[1]))

    return minutes_list, days_list


def generate_non_trivial_lag_times(data, height_treshold=0.1):
    """This script calculates an autocorrelation curve of the load trace. This curve is
        subsequently used to add additional lag times as features.

    Args:
        data (pandas.DataFrame): a pandas dataframe with input data in the form pd.DataFrame(index = datetime,
                             columns = [label, predictor_1,..., predictor_n])
        height_treshold (float): minimal autocorrelation value to be recognized as a peak.

    Returns:
        list of ints with aditional non-trivial minute lags


    """

    def autocorr(x, lags):
        """Function to make an autocorrelation curve"""
        mean = x.mean()
        var = np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, "full")[len(x) - 1 :] / var / len(x)

        return corr[: len(lags)]

    try:
        # Get rid of nans as the autocorrelation handles these values badly
        data = data[data.columns[0]].dropna()  # First column contains the load
        # Get autocorrelation curve
        y = autocorr(data, range(10000))
        # Determine the peaks (positive and negative) larger than a specified threshold
        peaks = scipy.signal.find_peaks(np.abs(y), height=height_treshold)
        peaks = peaks[0]
        # Convert peaks to lag times in minutes
        peaks = peaks[peaks < (60 * 4)]
        additional_minute_space = peaks * 15
    except Exception:
        return []
    # Return list of additional minute lags to be procceses by apply features
    return list(additional_minute_space)
