import os

import numpy as np
import pandas as pd
import secrets

import random

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
from dateutil.parser import parse


def idx_start_end_date(data, amount_day):
    """
    Retrieve the index of the last cell of the first day, and the first cell of the last day.
    This is due to different length of period_timedelta for those two days.

    Args:
        data (pandas.DataFrame): Clean data with features
        amount_day (int):  Duration of the periods (in T-15) that are in a day (default = 96)
    Returns:
        end_first (int): Index of the last cell of first day
        begin_last (int) : Index of the first cell of the last day

    """

    start_date_idx = parse(data.index[0].strftime("%Y-%m-%d")).day
    end_first = 0
    begin_last = 96
    for i in range(0, amount_day):
        day_i = parse(data.index[i].strftime("%Y-%m-%d")).day
        if day_i != start_date_idx:
            end_first = i
            break

    end_date_idx = parse(data.index[len(data.index) - 1].strftime("%Y-%m-%d")).day
    for i in range(len(data.index), -1, -1):
        day_i = parse(data.index[i - 1].strftime("%Y-%m-%d")).day
        if day_i != end_date_idx:
            begin_last = len(data.index) - i + 2
            break

    return end_first, begin_last


def fill_days(data, end_first, begin_last, amount_day, period_timedelta):
    """
    Fill an array with the respective number for the day (increasing)

    Args:
        data (pandas.DataFrame): Clean data with features
        end_first (int): Index of the last cell of first day
        begin_last (int) : Index of the first cell of the last day
        amount_day (int): Duration of the periods (in T-15) that are in a day (default = 96)
        period_timedelta (int): Duration of the periods (in days) that will
            be sampled as validation data for each split.

    Returns:
        days_np (np.array): Array with increasing number of the days

    """

    number_i = 0
    days_np = np.array(end_first * [number_i])
    number_i += 1

    days_idx = np.arange(
        end_first + 1,
        len(data["load"]) - (amount_day / period_timedelta),
        (amount_day / period_timedelta),
    )

    for i in days_idx:
        days_np = np.append(days_np, int(amount_day / period_timedelta) * [number_i])
        number_i += 1

    days_np = np.append(days_np, begin_last * [number_i])

    return days_np


def min_max_peaks(data, start, end, percentiles):
    """
    Retrieve minimum and maximum quantiles for a day (start till end)

    Args:
        data (pandas.DataFrame): Clean data with features
        start (int): Index of the start of the day
        end (int) : Index of the end of the day
        percentiles (list:int): List with the lower and upper quantiles

    Returns:
        min_peaks (float): Minimum peak of a day
        max_peaks (float): Maximum peak of a day

    """
    min_percentile = percentiles[0]
    max_percentile = percentiles[1]

    try:
        min_peaks = min(
            data["load"][start:end][
                data["load"] < data["load"].quantile(min_percentile)
                ]
        )
    except:
        min_peaks = np.nan

    try:
        max_peaks = max(
            data["load"][start:end][
                data["load"] > data["load"].quantile(max_percentile)
                ]
        )
    except:
        max_peaks = np.nan

    return min_peaks, max_peaks


def min_max_fill_days(
        data, end_first, begin_last, percentiles, amount_day, period_timedelta
):
    """
    Fill arrays with the minimum and maximum quantiles for each day

    Args:
        data (pandas.DataFrame): Clean data with features
        end_first (int): Index of the last cell of first day
        begin_last (int) : Index of the first cell of the last day
        percentiles (list:int): List with the lower and upper quantiles
        amount_day (int): Duration of the periods (in T-15) that are in a day (default = 96)
        period_timedelta (int): Duration of the periods (in days) that will
            be sampled as validation data for each split.

    Returns:
        fill_min (np.array): Array filled with the minimum quantile for each day
        fill_max (np.array): Array filled with the maximum quantile for each day

    """

    min_i, max_i = min_max_peaks(data, 0, end_first, percentiles)

    fill_min = np.array(end_first * [min_i])
    fill_max = np.array(end_first * [max_i])

    start = end_first + 1
    end = start + int(amount_day / period_timedelta)

    days_idx = np.arange(
        start,
        len(data["load"]) - (amount_day / period_timedelta),
        (amount_day / period_timedelta),
    )

    for i in days_idx:
        min_i, max_i = min_max_peaks(data, start + 1, end, percentiles)

        fill_min = np.append(fill_min, int(amount_day / period_timedelta) * [min_i])
        fill_max = np.append(fill_max, int(amount_day / period_timedelta) * [max_i])

        start += int(amount_day / period_timedelta)
        end += int(amount_day / period_timedelta)

    min_i, max_i = min_max_peaks(data, start + 1, len(data["load"]), percentiles)

    fill_min = np.append(fill_min, begin_last * [min_i])
    fill_max = np.append(fill_max, begin_last * [max_i])

    return fill_min, fill_max


def peak_present(data):
    """
    Checks if there is a peak present, there is at least one peak (minimum or maximum)

    Args:
        data (pandas.DataFrame): Clean data with features

    Returns:
        boolean: Value for the presence of a peak

    """

    if (data["min_peak"] == True) or (data["max_peak"] == True):
        return True
    else:
        return False


def sample_indices_train_val(data, peaks, period_lengths, end_first, begin_last):
    """
    Sample indices of given period length assuming the peaks are evenly spreaded.

    Args:
        data (pandas.DataFrame): Clean data with features
        peaks (list:int): List of selected peaks to sample the indices from
        period_lengths (int): Duration of the periods (default = 96)
        end_first (int): Index of the last cell of first day
        begin_last (int) : Index of the first cell of the last day

    Returns:
        np.array: Sorted list with the indices corresponding to the peak

    """

    sampled = set()
    for peak in peaks:
        if peak < (len(data)):
            start_point = int(peak * period_lengths) + end_first
            if start_point < (len(data.index) - begin_last + 1):
                end_point = int(start_point + period_lengths)
            else:
                end_point = int(start_point + end_first)
            sampled |= set(range(start_point, end_point))
    return np.sort(list(sampled))


def random_sample(list_sample: list, k: int) -> list:
    """
    Random sampling of numbers out of a list

    Args:
        list_sample (list:int): List with numbers to sample from
        k (int): Number of wanted samples

    Returns:
        list: Sorted list with the random samples

    """

    random_list = []
    for i in range(k):
        element_random = secrets.choice(list_sample)
        list_sample.remove(element_random)
        random_list.append(element_random)
    return random_list


def split_data_train_validation_test(
        data_,
        test_fraction=0.1,
        validation_fraction=0.15,
        back_test=False,
        stratification=True,
        period_timedelta=1,
):
    """
    Split input data into train, test and validation set.

    Function for splitting data with features in a train, test and
    validation dataset. In an operational setting the folowing sequence is
    returned (when using stratification):

    Test >> Train >> Validation

    For a back test (indicated with argument "back_test") the folowing sequence
    is returned:

    Train >> Validation >> Test

    The ratios of the different types can be set with test_fraction and
    validation fraction.

    Args:
        data_ (pandas.DataFrame): Cleaned data with features
        test_fraction (float): Number between 0 and 1 that indicates the desired
            fraction of test data.
        validation_fraction (float): Number between 0 and 1 that indicates the
            desired fraction of validation data.
        back_test (bool): Indicates if data is intended for a back test.
        stratification (bool): Indicates if validation data must be sampled as
            periods, using stratification
        period_timedelta (int): Duration of the periods (in days) that will
            be sampled as validation data for each split.

    Returns:
        data (pandas.DataFrame): Clean data with features
        train_data (pandas.DataFrame): Train data.
        validation_data (pandas.DataFrame): Validation data.
        test_data (pandas.DataFrame): Test data.

    """

    amount_day = 96

    # Check input
    data_["timestamp"] = data_.index
    if "Horizon" in data_.columns:
        data = data_[data_["Horizon"] == 47]
    else:
        data = data_

    train_fraction = 1 - (test_fraction + validation_fraction)
    train_val_fraction = train_fraction + validation_fraction
    if train_fraction < 0:
        raise ValueError(
            "Test ({test_fraction}) and validation fraction ({validation_fraction}) too high."
        )

    idx_first_last, idx_begin_end = idx_start_end_date(data, amount_day)

    # Quantiles in order to identify the peaks
    max_quantile = 0.95
    min_quantile = 0.05

    # Get start date from the index
    start_date = data.index.min().to_pydatetime()

    # Calculate total of quarter hours (PTU's) in input data
    number_indices = len(data.index.unique())  # Total number of unique timepoints
    delta = (
            data.index.unique().sort_values()[1] - data.index.unique().sort_values()[0]
    )  # Delta t, assumed to be constant troughout DataFrame
    delta = timedelta(
        seconds=delta.seconds
    )  # Convert from pandas timedelta to original python timedelta

    # Identify the peaks and list them
    peaks_min, peaks_max = min_max_fill_days(
        data,
        idx_first_last,
        idx_begin_end,
        [min_quantile, max_quantile],
        amount_day,
        period_timedelta,
    )
    min_nan = np.where(np.isnan(peaks_min), False, True)
    max_nan = np.where(np.isnan(peaks_max), False, True)

    data.loc[:, "min_peak"] = min_nan[: (len(data))]
    data.loc[:, "max_peak"] = max_nan[: (len(data))]
    data.loc[:, "days"] = fill_days(
        data, idx_first_last, idx_begin_end, amount_day, period_timedelta
    )[: (len(data))]
    data.loc[:, "peaks_day"] = data.apply(peak_present, axis=1)

    peak_n_days = len(np.unique(data[data["peaks_day"] == True]["days"]))
    peak_all_days = list(np.unique(data[data["peaks_day"] == True]["days"]))

    if len(peak_all_days) < 3:
        stratification = False

    # Default sampling, take a single validation set.
    if not stratification:
        if back_test:
            start_date_val = start_date
            start_date_train = (
                    start_date_val + np.round(number_indices * validation_fraction) * delta
            )
            start_date_test = (
                    start_date_train
                    + np.round(number_indices * (1 - validation_fraction - test_fraction))
                    * delta
            )
            train_data = data[start_date_train:start_date_test]
            test_data = data[start_date_test:None]
        else:
            start_date_test = start_date
            start_date_val = (
                    start_date + np.round(number_indices * test_fraction) * delta
            )
            start_date_train = (
                    start_date_val + np.round(number_indices * validation_fraction) * delta
            )
            test_data = data[start_date_test:start_date_val]
            train_data = data[start_date_train:None]

        validation_data = data[start_date_val:start_date_train]

    # Sample periods in the training part as the validation set using stratification (peaks).
    else:
        test_amount = int(test_fraction * (peak_n_days))
        train_val_amount = int(train_val_fraction * (peak_n_days))
        split_val = int((peak_n_days * validation_fraction) / period_timedelta)

        if back_test:
            # Train + Val >> Test

            start_date_train_val = start_date

            start_idx_test = int(random_sample(peak_all_days[train_val_amount: train_val_amount + 1], k=1)[0])
            start_date_test = (
                    start_date_train_val
                    + start_idx_test * (amount_day / period_timedelta) * delta
            )

            test_data = data[start_date_test:None]

            idx_val_split = sample_indices_train_val(
                data, random_sample(peak_all_days[:train_val_amount], k=split_val + 1),
                (amount_day / period_timedelta),
                idx_first_last,
                idx_begin_end,
            )

            validation_data = data.loc[data.index.unique()[idx_val_split]]
            train_data = data[~data.index.isin(validation_data.index)]
            train_data = train_data[~train_data.index.isin(test_data.index)]
        else:
            # Test >> Train + Val

            start_date_test = start_date

            start_idx_train_val = int(
                random_sample(peak_all_days[test_amount: test_amount + 1], k=1)[0]
            )
            start_date_train_val = (
                    start_date_test
                    + start_idx_train_val * (amount_day / period_timedelta) * delta
            )

            test_data = data[start_date_test:start_date_train_val]

            idx_val_split = sample_indices_train_val(
                data,
                random_sample(peak_all_days[start_idx_train_val:], k=split_val + 1),
                (amount_day / period_timedelta),
                idx_first_last,
                idx_begin_end,
            )

            validation_data = data.loc[data.index.unique()[idx_val_split]]
            train_data = data[~data.index.isin(validation_data.index)]
            train_data = train_data[~train_data.index.isin(test_data.index)]

    train_data = train_data.sort_values(by="timestamp")
    validation_data = validation_data.sort_values(by="timestamp")
    test_data = test_data.sort_values(by="timestamp")

    return (
        data,
        train_data.iloc[:, :-5],
        validation_data.iloc[:, :-5],
        test_data.iloc[:, :-5],
    )
