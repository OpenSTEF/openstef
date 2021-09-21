# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
import secrets

from datetime import datetime, timedelta
from dateutil.parser import parse

from typing import List

AMOUNT_DAY = 96  # Duration of the periods (in T-15) that are in a day (default = 96)
PERIOD_TIMEDELTA = 1  # Duration of the periods (in days) that will be sampled as validation data for each split.
MAX_QUANTILE = 0.95
MIN_QUANTILE = 0.05


def find_min_max_peaks(data: pd.DataFrame) -> (List[float], List[float]):
    """
    Retrieve minimum and maximum quantiles for a day (start till end)

    Args:
        data (pandas.DataFrame): Clean data with features

    Returns:
        min_peaks (list:float): Minimum peaks of a day
        max_peaks (list:float): Maximum peaks of a day

    """

    try:
        min_peaks = (
            data["days"][:][data["load"] < data["load"].quantile(MIN_QUANTILE)]
        )
    except ValueError:
        min_peaks = np.nan

    try:
        max_peaks = (
            data["days"][:][data["load"] > data["load"].quantile(MAX_QUANTILE)]
        )
    except ValueError:
        max_peaks = np.nan

    return min_peaks, max_peaks


def find_corresponding_min_max_days(
    data: pd.DataFrame) -> np.array:
    """
    Checks if there is a peak present, there is at least one peak (minimum or maximum)

    Args:
        data (pandas.DataFrame): Clean data with features
    Returns:
        present_peaks (np.array): Array filled with the days with peaks

    """

    min_peaks = find_min_max_peaks(data)[0].to_numpy()
    max_peaks = find_min_max_peaks(data)[1].to_numpy()

    present_peaks = np.unique(np.append(min_peaks,max_peaks))

    return present_peaks


def sample_indices_train_val(
    data: pd.DataFrame, peaks: List[int]
) -> np.array:
    """
    Sample indices of given period length assuming the peaks are evenly spreaded.

    Args:
        data (pandas.DataFrame): Clean data with features
        peaks (list:int): List of selected peaks to sample the indices from

    Returns:
        np.array: Sorted list with the indices corresponding to the peak

    """

    sampled = set()

    for peak in peaks:
        sampled |= set(data[data["days"] == peak].idx_num.to_numpy())
    return np.sort(list(sampled))


def random_sample(list_sample: List[int], k: int) -> List[int]:
    """
    Random sampling of numbers out of a list
    (implemented due to security sonar cloud not accepting the random built-in functions)

    Args:
        list_sample (list:int): List with numbers to sample from
        k (int): Number of wanted samples

    Returns:
        (list:int): Sorted list with the random samples

    """

    random_list = []
    for _ in range(k):
        element_random = secrets.choice(list_sample)
        list_sample.remove(element_random)
        random_list.append(element_random)
    return random_list


def split_data_train_validation_test(
    data_: pd.DataFrame,
    test_fraction: float = 0.1,
    validation_fraction: float = 0.15,
    back_test: bool = False,
    stratification: bool = True,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split input data into train, test and validation set.

    Function for splitting data with features in a train, test and
    validation dataset. In an operational setting the following sequence is
    returned (when using stratification):

    Test >> Train >> Validation

    For a back test (indicated with argument "back_test") the following sequence
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

    Returns:
        train_data (pandas.DataFrame): Train data.
        validation_data (pandas.DataFrame): Validation data.
        test_data (pandas.DataFrame): Test data.

    """

    # Check input
    data_["timestamp"] = data_.index
    if "Horizon" in data_.columns:
        data = data_[data_["Horizon"] == 47]
    else:
        data = data_
        stratification = False

    data["idx_num"] = range(0, len(data))

    train_fraction = 1 - (test_fraction + validation_fraction)
    train_val_fraction = train_fraction + validation_fraction
    if train_fraction < 0:
        raise ValueError(
            "Test ({test_fraction}) and validation fraction ({validation_fraction}) too high."
        )

    # Get start date from the index
    start_date = data.index.min().to_pydatetime()

    # Calculate total of quarter hours (PTU's) in input data
    number_indices = len(data.index.unique())  # Total number of unique timepoints
    delta = (
        data.index.unique().sort_values()[1] - data.index.unique().sort_values()[0]
    )  # Delta t, assumed to be constant throughout DataFrame
    delta = timedelta(
        seconds=delta.seconds
    )  # Convert from pandas timedelta to original python timedelta

    # Identify the peaks and list them
    data.loc[:, "days"] = (
        pd.to_datetime(data.index, format="%Y%m%d")
        .strftime("%Y%m%d")
        .astype(int)
        .tolist()
    )
    data.loc[:, "days"] = data["days"].diff().fillna(0).astype(int)[: (len(data))]
    data.loc[:, "days"] = np.where((data.days > 0), 1, data.days).cumsum()

    peak_all_days = find_corresponding_min_max_days(data).tolist()
    peak_n_days = len(peak_all_days)

    if peak_n_days < 3:
        stratification = False

    # Sample periods in the training part as the validation set using stratification (peaks).
    if stratification:
        test_amount = int(test_fraction * peak_n_days)
        train_val_amount = int(train_val_fraction * peak_n_days)
        split_val = int((peak_n_days * validation_fraction) / PERIOD_TIMEDELTA)

        if back_test:
            # Train + Val >> Test

            start_date_train_val = start_date

            start_idx_test = int(
                random_sample(
                    peak_all_days[train_val_amount : train_val_amount + 1], k=1
                )[0]
            )
            start_date_test = (
                start_date_train_val
                + start_idx_test * (AMOUNT_DAY / PERIOD_TIMEDELTA) * delta
            )

            test_data = data[start_date_test:None]

            idx_val_split = sample_indices_train_val(
                data,
                random_sample(peak_all_days[:train_val_amount], k=split_val + 2)
            )

            validation_data = data.loc[data.index.unique()[idx_val_split]]
            train_data = data[~data.index.isin(validation_data.index)]
            train_data = train_data[~train_data.index.isin(test_data.index)]
        else:
            # Test >> Train + Val

            start_date_test = start_date

            start_idx_train_val = int(
                random_sample(peak_all_days[test_amount : test_amount + 1], k=1)[0]
            )
            start_date_train_val = (
                start_date_test
                + start_idx_train_val * (AMOUNT_DAY / PERIOD_TIMEDELTA) * delta
            )

            test_data = data[start_date_test:start_date_train_val]

            idx_val_split = sample_indices_train_val(
                data,
                random_sample(peak_all_days[start_idx_train_val:], k=split_val + 2)
            )

            validation_data = data.loc[data.index[idx_val_split]]
            train_data = data[~data.index.isin(validation_data.index)]
            train_data = train_data[~train_data.index.isin(test_data.index)]

    # Default sampling, take a single validation set.
    else:
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

    train_data = train_data.sort_values(by="timestamp")
    validation_data = validation_data.sort_values(by="timestamp")
    test_data = test_data.sort_values(by="timestamp")

    return (
        train_data.iloc[:, :-3],
        validation_data.iloc[:, :-3],
        test_data.iloc[:, :-3],
    )
