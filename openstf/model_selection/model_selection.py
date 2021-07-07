# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
import structlog


def split_data_train_validation_test(
    data: pd.DataFrame,
    test_fraction: float = 0.0,
    validation_fraction: float = 0.15,
    backtest: bool = False,
    period_sampling: bool = True,
    period_timedelta: timedelta = timedelta(days=2),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split input data into train, test and validation set.

    Function for splitting cleaned data with features in a train, test and
    validation dataset. In an operational setting the folowing sequence is
    returned:

    Test >> Validation >> Train

    For a back test (indicated with argument "back_test") the folowing sequence
    is returned:

    Validation >> Train >> Test

    The ratios of the different types can be set with test_fraction and
    validation fraction.

    Args:
        data (pandas.DataFrame): Clean data with features
        test_fraction (float): Number between 0 and 1 that indicates the desired
            fraction of test data.
        validation_fraction (float): Number between 0 and 1 that indicates the
            desired fraction of validation data.
        backtest (bool): Indicates if data is intended for a back test.
        period_sampling (bool): Indicates if validation data must be sampled as
            periods.
        period_timedelta (datetime.timedelta): Duration of the periods that will
            be sampled as validation data. Only used for period_sampling=True.

    Returns:
        Tuple with train data, validation data and test data:
            [0] (pandas.DataFrame): Train data
            [1] (pandas.DataFrame): Validation data
            [2] (pandas.DataFrame): Test data
    """
    MIN_TRAIN_FRACTION = 0.5
    logger = structlog.get_logger(__name__)

    # Check input
    train_fraction = 1 - (test_fraction + validation_fraction)

    if train_fraction < 0:
        raise ValueError(
            "Test ({test_fraction}) and validation fraction ({validation_fraction}) too high."
        )

    if train_fraction < MIN_TRAIN_FRACTION:
        # TODO no action if above threshold? Which settings are meant here?
        logger.warning("Current settings only allow for 50% train data")

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

    # Default sampling, take a single validation set.
    if not period_sampling:
        # Define start and end datetimes of test, train, val sets based on input
        if backtest:
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

        # In either case validation data is before the training data
        validation_data = data[start_date_val:start_date_train]

    # Sample periods in the training part as the validation set.
    else:
        if backtest:
            start_date_combined = start_date
            start_date_test = (
                start_date_combined
                + np.round(number_indices * (1 - test_fraction)) * delta
            )

            combined_data = data[start_date_combined:start_date_test]
            test_data = data[start_date_test:None]
        else:
            start_date_test = start_date
            start_date_combined = (
                start_date + np.round(number_indices * test_fraction) * delta
            )

            combined_data = data[start_date_combined:]
            test_data = data[start_date_test:start_date_combined]

        train_data, validation_data = sample_validation_data_periods(
            combined_data,
            validation_fraction=validation_fraction / (1 - test_fraction),
            period_length=int(period_timedelta / delta),
        )

    # Return datasets
    return train_data, validation_data, test_data


def sample_validation_data_periods(
    data: pd.DataFrame, validation_fraction: float = 0.15, period_length: int = 192
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data in train and validation dataset.

    Tries to sample random periods of given length to form a validation set of
    the desired size. Will raise an error if the number of attempts exceeds the
    maximum given to this function (default: 10).

    Args:
        data (pandas.DataFrame): Clean data with features
        validation_fraction (float): Number between 0 and 1 that indicates the
            desired fraction of validation data. Using a value larger than ~0.4 might
            lead to this function failing.
        period_length (int): Desired size of the sampled periods. The actual
            values can be slightly different if this is required to create the
            right fraction of validation data. Each period will have a duration
            of at least half period_length and at most one and a half
            period_length.

    Returns:
        train_data (pandas.DataFrame): Train data.
        validation_data (pandas.DataFrame): Validation data.
    """

    data_size = len(data.index.unique())
    validation_size = np.round(data_size * validation_fraction).astype(int)
    number_periods = np.round(validation_size / period_length).astype(int)

    # Always atleast one validation period
    if number_periods < 1:
        number_periods = 1

    period_lengths = [period_length] * (number_periods - 1)
    period_lengths += [validation_size - sum(period_lengths)]

    # Default buffer is equal to period_length
    buffer_length = period_length

    # Check if the dataset has enough points for the current settings
    if validation_size + 2 * number_periods * buffer_length >= data_size:
        # Use half period_length otherwise
        buffer_length = np.round(buffer_length / 2).astype(int)

    # Sample indices as validation data
    try:
        validation_indices = _sample_indices(
            data_size - max(period_lengths), period_lengths, buffer_length
        )
    except ValueError:
        raise ValueError(
            "Could not sample {} periods from data of size {}. Maybe the \
            validation_fraction is too high (>0.4)?".format(
                period_lengths, data_size
            )
        )

    # Select validation data
    validation_data = data.loc[data.index.unique()[validation_indices]]

    # Select the other data as training data
    train_data = data[~data.index.isin(validation_data.index)]

    return train_data, validation_data


def _sample_indices(
    number_indices: int, period_lengths: List[int], buffer_length: int
) -> np.array:
    """Samples periods of given length with the given buffer.

    Args:
        number_indices (int): Total number of indices that are available for
            sampling.
        period_lengths (list:int): List of lengths for each period that will be
            sampled.
        buffer_length (int): Number of indices between each sampled period that
            will be removed from the sampling set.

    Returns:
        numpy.array: Sorted (ascending) list of sampled indices.

    """
    stockpile = set(range(number_indices))

    rng = np.random.default_rng()
    sampled = set()
    for period_length in period_lengths:
        # Sample random starting indices from indices set
        start_point = rng.choice(list(stockpile))
        end_point = start_point + period_length

        # Append sampled indices
        sampled |= set(range(start_point, end_point))

        # Remove sampled indices plus a buffer zone.
        stockpile -= set(
            range(
                start_point - period_length - buffer_length, end_point + buffer_length
            )
        )

    return np.sort(list(sampled))
