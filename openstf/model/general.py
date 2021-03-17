# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd
from ktpbase.log import logging

from openstf import PROJECT_ROOT
from openstf.data_validation import data_validation
from openstf.feature_engineering.apply_features import apply_multiple_horizon_features
from openstf.feature_engineering.general import (
    apply_fit_insol,
    apply_persistence,
    remove_features_not_in_set,
)


# TODO replace this with ModelType (MLModelType == Machine Learning model type)
class MLModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    LGB = "lgb"


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


# TODO move to config
PV_COEFS_FILEPATH = PROJECT_ROOT / "openstf" / "data" / "pv_single_coefs.csv"


def split_data_train_validation_test(
    data,
    test_fraction=0.0,
    validation_fraction=0.15,
    back_test=False,
    period_sampling=True,
    period_timedelta=timedelta(days=2),
):
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
        back_test (bool): Indicates if data is intended for a back test.
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
    logger = logging.get_logger(__name__)

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

        # In either case validation data is before the training data
        validation_data = data[start_date_val:start_date_train]

    # Sample periods in the training part as the validation set.
    else:
        if back_test:
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


def sample_validation_data_periods(data, validation_fraction=0.15, period_length=192):
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


def _sample_indices(number_indices, period_lengths, buffer_length):
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


def pre_process_data(data, featureset=None, horizons=None):
    """Function that automates the pre processing of the data.

    Args:
        data (pd.DataFrame): Data with (unvalidated) input data and without features.

    Returns:

        pd.DataFrame: Cleaned data with features.

    """
    if horizons is None:
        horizons = [0.25, 47]

    # Validate input data
    validated_data = data_validation.validate(data)

    # Apply features
    # TODO it would be nicer to only generate the required features
    validated_data_data_with_features = apply_multiple_horizon_features(
        validated_data, h_aheads=horizons
    )

    # remove features not in requested set if required
    if featureset is not None:
        validated_data_data_with_features = remove_features_not_in_set(
            validated_data_data_with_features, featureset
        )

    # Clean up data
    clean_data_with_features = data_validation.clean(validated_data_data_with_features)

    return clean_data_with_features


def combine_forecasts(forecasts, combination_coefs):
    """This function combines several independent forecasts into one, using
        predetermined coefficients.

    Input:
        - forecasts: pd.DataFrame(index = datetime, algorithm1, ..., algorithmn)
        - combinationcoefs: pd.DataFrame(param1, ..., paramn, algorithm1, ..., algorithmn)

    Output:
        - pd.DataFrame(datetime, forecast)"""

    models = [x for x in list(forecasts) if x not in ["created", "datetime"]]

    # Add subset parameters to df
    # Identify which parameters should be used to define subsets based on the
    # combinationcoefs
    subset_columns = [
        "tAhead",
        "hForecasted",
        "weekday",
        "hForecastedPer6h",
        "tAheadPer2h",
        "hCreated",
    ]
    subset_defs = [x for x in list(combination_coefs) if x in subset_columns]

    df = forecasts.copy()
    # Now add these subsetparams to df
    if "tAhead" in subset_defs:
        t_ahead = (df["datetime"] - df["created"]).dt.total_seconds() / 3600
        df["tAhead"] = t_ahead

    if "hForecasted" in subset_defs:
        df["hForecasted"] = df.datetime.dt.hour

    if "weekday" in subset_defs:
        df["weekday"] = df.datetime.dt.weekday

    if "hForecastedPer6h" in subset_defs:
        df["hForecastedPer6h"] = pd.to_numeric(
            np.floor(df.datetime.dt.hour / 6) * 6, downcast="integer"
        )

    if "tAheadPer2h" in subset_defs:
        df["tAheadPer2h"] = pd.to_numeric(
            np.floor((df.datetime - df.created).dt.total_seconds() / 60 / 60 / 2) * 2,
            downcast="integer",
        )

    if "hCreated" in subset_defs:
        df["hCreated"] = df.created.dt.hour

    # Start building combinationcoef dataframe that later will be multiplied with the
    # individual forecasts
    # This is the best way for a backtest:
    #    uniquevalues = list([np.unique(df[param].values) for param in subsetDefs])
    #    permutations = list(itertools.product(*uniquevalues))

    # This is the best way for a single forecast
    permutations = [tuple(x) for x in df[subset_defs].values]

    result_df = pd.DataFrame()

    for subsetvalues in permutations:
        subset = df.copy()
        coefs = combination_coefs

        # Create subset based on all subsetparams, for forecasts and coefs
        for value, param in zip(subsetvalues, subset_defs):
            subset = subset.loc[subset[param] == value]
            # Define function which find closest match of a value from an array of values.
            #  Use this later to find best coefficient from the given subsetting dividers
            closest_match = min(coefs[param], key=lambda x: abs(x - value))
            coefs = coefs.loc[coefs[param] == closest_match]
            # Find closest matching value for combinationCoefParams corresponding to
            # available subsetValues

        # Of course, not all possible subsets have to be defined in the forecast.
        # Skip empty subsets
        if len(subset) == 0:
            continue

        # Multiply forecasts with their coefficients
        result = np.multiply(subset[models], np.array(coefs[models]))
        result["forecast"] = result.apply(np.nansum, axis=1)
        # Add handling with NA values for a single forecast
        result["coefsum"] = np.nansum(coefs[models].values)
        nanselector = np.isnan(subset[models].iloc[0].values)
        result["nonnacoefsum"] = np.nansum(coefs[models].values.flatten() * nanselector)
        result["forecast"] = (
            result["forecast"]
            * result["coefsum"]
            / (result["coefsum"] - result["nonnacoefsum"])
        )
        result["datetime"] = subset["datetime"]
        result["created"] = subset["created"]
        result = result[["datetime", "created", "forecast"]]

        result_df = result_df.append(result)

    # for safety: remove duplicate results to prevent multiple forecasts for the same
    # time created
    # resultDF.drop_duplicates(keep='last', inplace = True)

    #    #rename created column to datetime and add datetime for export
    #    resultDF.reset_index(inplace = True)
    #    resultDF.columns = ['datetime','created', 'forecast']
    #
    # sort by datetime
    result_df.sort_values(["datetime", "created"], inplace=True)

    return result_df


def fides(data, all_forecasts=False):
    """Fides makes a forecast based on persistence and a direct fit with insolation.
    Input:
        - data = pd.DataFrame(index = datetime, columns =['output','insolation'])
    Optional:
        - allForecasts = Bool. Should all forecasts be returned or only the combination
    Example:
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 300)))
    data['insolation'] = data.load * np.random.uniform(0.8, 1.2, len(index)) + 0.1
    data.loc[int(len(index)/3*2):,"load"] = np.NaN"""

    insolation_forecast = apply_fit_insol(data, add_to_df=False)
    persistence = apply_persistence(data, how="mean", smooth_entries=4, add_to_df=True)

    df = insolation_forecast.merge(persistence, left_index=True, right_index=True)

    coefs = pd.read_csv(PV_COEFS_FILEPATH)

    # Apply combination coefs
    df["created"] = df.loc[df.load.isnull()].index.min()
    forecast = combine_forecasts(
        df.loc[df.load.isnull(), ["forecaopenstfitInsol", "persistence", "created"]]
        .reset_index()
        .rename(columns=dict(index="datetime")),
        coefs,
    ).set_index("datetime")[["forecast"]]

    if all_forecasts:
        forecast = forecast.merge(
            df[["persistence", "forecaopenstfitInsol"]],
            left_index=True,
            right_index=True,
            how="left",
        )

    return forecast
