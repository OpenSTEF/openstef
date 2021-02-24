# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
""" This module contains all general feature engineering functions"""
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy

from openstf.feature_engineering import apply_features


def extract_minute_features(feature_names):
    """Creates a list of features that were used during training of the input model
    Args:
        feature_names (list[str]): Feature names

    Returns:
        minute_list (list[int]): list of minute lags that were used as features during training
    """

    minutes_list = []
    for feature in feature_names:
        # TODO Should this module really know how the feature names are written?
        # Perhaps better if this function uses a list with all possible minute features
        # or a tempalte. In any case the knowlegde should come from the module which
        # creates the features.
        m = re.search(r"T-(\d+)min", feature)

        if m is None:
            continue
        else:
            minutes_list.append(int(m[1]))

    return minutes_list


def calc_completeness(df, weights=None, time_delayed=False, homogenise=True):
    """Calculate the (weighted) completeness of a dataframe.

    NOTE: NA values count as incomplete

    Args:
        df (pd.DataFrame): Dataframe with a datetimeIndex index
        weights: Array-compatible with size equal to columns of df.
            used to weight the completeness of each column
        time_delayed (bool): Should there be a correction for T-x columns
        homogenise (bool): Should the index be resampled to median time delta -
            only available for DatetimeIndex

    Returns:
        float: Completeness
    """

    if weights is None:
        weights = np.array([1] * len(df.columns))
    weights = np.array(weights)

    if homogenise and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:

        median_timediff = int(
            df.reset_index().iloc[:, 0].diff().median().total_seconds() / 60.0
        )
        df = df.resample("{:d}T".format(median_timediff)).mean()

    if time_delayed is False:
        # Calculate completeness
        # Completeness per column
        completeness_per_column = df.count() / len(df)

    # if timeDelayed is True, we correct that time-delayed columns
    # also in the best case will have NA values. E.g. T-2d is not available
    # for times ahead of more than 2 days
    elif time_delayed:
        # assume 15 minute forecast resolution
        # timecols: {delay:number of points expected to be missing}
        # number of points expected to be missing = numberOfPointsUpToTwoDaysAhead - numberOfPointsAvailable
        timecols = {
            x: len(df) - eval(x[2:].replace("min", "/60").replace("d", "*24.0")) / 0.25
            for x in df.columns
            if x[:2] == "T-"
        }

        non_na_count = df.count()
        for col, value in timecols.items():
            non_na_count[col] += value

        # Correct for APX being only expected to be available up to 24h
        if "APX" in non_na_count.index:
            non_na_count["APX"] += max([len(df) - 96, 0])

        completeness_per_column = non_na_count / len(df)

    # scale to weights and normalize
    completeness = (completeness_per_column * weights).sum() / weights.sum()

    return completeness


def nan_repeated(df, max_length, column_name):
    """
    This function replaces repeating values (sequentially repeating values),
    which repeat longer than a set max_length (in data points) with NaNs.

    Args:
        df (pandas.DataFrame): Data from which you would like to set repeating values to nan
        max_length (int): If a value repeats more often, sequentially, than this value, all those points are set to NaN
        column_name (string): the pandas dataframe column name of the column you want to process

    Rrturns:
        pandas.DataFrame: data, similar to df, with the desired values set to NaN.
    """
    data = df.copy(deep=True)
    indices = []
    old_value = -1000000000000
    value = 0
    for index, r in data.iterrows():
        value = r[column_name]
        if value == old_value:
            indices.append(index)
        elif (value != old_value) & (len(indices) > max_length):
            indices = indices[max_length:]
            data.at[indices, column_name] = np.nan
            indices = []
            indices.append(index)
        elif (value != old_value) & (len(indices) <= max_length):
            indices = []
            indices.append(index)
        old_value = value
    if len(indices) > max_length:
        data.at[indices, column_name] = np.nan
    return data


def get_preprocessed_data(
    pj,
    db,
    datetime_start=datetime.utcnow() - timedelta(days=90),
    datetime_end=datetime.utcnow(),
):
    """Retrieves input data frame and pre-process it (calculate features, correct constant load, drop NAs etc.)
    Input:
        pj, the current prediction job
    Output:
        table: pd.DataFrame
        completeness: percentage of data kept after pre-processing"""
    location = [pj["lat"], pj["lon"]]

    if "+" in pj["description"]:
        name_var = pj["description"]
    else:
        name_var = pj["sid"]

    table = db.get_model_input(
        pid=pj["id"],
        location=location,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    if "load" not in table.columns:
        raise SystemError("No Load returned for {}".format(pj["description"]))

    # !!!!!
    # Work-around for cityzen data
    if name_var.split("_")[0] == "cable":
        print("NOTE! Rescaling input data!")
        table.iloc[:, 0] *= 1000.0

    # Drop 'false' measurements. e.g. where load appears to be constant.
    threshold = 6 * 4  # number of repeated values
    table = nan_repeated(table, threshold, table.columns[0])
    const_load_values = len(table) - len(table.iloc[:, 0].dropna())
    print("Changed {} values of constant load to NA.".format(const_load_values))

    # Apply model features to get the full table
    horizons = [0.25, 47]
    table = apply_features.apply_multiple_horizon_features(table, h_aheads=horizons)
    # Drop first rows where not enough data was present to make T-14d for example
    # For now, use fixed limit of first two weeks

    completeness = calc_completeness(table, time_delayed=True)

    return table, completeness


def calc_norm(data, how="max", add_to_df=True):
    """This script calculates the norm of a given dataset.
    Input:
        - data: pd.DataFrame(index = datetime, columns = [load])
        - how: str can be any function from numpy, recognized by np.'how'
        Optional:
        - add_to_df: Bool, add the norm to the data

    Output:
        - pd.DataFrame(index = datetime, columns = [load])
    NB: range of datetime of input is equal to range of datetime of output

    Example
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 200)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 200)))"""

    colname = list(data)[0]
    if how == "max":
        df = data.groupby(data.index.time).apply(lambda x: x.max(skipna=True))
    if how == "mean":
        df = data.groupby(data.index.time).apply(lambda x: x.mean(skipna=True))

    # rename
    df.rename(columns={colname: "Norm"}, inplace=True)

    # Merge to dataframe if add_to_df == True
    if add_to_df:
        df = data.merge(df, left_on=data.index.time, right_index=True)[
            [colname, "Norm"]
        ].sort_index()

    return df


def apply_persistence(data, how="mean", smooth_entries=4, add_to_df=True, colname=None):
    """This script calculates the persistence forecast
    Input:
        - data: pd.DataFrame(index = datetime, columns = [load]), datetime is expected to have historic values, as well as NA values
        Optional:
        - how: str, how to determine the norm (abs or mean)
        - smoothEntries: int, number of historic entries over which the persistence is smoothed
        - add_to_df: Bool, add the forecast to the data
        - option of specifying colname if load is not first column

    Output:
        - pd.DataFrame(index = datetime, columns = [(load,) persistence])
    NB: range of datetime of input is equal to range of datetime of output

    Example
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 300)))
    data.loc[200:,"load"] = np.nan"""

    data = data.sort_index()

    if colname is None:
        colname = list(data)[0]

    df = calc_norm(data, how=how, add_to_df=True)

    # this selects the last non NA values
    last_entries = df.loc[df[colname].notnull()][-smooth_entries:]

    norm_mean = last_entries.Norm.mean()
    if norm_mean == 0:
        norm_mean = 1

    factor = last_entries[colname].mean() / norm_mean
    df["persistence"] = df.Norm * factor

    if add_to_df:
        df = df[[colname, "persistence"]]
    else:
        df = df[["persistence"]]

    return df


def apply_fit_insol(data, add_to_df=True, hours_delta=None, polynomial=False):
    """This model fits insolation to PV yield and uses this fit to forecast PV yield.
    It uses a 2nd order polynomial

    Input:
        - data: pd.DataFrame(index = datetime, columns = [load, insolation])
        Optional:
        - hoursDelta: period of forecast in hours [int] (e.g. every 6 hours for KNMI)
        - addToDF: Bool, add the norm to the data

    Output:
        - pd.DataFrame(index = datetime, columns = [(load), forecaopenstfitInsol])
    NB: range of datetime of input is equal to range of datetime of output

    Example
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, len(index))))
    data['insolation'] = data.load * np.random.uniform(0.8, 1.2, len(index)) + 0.1
    data.loc[int(len(index)/3*2):,"load"] = np.NaN"""

    colname = list(data)[0]

    # Define subset, only keep non-NaN values and the most recent forecasts
    # This ensures a good training set
    if hours_delta is None:
        subset = data.loc[(data[colname].notnull()) & (data[colname] > 0)]
    else:
        subset = data.loc[
            (data[colname].notnull())
            & (data[colname] > 0)
            & (data["tAhead"] < timedelta(hours=hours_delta))
            & (data["tAhead"] >= timedelta(hours=0))
        ]

    def linear_fun(coefs, values):
        return coefs[0] * values + coefs[1]

    def second_order_poly(coefs, values):
        return coefs[0] * values ** 2 + coefs[1] * values + coefs[2]

    # Define function to be minimized and subsequently minimize this function
    if polynomial:
        # Define starting guess
        x0 = [1, 1, 0]  # ax**2 + bx + c.
        fun = (
            lambda x: (second_order_poly(x, subset.insolation) - subset[colname])
            .abs()
            .mean()
        )
        # , bounds = bnds, constraints = cons)
        res = scipy.optimize.minimize(fun, x0)
        # Apply fit
        df = second_order_poly(res.x, data[["insolation"]]).rename(
            columns=dict(insolation="forecaopenstfitInsol")
        )

    else:
        x0 = [1, 0]
        fun = (
            lambda x: (linear_fun(x, subset.insolation) - subset[colname]).abs().mean()
        )
        res = scipy.optimize.minimize(fun, x0)
        df = linear_fun(res.x, data[["insolation"]]).rename(
            columns=dict(insolation="forecaopenstfitInsol")
        )

    # Merge to dataframe if addToDF == True
    if add_to_df:
        if hours_delta is None:
            df = data.merge(df, left_index=True, right_index=True)
        else:
            df = pd.concat([data, df], axis=1)

    return df


def add_missing_feature_columns(input_data, featurelist):
    """Adds feature column for features in the featurelist.

    Add feature columns for features in the feature list if these columns don't
    exist in the input data. If a column is added, its value is set to NaN.
    """
    missing_features = [f for f in featurelist if f not in list(input_data)]

    for feature in missing_features:
        print(f"Warning: adding NaN column for missing feature: {feature}")
        input_data[feature] = np.nan

    return input_data


def remove_features_not_in_set(data_with_features, featureset):
    """Remove all features not in the given featureset.

    Args:
        data_with_features (pandas.DataFrame): Data with features
        featureset_name (list): Name of the featureset to keep (if exists)

    Returns:
        pandas.DataFrame: Data with features not in featureset removed
    """
    # always keep the load and the Horizon column
    columns_to_keep = ["load", "Horizon"]

    features_to_keep = (
        columns_to_keep
        +
        # this makes sure we only keep features that actually exist
        [f for f in featureset if f in data_with_features.columns]
    )

    return data_with_features[features_to_keep]
