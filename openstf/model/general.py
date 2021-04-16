# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from enum import Enum

import numpy as np
import pandas as pd

from openstf import PROJECT_ROOT
from openstf.validation import validation
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
    validated_data = validation.validate(data)

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
    clean_data_with_features = validation.clean(validated_data_data_with_features)

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

    Args:
        data: pd.DataFrame(index = datetime, columns =['output','insolation'])
        allForecasts (bool): Should all forecasts be returned or only the combination

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
