# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""apply_features.py.

This module provides functions for applying features to the input data.
This improves forecast accuracy. Examples of features that are added are:
    The load 1 day and 7 days ago at the same time
    If a day is a weekday or a holiday
    The extrapolated windspeed at 100m
    The normalised wind power according to the turbine-specific power curve

"""

import numpy as np
import pandas as pd
import scipy.signal

from openstf.feature_engineering.feature_free_days import create_holiday_functions
from openstf.feature_engineering.weather import humidity_calculations


def generate_lag_functions(data, minute_list=None, h_ahead=24):
    """Creates functions to generate lag features in a dataset.

    Args:
        data (pd.DataFrame): input data for an xgboost prediction or model training.
        minute_list (list of ints): minute lagtimes that where used during training
            of the model. If empty a new et will be automatically generated.
        h_ahead (int): Forecast horizon limit in hours.

    Returns:
        dict: dictionary with lag functions

    Example:
        lag_functions = generate_lag_functions(data,minute_list,h_ahead)
    """

    if minute_list is None:
        minute_list = []

    ##########
    # Define feature function groups
    lag_functions = {}
    # Add intraday-lags
    if h_ahead < 24:
        minminutes = int(np.round(np.ceil(h_ahead) * 60 / 15) * 15)
        minutespace = np.linspace(
            minminutes, 23 * 60, int(23 - minminutes / 60 + 1)
        ).tolist()

        # if no aditional lag time feature are defined investigate if we should add some
        if minute_list == []:
            minute_list = additional_minute_space(data)

        # add intra-hour values if hAhead > 1
        if h_ahead < 1:
            minutespace = set([15, 30, 45] + minutespace + minute_list)
        for minutes in minutespace:

            def func(x, shift=minutes):
                return x.shift(freq="T", periods=1 * shift)

            new = {"T-" + str(int(minutes)) + "min": func}
            lag_functions.update(new)

    # Define time lags in days:
    mindays = int(np.ceil(h_ahead / 24))
    for day in np.linspace(mindays, 14, 15 - mindays):

        def func(x, shift=day):
            return x.shift(freq="1d", periods=1 * shift)

        new = {"T-" + str(int(day)) + "d": func}
        lag_functions.update(new)
    return lag_functions


def additional_minute_space(data, height_treshold=0.1):
    """This script calculates an autocorrelation curve of the load trace. This curve is
        subsequently used to add additional lag times as features.

    Args:
        data (pandas.DataFrame): a pandas dataframe with input data in the form pd.DataFrame(index = datetime,
                             columns = [label, predictor_1,..., predictor_n])
        height_treshold (float): minimal autocorrelation value to be recognized as a peak.

    Returns:
        list of ints with aditional minute lags


    """

    def autocorr(x, lags):
        """Function to make a autocorrelation curve"""
        mean = x.mean()
        var = np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, "full")[len(x) - 1 :] / var / len(x)

        return corr[: len(lags)]

    try:
        # Get rid of nans as the autocorrelation handles these values badly
        data = data[data.columns[0]].dropna()
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


def apply_features(data, minute_list=None, h_ahead=24):
    """This script applies the feature functions defined in
        feature_functions.py and returns the complete dataframe. Features requiring
        more recent label-data are omitted.
    Args:
        data (pandas.DataFrame): a pandas dataframe with input data in the form:
                                    pd.DataFrame(
                                        index=datetime,
                                        columns=[label, predictor_1,..., predictor_n]
                                    )
        minute_list (list of ints): minute lagtimes that where used during training of
                                    the model. If empty a new et will be automatically
                                    generated.
        h_ahead (int): Forecast horizon limit in hours.

    Returns:
        pd.DataFrame(index = datetime, columns = [label, predictor_1,..., predictor_n,
            feature_1, ..., feature_m])

    Example:
        import pandas as pd
        import numpy as np
        index = pd.date_range(start = "2017-01-01 09:00:00",
        freq = '15T', periods = 200)
        data = pd.DataFrame(index = index,
                            data = dict(load=
                            np.sin(index.hour/24*np.pi)*
                            np.random.uniform(0.7,1.7, 200)))

    """
    if minute_list is None:
        minute_list = []

    lag_functions = generate_lag_functions(data, minute_list, h_ahead)

    # TODO it seems not all feature use the same convention
    # better to choose one and stick to it
    timedriven_functions = {
        "IsWeekendDay": lambda x: (x.index.weekday // 5) == 1,
        "IsWeekDay": lambda x: x.index.weekday < 5,
        "IsMonday": lambda x: x.index.weekday == 0,
        "IsTuesday": lambda x: x.index.weekday == 1,
        "IsWednesday": lambda x: x.index.weekday == 2,
        "IsThursday": lambda x: x.index.weekday == 3,
        "IsFriday": lambda x: x.index.weekday == 4,
        "IsSaturday": lambda x: x.index.weekday == 5,
        "IsSunday": lambda x: x.index.weekday == 6,
        "Month": lambda x: x.index.month,
        "Quarter": lambda x: x.index.quarter,
    }

    # Add check for specific hour
    for thour in np.linspace(0, 23, 24):

        def func(x, checkhour=thour):
            return x.index.hour == checkhour

        char = "0" if thour < 10 else ""
        new = {"Is" + char + str(int(thour)) + "Hour": func}
        timedriven_functions.update(new)

    df = data.copy()
    # holiday_function are imported at the beginning of the file and includes all dutch
    # school/work holidays
    holiday_functions = create_holiday_functions()

    for function_group in [lag_functions, timedriven_functions, holiday_functions]:
        for name, featfunc in function_group.items():
            df[name] = df.iloc[:, [0]].apply(featfunc)

    if "windspeed" in list(df):
        df["windspeed_100mExtrapolated"] = calculate_windspeed_at_hubheight(
            df.windspeed
        )
        df["windPowerFit_extrapolated"] = calculate_windturbine_power_output(
            df["windspeed_100mExtrapolated"]
        )

    if "windspeed_100m" in list(df):
        df["windpowerFit_harm_arome"] = calculate_windturbine_power_output(
            df["windspeed_100m"].astype(float)
        )

    # Try to add humidity  calculations, ignore if required columns are missing
    try:
        humidity_df = humidity_calculations(df.temp, df.humidity, df.pressure)
        df = df.join(humidity_df)
    except AttributeError:
        pass  # This happens when a required column for humidity_calculations
        # is not present

    return df


def apply_multiple_horizon_features(data, h_aheads=[0.25, 24], latency={"APX": 24}):
    """Based on ApplyFeatures, but suited for generating a table containing
    multiple time horizons at once./
    Also take function latency into consideration

    Input:
        - data: pd.DataFrame(index = datetime, columns =
                             [label, predictor_1,..., predictor_n])
        - hAhead: list. Forecast horizon limit in hours.
        Features requiring more recent label-data are omitted.
        - latency: dict(key=value). set data[key] to nan for horizon > value

    Output:
        - pd.DataFrame(index = datetime, columns =
            [label, predictor_1,..., predictor_n, feature_1, ..., feature_m])

    Example
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00",
                             freq = '15T', periods = 200)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*
                                    np.random.uniform(0.7,1.7, 200)))"""

    result = pd.DataFrame()
    cols = []
    for h_ahead in h_aheads:
        res = apply_features(data.copy(), h_ahead=h_ahead)
        res["Horizon"] = h_ahead
        if len(res.columns) > len(cols):
            cols = res.columns
        result = result.append(res, sort=False)  # appending unsorts columns

    # apply data latency
    for feature, time in latency.items():
        result.loc[result["Horizon"] > time, feature] = np.nan

    return result[cols].sort_index()


def calculate_windspeed_at_hubheight(windspeed, fromheight=10, hub_height=100):
    """
    function that extrapolates a wind from a certain height to 100m
    According to the wind power law (https://en.wikipedia.org/wiki/Wind_profile_power_law)

    input:
        - windspeed: float OR pandas series of windspeed at height = height
        - fromheight: height (m) of the windspeed data. Default is 10m
        - hubheight: height (m) of the turbine
    returns:
        - the windspeed at hubheight."""
    alpha = 0.143

    if not isinstance(windspeed, (np.ndarray, float, int, pd.core.series.Series)):
        raise TypeError(
            "The windspeed is not of the expected type!\n\
                        Got {}, expected np.ndarray, pd series or numeric".format(
                type(windspeed)
            )
        )

    try:
        if any(windspeed < 0):
            raise ValueError(
                "The windspeed cannot be negative, as it is the lenght of a vector"
            )
    except TypeError:
        if windspeed < 0:
            raise ValueError(
                "The windspeed cannot be negative, as it is the lenght of a vector"
            )
        windspeed = abs(windspeed)

    return windspeed * (hub_height / fromheight) ** alpha


def calculate_windturbine_power_output(windspeed, n_turbines=1, turbine_data=None):
    """This function calculates the generated wind power based on the wind speed.
    These values are related through the power curve, which is described by turbine_data.
    If no turbine_data is given, default values are used and results are normalized to 1MWp.
    If n_turbines=0, the result is normalized to a rated power of 1.

    Input:
        - windspeed: pd.DataFrame(index = datetime, columns = ["windspeedHub"])
        - nTurbines: int
        - turbineData: dict(slope_center, rated_power, steepness)

    Ouput:
        pd.DataFrame(index = datetime, columns = ["forecast"])"""

    if turbine_data is None:
        turbine_data = {
            "name": "Lagerwey L100",  # not used here
            "cut_in": 3,  # not used here
            "cut_off": 25,  # not used here
            "kind": "onshore",  # not used here
            "manufacturer": "Lagerwey",  # not used here
            "peak_capacity": 1,  # not used here
            "rated_power": 1,
            "slope_center": 8.07,
            "steepness": 0.664,
        }
    else:
        required_properties = ["rated_power", "steepness", "slope_center"]
        for prop in required_properties:
            if prop not in turbine_data.keys():
                raise KeyError(f"Required property '{prop}' not set in turbine data")

    generated_power = turbine_data["rated_power"] / (
        1
        + np.exp(
            -turbine_data["steepness"] * (windspeed - turbine_data["slope_center"])
        )
    )
    generated_power *= n_turbines

    return generated_power
