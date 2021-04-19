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

from openstf.feature_engineering.feature_free_days import create_holiday_functions
from openstf.feature_engineering.weather_features import (
    humidity_calculations,
    calculate_windspeed_at_hubheight,
    calculate_windturbine_power_output,
)
from openstf.feature_engineering.lag_features import generate_lag_feature_functions


def apply_features(data, feature_set_list=None, horizon=24, for_prediction=False):
    """This script applies the feature functions defined in
        feature_functions.py and returns the complete dataframe. Features requiring
        more recent label-data are omitted.
    Args:
        data (pandas.DataFrame): a pandas dataframe with input data in the form:
                                    pd.DataFrame(
                                        index=datetime,
                                        columns=[label, predictor_1,..., predictor_n]
                                    )
        feature_set_list (list of ints): minute lagtimes that where used during training of
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

    lag_functions = generate_lag_feature_functions(
        data, feature_set_list, horizon, for_prediction=for_prediction
    )

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
