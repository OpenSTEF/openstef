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

from openstf.feature_engineering.holiday_features import (
    create_holiday_feature_functions,
)
from openstf.feature_engineering.weather_features import (
    add_humidity_features,
    add_additional_wind_features,
)
from openstf.feature_engineering.lag_features import generate_lag_feature_functions


def apply_features(data, feature_set_list=None, horizon=24):
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
        horizon (int): Forecast horizon limit in hours.

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

    # Get lag feature functions
    lag_feature_functions = generate_lag_feature_functions(
        data, feature_set_list, horizon
    )

    # Get timedrivenfeature functions
    timedriven_feature_functions = {
        "IsWeekendDay": lambda x: (x.index.weekday // 5) == 1,
        "IsWeekDay": lambda x: x.index.weekday < 5,
        "IsSunday": lambda x: x.index.weekday == 6,
        "Month": lambda x: x.index.month,
        "Quarter": lambda x: x.index.quarter,
    }

    # Get holiday feature functions
    holiday_feature_functions = create_holiday_feature_functions()

    # Only select features that occur in feature_set_list,
    # if feature_set_list is none nothing is removed
    if feature_set_list is not None:
        timedriven_feature_functions = {
            key: timedriven_feature_functions[key]
            for key in feature_set_list
            if key in timedriven_feature_functions
        }
        holiday_feature_functions = {
            key: holiday_feature_functions[key]
            for key in feature_set_list
            if key in holiday_feature_functions
        }

    # Add the features to the dataframe using previously defined feature functions
    for function_group in [
        lag_feature_functions,
        timedriven_feature_functions,
        holiday_feature_functions,
    ]:
        for name, featfunc in function_group.items():
            data[name] = data.iloc[:, [0]].apply(featfunc)

    # Add additional winf features
    data = add_additional_wind_features(data, feature_set_list)

    # Add humidity features
    data = add_humidity_features(data, feature_set_list)

    # Return dataframe including all requested features
    return data
