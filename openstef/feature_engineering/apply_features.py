# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module provides functionality for applying features to the input data to improve forecast accuracy.

Examples of features that are added:
    - The load 1 day and 7 days ago at the same time.
    - If a day is a weekday or a holiday.
    - The extrapolated windspeed at 100m.
    - The normalised wind power according to the turbine-specific power curve.

"""

import pandas as pd

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.feature_engineering.historic_features import (
    add_historic_load_as_a_feature,
)
from openstef.feature_engineering.holiday_features import (
    generate_holiday_feature_functions,
)
from openstef.feature_engineering.lag_features import generate_lag_feature_functions
from openstef.feature_engineering.weather_features import (
    add_additional_solar_features,
    add_additional_wind_features,
    add_humidity_features,
)


def apply_features(
    data: pd.DataFrame,
    pj: PredictionJobDataClass = None,
    feature_names: list[str] = None,
    horizon: float = 24.0,
) -> pd.DataFrame:
    """Applies the feature functions defined in ``feature_functions.py`` and returns the complete dataframe.

    Features requiring more recent label-data are omitted.

    .. note::
        For the time deriven features only the onces in the features list will be added. But for the weather features all will be added at present.
        These unrequested additional features have to be filtered out later.

    Args:
        data (pandas.DataFrame): a pandas dataframe with input data in the form:
                                    pd.DataFrame(
                                        index=datetime,
                                        columns=[label, predictor_1,..., predictor_n]
                                    )
        pj (PredictionJobDataClass): Prediction job.
        feature_names (list[str]): list of reuqested features
        horizon (float): Forecast horizon limit in hours.

    Returns:
        pd.DataFrame(index = datetime, columns = [label, predictor_1,..., predictor_n, feature_1, ..., feature_m])

    Example output:

    .. code-block:: py

        import pandas as pd
        import numpy as np
        index = pd.date_range(start = "2017-01-01 09:00:00",
        freq = '15T', periods = 200)
        data = pd.DataFrame(index = index,
                            data = dict(load=
                            np.sin(index.hour/24*np.pi)*
                            np.random.uniform(0.7,1.7, 200)))

    """
    # Add if needed the proloaf feature (historic_load)
    data = add_historic_load_as_a_feature(data, pj)

    # Get lag feature functions
    feature_functions = generate_lag_feature_functions(feature_names, horizon)

    # Get timedrivenfeature functions
    feature_functions.update(
        {
            "IsWeekendDay": lambda x: (x.index.weekday // 5) == 1,
            "IsWeekDay": lambda x: x.index.weekday < 5,
            "IsSunday": lambda x: x.index.weekday == 6,
            "Month": lambda x: x.index.month,
            "Quarter": lambda x: x.index.quarter,
        }
    )

    # Get holiday feature functions
    feature_functions.update(generate_holiday_feature_functions())

    # Add the features to the dataframe using previously defined feature functions
    for key, featfunc in feature_functions.items():
        # Don't generate feature is not in features
        if feature_names is not None and key not in feature_names:
            continue
        data.loc[:, key] = data.iloc[:, [0]].apply(featfunc)

    # Add additional wind features
    data = add_additional_wind_features(data, feature_names)

    # Add humidity features
    data = add_humidity_features(data, feature_names)

    # Add solar features; when pj is unavailable a default location is used.
    data = add_additional_solar_features(data, pj, feature_names)

    # Return dataframe including all requested features
    return data
