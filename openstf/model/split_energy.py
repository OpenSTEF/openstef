# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import numpy as np
import pandas as pd
from ktpbase.database import DataBase
from ktpbase.log import logging
import scipy.optimize

import openstf.monitoring.teams as monitoring


def split_energy(pid):
    """Function that caries out the energy splitting for a specific prediction job with id pid

    Args:
        pid (int): Prediction job id

    Returns:
        pandas.DataFrame: Energy splitting coefficients.
    """
    # Make database connection
    db = DataBase()
    logger = logging.get_logger(__name__)

    # Get Prediction job
    pj = db.get_prediction_job(pid)

    logger.info("Start splitting energy", pid=pj["id"])

    # Get input for splitting
    input_split_function = db.get_input_energy_splitting(pj)

    # Carry out the splitting
    components, coefdict = find_components(input_split_function)

    # Calculate mean absolute error (MAE)
    # TODO: use a standard metric function for this
    error = components[["load", "Inschatting"]].diff(axis=1).iloc[:, 1]
    mae = error.abs().mean()
    coefdict.update({"MAE": mae})

    # Get average coefs of previous runs
    mean_coefs = db.get_energy_split_coefs(pj, mean=True)
    # Loop over keys and check if the difference with the average value is not more than 100%
    # In case the difference is more tha 100% of the average set KPI coefs as expected to False
    # If no previous coefs are stored an mean_coefs is empty and this loop wil not run
    for key in mean_coefs.keys():
        diff = mean_coefs[key] - coefdict[key]
        if diff > mean_coefs[key]:
            # Send teams message something strange is happening
            monitoring.post_teams_alert(
                "New splitting coefficients for pid {} deviate strongly from previously stored coefficients".format(
                    pj["id"]
                )
            )

    # Prepare dataframe to store in SQL database
    sql_column_labels = ["pid", "date_start", "date_end", "created"]
    sql_colum_values = [
        pj["id"],
        input_split_function.index.min().date(),
        input_split_function.index.max().date(),
        datetime.utcnow(),
    ]
    coefsdf = pd.DataFrame(
        {"coef_name": list(coefdict.keys()), "coef_value": list(coefdict.values())}
    )
    for i, column in enumerate(sql_column_labels):
        coefsdf[column] = sql_colum_values[i]

    # Save Results
    db.write_energy_splitting_coefficients(coefsdf, if_exists="append")
    logger.info("Succesfully wrote energy split coefficients to database", pid=pj["id"])
    return coefsdf


def find_components(df, zero_bound=True):
    """Function that does the actual energy splitting

    Args:
        df: Pandas data frame with input data. The dataframe should contain these
            columns in exactly this order: [load, wind_ref, pv_ref, mulitple tdcv colums]
        zerobound: Tells us wheter coefficients can be negative, true if this cannot be
            the case.

    Returns:
        tuple:
            [0] components: pandas dataframe containing the wind and solar components
            [0] pandas.DataFrame:
            [1] coefs: dict containing the coefficients that result from the fitting"""

    # Define function to fit
    def weighted_sum(x, *args):
        if len(x) != len(args):
            raise Exception("Length of args should match len of x")
        weights = np.array([v for v in args])
        return np.dot(x.T, weights)

    load = df.iloc[:, 0]
    wind_ref = df.iloc[:, 1]
    pv_ref = df.iloc[:, 2]

    # Define scaler
    nedu_scaler = (load.max() - load.min()) / 10

    # Come up with inital guess for the fitting
    p_wind_guess = 1.0
    ppv_guess = 1.0
    p0 = [p_wind_guess, ppv_guess] + (len(df.columns) - 3) * [nedu_scaler]

    # Define fitting bounds
    if zero_bound:
        bounds = (0, "inf")
    else:
        bounds = ("-inf", "inf")

    # Carry out fitting
    coefs, cov = scipy.optimize.curve_fit(
        weighted_sum,
        df.iloc[:, 1:].values.T,
        load.values,
        p0=p0,
        bounds=bounds,
        method="trf",
    )

    # Set 'almost zero' to zero
    coefs[coefs < 0.1] = 0

    # Reconstuct historical load
    hist = weighted_sum(df.iloc[:, 1:].values.T, *coefs)
    histp0 = weighted_sum(df.iloc[:, 1:].values.T, *p0)

    # Make a nice dataframe to return the components
    components = df.iloc[:, [0]].copy()
    components["Inschatting"] = hist.T
    components["p0"] = histp0.T
    components["Windopwek"] = wind_ref * coefs[0]
    components["Zonne-opwek"] = pv_ref * coefs[1]
    components["StandaardVerbruik"] = (df.iloc[:, 3:] * coefs[2:]).sum(axis=1)
    components["Residu"] = -1 * components.iloc[:, 0:2].diff(axis=1).iloc[:, 1]

    # Make nice dictinary to return coefficents
    coefdict = {name: value for name, value in zip(df.columns[1:], coefs)}

    # Return result
    return components, coefdict
