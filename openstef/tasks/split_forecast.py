# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""This module contains the CRON job that is periodically executed to make prognoses of solar features.

These features are usefull for splitting the load
in solar and wind contributions.
This is achieved by carrying out the folowing steps:
  1. Get the wind and solar reference data for the specific location of the
     customer
  2. Get the TDCV (Typical Domestic Consumption Values) data
  3. Fit a linear combination of above time series to the historic load data to
     determine the contributions of each energy source.
  4. Write the resulting coeficients to the SQL database.

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python split_forecast.py

"""
import logging
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
import structlog

import openstef.monitoring.teams as monitoring
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import ModelType
from openstef.settings import Settings
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

COEF_MAX_FRACTION_DIFF = 0.3


def main(config=None, database=None):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        model_type = [ml.value for ml in ModelType]

        PredictionJobLoop(
            context,
            model_type=model_type,
        ).map(split_forecast_task, context)


def split_forecast_task(
    pj: PredictionJobDataClass,
    context: TaskContext,
) -> pd.DataFrame:
    """Function that caries out the energy splitting for a specific prediction job with id pid.

    Args:
        pid: Prediction job id

    Returns:
        Energy splitting coefficients.

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)

    logger.info("Start splitting energy", pid=pj["id"])

    # Get input for splitting
    input_split_function = context.database.get_input_energy_splitting(pj)

    # Old split method;
    # find_components() gives two things:
    # - the split components (load, solar, wind, consumption, Inschatting (=sum of others) )
    # - coefdict: coefficients of each component; these are not yet an output of Dazls. Lets discuss with JM if we want that

    # Carry out the splitting
    components, coefdict = find_components(input_split_function)

    # Calculate mean absolute error (MAE)
    error = components[["load", "Inschatting"]].diff(axis=1).iloc[:, 1]
    mae = error.abs().mean()
    coefdict.update({"MAE": mae})
    coefsdf = convert_coefdict_to_coefsdf(pj, input_split_function, coefdict)

    # Get the coefs of previous runs and check if new coefs are valid
    last_coefsdict = context.database.get_energy_split_coefs(pj)
    last_coefsdf = convert_coefdict_to_coefsdf(pj, input_split_function, last_coefsdict)
    invalid_coefs = determine_invalid_coefs(coefsdf, last_coefsdf)
    if not invalid_coefs.empty:
        # If coefs not valid, do not update the coefs in the db and send teams
        # message that something strange is happening
        monitoring.post_teams(
            f"New splitting coefficient(s) for pid **{pj['id']}** deviate strongly "
            "from previously stored coefficients.",
            url=context.config.teams_monitoring_url,
            invalid_coefficients=invalid_coefs,
            coefficients_df=coefsdf,
        )
        # Use the last known coefficients for further processing
        return last_coefsdf
    else:
        # Save Results
        context.database.write_energy_splitting_coefficients(
            coefsdf, if_exists="append"
        )
        logger.info(
            "Succesfully wrote energy split coefficients to database", pid=pj["id"]
        )
        return coefsdf


def determine_invalid_coefs(
    new_coefs: pd.DataFrame, last_coefs: pd.DataFrame
) -> pd.DataFrame:
    """Determine which new coefficients are valid and return them.

    Args:
        new_coefs: df of new coefficients for standard load
            profiles (i.e. wind, solar, household)
        last_coefs: df of last coefficients for standard load
            profiles (i.e. wind, solar, household)

    Returns:
        Dataframe with invalid coefficients

    """
    merged_coefs = pd.merge(
        last_coefs, new_coefs, on="coef_name", how="left", suffixes=["_last", "_new"]
    )
    # calculate difference between new and last coefficients, if no new
    # coefficient, set difference to inf
    # If coefficient name is not present in new coefficients list, fail. If coefficient
    # name is not present in last coefficients list, add it.
    merged_coefs["difference"] = (
        (merged_coefs.coef_value_last - merged_coefs.coef_value_new)
        .abs()
        .fillna(np.inf)
    )
    # Check if the absolute difference between last coefficients and new coefficients
    # is more than COEF_MAX_FRACTION_DIFF x absolute value of last coefficient
    invalid_coefs = merged_coefs[
        merged_coefs.difference
        > (COEF_MAX_FRACTION_DIFF * merged_coefs.coef_value_last).abs()
    ]
    return invalid_coefs


def convert_coefdict_to_coefsdf(
    pj: PredictionJobDataClass, input_split_function: pd.DataFrame, coefdict: dict
) -> pd.DataFrame:
    """Convert dictionary of coefficients to dataframe with additional data for db storage.

    Args:
        pj: prediction job
        input_split_function: df of columns of standard load profiles,
            i.e. wind, solar, household
        coefdict: dict of coefficient per standard load profile

    Returns:
        DataFrame of coefficients to insert in sql

    """
    #
    sql_column_labels = ["pid", "date_start", "date_end", "created"]
    sql_colum_values = [
        pj["id"],
        input_split_function.index.min().date(),
        input_split_function.index.max().date(),
        datetime.now(tz=UTC),
    ]
    coefsdf = pd.DataFrame(
        {"coef_name": list(coefdict.keys()), "coef_value": list(coefdict.values())}
    )
    for i, column in enumerate(sql_column_labels):
        coefsdf[column] = sql_colum_values[i]

    return coefsdf


def find_components(
    df: pd.DataFrame, zero_bound: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Function that does the actual energy splitting.

    Args:
        df: Input data. The dataframe should contain these columns
            in exactly this order: [load, wind_ref, pv_ref, mulitple tdcv colums]
        zero_bound: If zero_bound is True coefficients can't be negative.

    Returns:
        tuple:
            - DataFrame containing the wind and solar components
            - Dict with the coefficients that result from the fitting

    """
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

    # Define function to fit
    def weighted_sum(x, *args):
        if len(x) != len(args):
            raise ValueError("Length of args should match len of x")
        weights = np.array([v for v in args])
        return np.dot(x.T, weights)

    # Carry out fitting
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html # noqa
    coefs, _ = scipy.optimize.curve_fit(
        weighted_sum,
        xdata=df.iloc[:, 1:].values.T,
        ydata=load.values,
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


if __name__ == "__main__":
    main()
