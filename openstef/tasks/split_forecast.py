# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""split_forecast.py

This module contains the CRON job that is periodically executed to make
prognoses of solar features. These features are usefull for splitting the load
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

Attributes:


"""

#imports from dazls model

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, normalize
from chemsy.predict.methods import
from sklearn.linear_model import Lasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from chemsy.prep import *
from sklearn.decomposition import KernelPCA, PCA
import random
from sklearn.utils import shuffle
from scipy.linalg import fractional_matrix_power
from sklearn.covariance import LedoitWolf, OAS
from sklearn.base import BaseEstimator

# Seed, path, etc,
random.seed(999)
np.random.seed(999)

path = os.path.dirname(os.path.abspath(__file__))
folder = ['\\prep_data\\']
combined_data = []
station_name = []

# Read prepared data
for file_name in glob.glob(path + folder[0] + '*.csv'):
    x = pd.read_csv(file_name, low_memory=False, parse_dates=["datetime"])
    x["datetime"] = pd.to_datetime(x["datetime"])
    x = x.set_index('datetime')
    combined_data.append(x)
    sn = os.path.basename(file_name)
    station_name.append(sn[:len(sn) - 4])

n_delay=1
# CHOOSE THE DATA, METADATA and TARGET, ETC. BY INDEX
cc=len(combined_data[0].columns)-4
xindex=list(np.arange(0,n_delay*3))+list(np.arange(n_delay*3+2,cc))
x2index=list(np.arange(n_delay*3+2,cc))
yindex=[n_delay*3,n_delay*3+1]

#PREPARATION
ori_combined_data=combined_data.copy() #Good procedure to prevent data changing in-place
domain_model_clf=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the domain model
adaptation_model_clf=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the adaptation model

nn=len(station_name)
for n in range(nn): #loop through all stations (leave one out)
    print(station_name[n])
    model=DAZLS() #Initialize DAZLS model
    model.fit(combined_data=ori_combined_data, xindex=xindex,x2index=x2index,yindex=yindex,n=n,domain_model_clf=domain_model_clf,adaptation_model_clf=adaptation_model_clf,n_delay=n_delay,cc=cc) #Fit model
    y=model.predict() #get predicted y
    model.score() #print prediction performance

#end here

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
import structlog

import openstef.monitoring.teams as monitoring
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import MLModelType
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

COEF_MAX_FRACTION_DIFF = 0.3


def main(config=None, database=None):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a configmanager and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        model_type = [ml.value for ml in MLModelType]

        PredictionJobLoop(
            context,
            model_type=model_type,
        ).map(split_forecast_task, context)


def split_forecast_task(
    pj: PredictionJobDataClass,
    context: TaskContext,
):
    """Function that caries out the energy splitting for a specific prediction job with
    id pid.

    Args:
        pid (int): Prediction job id

    Returns:
        pandas.DataFrame: Energy splitting coefficients.
    """
    logger = structlog.get_logger(__name__)

    logger.info("Start splitting energy", pid=pj["id"])

    # Get input for splitting
    input_split_function = context.database.get_input_energy_splitting(pj)

    # Carry out the splitting
    components, coefdict = find_components(input_split_function)

    # Calculate mean absolute error (MAE)
    # TODO: use a standard metric function for this
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
            url=context.config.teams.monitoring_url,
            invalid_coefs=invalid_coefs,
            coefsdf=coefsdf,
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



def determine_invalid_coefs(new_coefs, last_coefs):
    """Determine which new coefficients are valid and return them.

    Args:
        new_coefs (pd.DataFrame): df of new coefficients for standard load
            profiles (i.e. wind, solar, household)
        last_coefs (pd.DataFrame): df of last coefficients for standard load
            profiles (i.e. wind, solar, household)

    Returns:
        pd.DataFrame: df of invalid coefficients
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


def convert_coefdict_to_coefsdf(pj, input_split_function, coefdict):
    """Convert dictionary of coefficients to dataframe with additional data for db
    storage.

    Args:
        pj (PredictionJobDataClass): prediction job
        input_split_function (pd.DataFrame): df of columns of standard load profiles,
            i.e. wind, solar, household
        coefdict (dict): dict of coefficient per standard load profile

    Returns:
        pd.DataFrame: df of coefficients to insert in sql
    """
    #
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

    return coefsdf


def find_components(df, zero_bound=True):
    """Function that does the actual energy splitting

    Args:
        df (pandas.DataFrame): Input data. The dataframe should contain these columns
            in exactly this order: [load, wind_ref, pv_ref, mulitple tdcv colums]
        zero_bound (bool): If zero_bound is True coefficients can't be negative.

    Returns:
        tuple:
            [0] pandas.DataFrame: Containing the wind and solar components
            [1] dict: The coefficients that result from the fitting
    """

    # Define function to fit
    def weighted_sum(x, *args):
        if len(x) != len(args):
            raise ValueError("Length of args should match len of x")
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
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html # noqa
    coefs, cov = scipy.optimize.curve_fit(
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
