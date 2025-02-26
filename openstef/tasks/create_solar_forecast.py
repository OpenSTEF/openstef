# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the CRON job that is periodically executed to make prognoses of solar features.

These are useful for splitting the load in solar and wind contributions.

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::
        $ python create_solar_forecast

"""
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from openstef import PROJECT_ROOT
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

PV_COEFS_FILEPATH = PROJECT_ROOT / "openstef" / "data" / "pv_single_coefs.csv"


def make_solar_prediction_pj(pj, context, radius=30, peak_power=180961000.0):
    """Make a solar prediction for a specific prediction job.

    Args:
        pj: (dict) prediction job
        context: Task context
        radius: Radius us to collect PV systems.
        peak_power: Peak power.

    """
    context.logger.info("Get solar input data from database")
    # pvdata is only stored in the prd database
    solar_input = context.database.get_solar_input(
        (pj["lat"], pj["lon"]),
        pj["horizon_minutes"],
        pj["resolution_minutes"],
        radius=radius,
        sid=pj["sid"],
    )

    if len(solar_input) == 0:
        raise ValueError("Empty solar input")

    context.logger.info("Make solar prediction using Fides")
    power = fides(
        solar_input[["aggregated", "radiation"]].rename(
            columns=dict(radiation="insolation", aggregated="load")
        )
    )

    # if the forecast is for a region, output should be scaled to peak power
    if (radius != 0) and (not np.isnan(peak_power)):
        power = peak_power / max(solar_input.aggregated) * power
    context.logger.info("Store solar prediction in database")
    power["pid"] = pj["id"]
    power["type"] = "solar"
    power["algtype"] = "Fides"
    power["customer"] = pj["name"]
    power["description"] = pj["description"]
    context.database.write_forecast(power)


def combine_forecasts(forecasts, combination_coefs):
    """This function combines several independent forecasts into one, using predetermined coefficients.

    Input:
        - forecasts: pd.DataFrame(index = datetime, algorithm1, ..., algorithmn)
        - combinationcoefs: pd.DataFrame(param1, ..., paramn, algorithm1, ..., algorithmn)

    Output:
        - pd.DataFrame(datetime, forecast)

    """
    models = [x for x in list(forecasts) if x not in ["created", "datetime"]]

    # Add subset parameters to df
    # Identify which parameters should be used to define subsets based on the
    # combination coefs
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
            closest_match = min(coefs[param], key=lambda x, val=value: abs(x - val))
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
        result_df = pd.concat([result_df, result])
    # sort by datetime
    result_df.sort_values(["datetime", "created"], inplace=True)

    return result_df


def fides(data: pd.DataFrame, all_forecasts: bool = False):
    """Fides makes a forecast based on persistence and a direct fit with insolation.

    Args:
        data: pd.DataFrame(index = datetime, columns =['output','insolation'])
        all_forecasts: Should all forecasts be returned or only the combination

    Example:
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 300)))
    data['insolation'] = data.load * np.random.uniform(0.8, 1.2, len(index)) + 0.1
    data.loc[int(len(index)/3*2):,"load"] = np.nan

    """
    insolation_forecast = apply_fit_insol(data, add_to_df=False)
    persistence = apply_persistence(data, how="mean", smooth_entries=4, add_to_df=True)

    df = insolation_forecast.merge(persistence, left_index=True, right_index=True)

    coefs = pd.read_csv(PV_COEFS_FILEPATH)

    # Apply combination coefs
    df["created"] = df.loc[df.load.isnull()].index.min()
    forecast = combine_forecasts(
        df.loc[df.load.isnull(), ["forecaopenstefitInsol", "persistence", "created"]]
        .reset_index()
        .rename(columns=dict(index="datetime")),
        coefs,
    ).set_index("datetime")[["forecast"]]

    if all_forecasts:
        forecast = forecast.merge(
            df[["persistence", "forecaopenstefitInsol"]],
            left_index=True,
            right_index=True,
            how="left",
        )

    return forecast


def main(config=None, database=None, **kwargs):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specify a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        context.logger.info("Querying solar prediction jobs from database")
        prediction_jobs = context.database.get_prediction_jobs_solar()
        num_prediction_jobs = len(prediction_jobs)

        # only make customer = Provincie once an hour
        utc_now_minute = datetime.now(tz=UTC)().minute
        if utc_now_minute >= 15:
            prediction_jobs = [
                pj for pj in prediction_jobs if str(pj["name"]).startswith("Provincie")
            ]
            num_removed_jobs = num_prediction_jobs - len(prediction_jobs)
            num_prediction_jobs = len(prediction_jobs)
            context.logger.info(
                "Remove 'Provincie' solar predictions",
                num_removed_jobs=num_removed_jobs,
                num_prediction_jobs=num_prediction_jobs,
            )

        PredictionJobLoop(context, prediction_jobs=prediction_jobs).map(
            make_solar_prediction_pj, context, kwargs=kwargs
        )


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

    Example:
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 200)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 200)))

    """
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
    """This script calculates the persistence forecast.

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

    Example:
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, 300)))
    data.loc[200:,"load"] = np.nan

    """
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
    """This model fits insolation to PV yield and uses this fit to forecast PV yield. It uses a 2nd order polynomial.

    Input:
        - data: pd.DataFrame(index = datetime, columns = [load, insolation])
        Optional:
        - hoursDelta: period of forecast in hours [int] (e.g. every 6 hours for KNMI)
        - addToDF: Bool, add the norm to the data

    Output:
        - pd.DataFrame(index = datetime, columns = [(load), forecaopenstefitInsol])
    NB: range of datetime of input is equal to range of datetime of output

    Example:
    import pandas as pd
    import numpy as np
    index = pd.date_range(start = "2017-01-01 09:00:00", freq = '15T', periods = 300)
    data = pd.DataFrame(index = index,
                        data = dict(load=np.sin(index.hour/24*np.pi)*np.random.uniform(0.7,1.7, len(index))))
    data['insolation'] = data.load * np.random.uniform(0.8, 1.2, len(index)) + 0.1
    data.loc[int(len(index)/3*2):,"load"] = np.nan

    """
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
        return coefs[0] * values**2 + coefs[1] * values + coefs[2]

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
        res = optimize.minimize(fun, x0)
        # Apply fit
        df = second_order_poly(res.x, data[["insolation"]]).rename(
            columns=dict(insolation="forecaopenstefitInsol")
        )

    else:
        x0 = [1, 0]
        fun = (
            lambda x: (linear_fun(x, subset.insolation) - subset[colname]).abs().mean()
        )
        res = optimize.minimize(fun, x0)
        df = linear_fun(res.x, data[["insolation"]]).rename(
            columns=dict(insolation="forecaopenstefitInsol")
        )

    # Merge to dataframe if addToDF == True
    if add_to_df:
        if hours_delta is None:
            df = data.merge(df, left_index=True, right_index=True)
        else:
            df = pd.concat([data, df], axis=1)

    return df


if __name__ == "__main__":
    main()
