# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""create_solar_forecast
This module contains the CRON job that is periodically executed to make
prognoses of solar features that are usefull for splitting the load in solar and
wind contributions.
Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::
        $ python create_solar_forecast
Attributes:

"""
from datetime import datetime

import numpy as np
import pandas as pd

from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext
from openstf import PROJECT_ROOT
from openstf.feature_engineering.general import apply_fit_insol, apply_persistence

# TODO move to config
PV_COEFS_FILEPATH = PROJECT_ROOT / "openstf" / "data" / "pv_single_coefs.csv"


def make_solar_predicion_pj(pj, context):
    """Make a solar prediction for a spcecific prediction job.

    Args:
        pj: (dict) prediction job
    """
    context.logger.info("Get solar input data from database")
    # pvdata is only stored in the prd database
    solar_input = context.database.get_solar_input(
        (pj["lat"], pj["lon"]),
        pj["horizon_minutes"],
        pj["resolution_minutes"],
        radius=pj["radius"],
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
    if (pj["radius"] != 0) and (not np.isnan(pj["peak_power"])):
        power = pj["peak_power"] / max(solar_input.aggregated) * power
    context.logger.info("Store solar prediction in database")
    power["pid"] = pj["id"]
    power["type"] = "solar"
    power["algtype"] = "Fides"
    power["customer"] = pj["name"]
    power["description"] = pj["description"]
    context.database.write_forecast(power)


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


def main():
    with TaskContext(__file__) as context:
        context.logger.info("Querying wind prediction jobs from database")
        prediction_jobs = context.database.get_prediction_jobs_solar()
        num_prediction_jobs = len(prediction_jobs)

        # only make customer = Provincie once an hour
        utc_now_minute = datetime.utcnow().minute
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
            make_solar_predicion_pj, context
        )


if __name__ == "__main__":
    main()
