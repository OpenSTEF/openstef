# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""This module contains the CRON job that is periodically executed to make prognoses of wind features.

These features are usefull for splitting the load in solar and wind contributions and
making prognoses.

Example:
    This module is meant to be called directly from a CRON job. A description of the
    CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::
        $ python create_wind_forecast

"""
from pathlib import Path

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.feature_engineering import weather_features
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext
from enums import WeatherColumnName, LocationColumnName, ForecastColumnName


def make_wind_forecast_pj(pj: PredictionJobDataClass, context: TaskContext) -> None:
    """Make a wind prediction for a specific prediction job.

    Args:
        pj: Prediction job
        context: Context manager

    """
    context.logger.info("Get turbine data", turbine_type=pj[WeatherColumnName.TURBINE_TYPE])
    turbine_data = context.database.get_power_curve(pj[WeatherColumnName.TURBINE_TYPE])

    context.logger.info(
        "Get windspeed", location=[pj[LocationColumnName.LAT], pj[LocationColumnName.LON]], hub_height=pj[WeatherColumnName.WIND_HUB_HEIGHT]
    )
    windspeed = context.database.get_wind_input(
        (pj[LocationColumnName.LAT], pj[LocationColumnName.LON]),
        pj[WeatherColumnName.WIND_HUB_HEIGHT],
        pj["horizon_minutes"],
        pj["resolution_minutes"],
    )

    context.logger.info("Calculate windturbine power", n_turbines=pj[WeatherColumnName.NUMBER_TURBINES])
    power = weather_features.calculate_windturbine_power_output(
        windspeed, pj[WeatherColumnName.NUMBER_TURBINES], turbine_data
    ).rename(columns=dict(windspeed_100m="forecast"))

    context.logger.info("Store wind prediction in database")
    power[ForecastColumnName.PID] = pj["id"]
    power[ForecastColumnName.TYPE] = "wind"
    power[ForecastColumnName.GENERAL_TYPE] = "powerCurve"
    power[ForecastColumnName.CUSTOMER] = pj["name"]
    power[ForecastColumnName.DESCRIPTION] = pj[ForecastColumnName.DESCRIPTION]
    context.database.write_forecast(power, t_ahead_series=True)


def main(config=None, database=None):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        context.logger.info("Querying wind prediction jobs from database")
        prediction_jobs = context.database.get_prediction_jobs_wind()
        prediction_jobs = [x for x in prediction_jobs if x["model"] == "latest"]

        PredictionJobLoop(context, prediction_jobs=prediction_jobs).map(
            make_wind_forecast_pj, context
        )


if __name__ == "__main__":
    main()
