# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the CRON job that is periodically executed to make the components prognoses.

This code assumes trained models are available from the persistent storage.
If these are not available run model_train.py to train all models.
To provide the prognoses the following steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and day_ahead_electricity_price price data)
  2. Apply features
  3. Load model
  4. Make component prediction
  5. Write prediction to the database
  6. Send Teams message if something goes wrong

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python create_components_forecast.py

"""
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pandas as pd
import structlog

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import ModelType
from openstef.exceptions import ComponentForecastTooShortHorizonError
from openstef.pipeline.create_component_forecast import (
    create_components_forecast_pipeline,
)
from openstef.settings import Settings
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

T_BEHIND_DAYS = 0
T_AHEAD_DAYS = 3


def create_components_forecast_task(
    pj: PredictionJobDataClass,
    context: TaskContext,
    t_behind_days: int = T_BEHIND_DAYS,
    t_ahead_days: int = T_AHEAD_DAYS,
) -> None:
    """Top level task that creates a components forecast.

    On this task level all database and context manager dependencies are resolved.

    Args:
        pj: Prediction job
        context: Contect object that holds a config manager and a database connection
        t_behind_days: number of days in the past that the component forecast is created for
        t_ahead_days: number of days in the future that the component forecast is created for

    Raises:
        ComponentForecastTooShortHorizonError: If the forecast horizon is too short
         (less than 30 minutes in advance)

    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(Settings.log_level)
        )
    )
    logger = structlog.get_logger(__name__)
    if pj["train_components"] == 0:
        context.logger.info(
            "Skip prediction job", train_components=pj["train_components"]
        )
        return

    # Define datetime range for input data
    datetime_start = datetime.now(tz=UTC) - timedelta(days=t_behind_days)
    datetime_end = datetime.now(tz=UTC) + timedelta(days=t_ahead_days)

    logger.info(
        "Get predicted load", datetime_start=datetime_start, datetime_end=datetime_end
    )
    # Get most recent load forecast as input_data,
    # we use a regular forecast as input point for creating component forecasts
    input_data = context.database.get_predicted_load(
        pj, start_time=datetime_start, end_time=datetime_end
    )
    # Check if input_data is not empty
    if len(input_data) == 0:
        logger.warning(f"No forecast found. Skipping pid", pid=pj["id"])
        return

    logger.info("retrieving weather data")
    # TODO make openstef_dbc function to retrieve inputdata for component forecast in one call,
    #  this will make this function much shorter
    # Get required weather data
    weather_data = context.database.get_weather_data(
        [pj["lat"], pj["lon"]],
        [
            "radiation",
            "windspeed_100m",
        ],  # These variables are used when determing the splitting coeficients, and should therefore be reused when making the component forcasts.
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    # Make forecast for the demand, wind and pv components
    forecasts = create_components_forecast_pipeline(pj, input_data, weather_data)

    ## Perform sanity check on index
    if not isinstance(forecasts.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise ValueError(
            f"Index is not datetime. Received forecasts:{forecasts.head()}"
        )

    # save forecast to database #######################################################
    context.database.write_forecast(forecasts)
    logger.debug("Written forecast to database")

    # Check if forecast was complete enough, otherwise raise exception
    if forecasts.index.max() < datetime.now(tz=UTC) + timedelta(hours=30):
        # Check which input data is missing the most.
        # Do this by counting the NANs for (load)forecast, radiation and windspeed
        max_index = forecasts.index.max()
        n_nas = dict(
            nans_load_forecast=input_data.loc[max_index:, "forecast"].isna().sum(),
            nans_radiation=weather_data.loc[max_index:, "radiation"].isna().sum(),
            nans_windspeed_100m=weather_data.loc[max_index:, "windspeed_100m"]
            .isna()
            .sum(),
        )
        max_na = max(n_nas, key=n_nas.get)

        raise ComponentForecastTooShortHorizonError(
            f"Could not make component forecast for two days ahead, probably input data is missing, {max_na}: {n_nas[max_na]}"
        )


def main(config: object = None, database: object = None, **kwargs):
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
        ).map(create_components_forecast_task, context, **kwargs)


if __name__ == "__main__":
    main()
