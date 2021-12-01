# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""create_components_forecast.py

This module contains the CRON job that is periodically executed to make
the components prognoses and save them in the database.

This code assumes trained models are available from the persistent storage.
If these are not available run model_train.py to train all models.
To provide the prognoses the following steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
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

Attributes:


"""
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from openstef_dbc.services.prediction_job import PredictionJobDataClass

from openstef.enums import MLModelType
from openstef.pipeline.create_component_forecast import (
    create_components_forecast_pipeline,
)
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

T_BEHIND_DAYS = 0
T_AHEAD_DAYS = 3


def create_components_forecast_task(pj: PredictionJobDataClass, context: TaskContext):
    """Top level task that creates a components forecast.
    On this task level all database and context manager dependencies are resolved.

    Args:
        pj (PredictionJobDataClass): Prediction job
        context (TaskContext): Contect object that holds a config manager and a database connection
    """
    logger = structlog.get_logger(__name__)
    if pj["train_components"] == 0:
        context.logger.info(
            "Skip prediction job", train_components=pj["train_components"]
        )
        return

    # Define datetime range for input data
    datetime_start = datetime.utcnow() - timedelta(days=T_BEHIND_DAYS)
    datetime_end = datetime.utcnow() + timedelta(days=T_AHEAD_DAYS)

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

    # Get splitting coeficients
    split_coefs = context.database.get_energy_split_coefs(pj)

    if len(split_coefs) == 0:
        logger.warning(f"No Coefs found. Skipping pid", pid=pj["id"])
        return

    # Make forecast for the demand, wind and pv components
    forecasts = create_components_forecast_pipeline(
        pj, input_data, weather_data, split_coefs
    )

    # save forecast to database #######################################################
    context.database.write_forecast(forecasts)
    logger.debug("Written forecast to database")


def main():
    taskname = Path(__file__).name.replace(".py", "")

    with TaskContext(taskname) as context:

        model_type = [ml.value for ml in MLModelType]

        PredictionJobLoop(
            context,
            model_type=model_type,
        ).map(create_components_forecast_task, context)


if __name__ == "__main__":
    main()
