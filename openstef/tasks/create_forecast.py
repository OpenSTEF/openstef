# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""create_forecast.py

This module contains the CRON job that is periodically executed to make prognoses and
save them in to the database.
This code assumes trained models are available from the persistent storage. If these
are not available run model_train.py to train all models.
To provide the prognoses the folowing steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Apply features
  3. Load model
  4. Make prediction
  5. Write prediction to the database
  6. Send Teams message if something goes wrong

Example:
    This module is meant to be called directly from a CRON job.
    Alternatively this code can be run directly by running::

        $ python create_forecast.py

Attributes:


"""
from datetime import datetime, timedelta
from pathlib import Path

from openstef_dbc.services.prediction_job import PredictionJobDataClass

from openstef.enums import MLModelType
from openstef.pipeline.create_forecast import create_forecast_pipeline
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

T_BEHIND_DAYS: int = 14
T_AHEAD_DAYS: int = 2


def create_forecast_task(pj: PredictionJobDataClass, context: TaskContext) -> None:
    """Top level task that creates a forecast.

    On this task level all database and context manager dependencies are resolved.

    Expected prediction job keys: "id", "lat", "lon", "resolution_minutes",
        "horizon_minutes", "type", "name", "quantiles"

    Args:
        pj (PredictionJobDataClass): Prediction job
        context (TaskContext): Contect object that holds a config manager and a database connection
    """
    # Extract trained models folder
    trained_models_folder = context.config.paths.trained_models_folder

    # Define datetime range for input data
    datetime_start = datetime.utcnow() - timedelta(days=T_BEHIND_DAYS)
    datetime_end = datetime.utcnow() + timedelta(days=T_AHEAD_DAYS)

    # Retrieve input data
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )
    # Make forecast with the forecast pipeline
    forecast = create_forecast_pipeline(pj, input_data, trained_models_folder)

    # Write forecast to the database
    context.database.write_forecast(forecast, t_ahead_series=True)


def main(model_type=None):

    taskname = Path(__file__).name.replace(".py", "")

    with TaskContext(taskname) as context:

        if model_type is None:
            model_type = [ml.value for ml in MLModelType]

        PredictionJobLoop(context, model_type=model_type).map(
            create_forecast_task, context
        )


if __name__ == "__main__":
    main()
