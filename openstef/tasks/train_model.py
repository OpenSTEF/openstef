# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""model_train.py

This module contains the CRON job that is periodically executed to retrain the
prognosis models. For this the folowing steps are caried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Apply features
  3. Train and Test the new model
  4. Check if new model performs better than the old model
  5. Store the model if it performs better
  6. Send slack message to inform the users

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python model_train.py

"""
from datetime import datetime, timedelta
from pathlib import Path

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import MLModelType
from openstef.pipeline.train_model import train_model_pipeline
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

TRAINING_PERIOD_DAYS: int = 120
DEFAULT_CHECK_MODEL_AGE: bool = True


def train_model_task(
    pj: PredictionJobDataClass,
    context: TaskContext,
    check_old_model_age: bool = DEFAULT_CHECK_MODEL_AGE,
    datetime_start: datetime = None,
    datetime_end: datetime = None,
) -> None:
    """Train model task.

    Top level task that trains a new model and makes sure the beast available model is
    stored. On this task level all database and context manager dependencies are resolved.

    Expected prediction job keys:  "id", "model", "lat", "lon", "name"

    Args:
        pj (PredictionJobDataClass): Prediction job
        context (TaskContext): Contect object that holds a config manager and a
            database connection.
        check_old_model_age (bool): check if model is too young to be retrained
    """
    # Get the paths for storing model and reports from the config manager
    mlflow_tracking_uri = context.config.paths.mlflow_tracking_uri
    context.logger.debug(f"MLflow tracking uri: {mlflow_tracking_uri}")
    artifact_folder = context.config.paths.artifact_folder
    context.logger.debug(f"Artifact folder: {artifact_folder}")

    context.perf_meter.checkpoint("Added metadata to PredictionJob")

    # Define start and end of the training input data
    if datetime_end is None:
        datetime_end = datetime.utcnow()
    if datetime_start is None:
        datetime_start = datetime_end - timedelta(days=TRAINING_PERIOD_DAYS)

    # todo: See if we can check model age before getting the data
    # Get training input data from database
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    context.perf_meter.checkpoint("Retrieved timeseries input")

    # Excecute the model training pipeline
    data_sets = train_model_pipeline(
        pj,
        input_data,
        check_old_model_age=check_old_model_age,
        mlflow_tracking_uri=mlflow_tracking_uri,
        artifact_folder=artifact_folder,
    )

    if pj.save_train_forecasts:
        if data_sets is None:
            raise RuntimeError("Forecasts were not retrieved")
        if not hasattr(context.database, "write_train_forecasts"):
            raise RuntimeError(
                "Database connector does dot support 'write_train_forecasts' while "
                "'save_train_forecasts option was activated.'"
            )
        context.database.write_train_forecasts(pj, data_sets)

    context.perf_meter.checkpoint("Model trained")


def main(model_type=None, config=None, database=None):

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a configmanager and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    if model_type is None:
        model_type = [ml.value for ml in MLModelType]

    taskname = Path(__file__).name.replace(".py", "")
    datetime_now = datetime.utcnow()
    with TaskContext(taskname, config, database) as context:
        PredictionJobLoop(context, model_type=model_type).map(
            train_model_task, context, datetime_end=datetime_now
        )


if __name__ == "__main__":
    main()
