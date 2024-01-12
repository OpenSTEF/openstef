# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the CRON job that is periodically executed to retrain the prognosis models.

For this the folowing steps are caried out:
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

from openstef.enums import MLModelType, PipelineType
from openstef.exceptions import (
    SkipSaveTrainingForecasts,
    InputDataOngoingZeroFlatlinerError,
)
from openstef.pipeline.train_model import (
    train_model_pipeline,
    train_pipeline_step_load_model,
    MAXIMUM_MODEL_AGE,
)
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

from openstef.model.serializer import MLflowSerializer

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

    Top level task that trains a new model and makes sure the best available model is
    stored. On this task level all database and context manager dependencies are resolved.

    Expected prediction job keys:  "id", "model", "lat", "lon", "name"

    Args:
        pj: Prediction job
        context: Contect object that holds a config manager and a
            database connection.
        check_old_model_age: check if model is too young to be retrained
        datetime_start: Start
        datetime_end: End

    """
    # Check pipeline types
    if PipelineType.TRAIN not in pj.pipelines_to_run:
        context.logger.info(
            "Skip this PredictionJob because train pipeline is not specified in the pj."
        )
        return

    # TODO: Improve implementation by using a field in the database and leveraging the
    #       `pipelines_to_run` attribute of the `PredictionJobDataClass` object. This
    #       would require a change to the MySQL datamodel.
    if (
        context.config.externally_posted_forecasts_pids
        and pj.id in context.config.externally_posted_forecasts_pids
    ):
        context.logger.info(
            "Skip this PredictionJob because its forecasts are posted by an external process."
        )
        return

    # Get the paths for storing model and reports from the config manager
    mlflow_tracking_uri = context.config.paths_mlflow_tracking_uri
    context.logger.debug(f"MLflow tracking uri: {mlflow_tracking_uri}")
    artifact_folder = context.config.paths_artifact_folder
    context.logger.debug(f"Artifact folder: {artifact_folder}")

    context.perf_meter.checkpoint("Added metadata to PredictionJob")

    # Check the model age before retrieving the input data to speed up train job.
    # (The exact same model age check is also part of the "train_model_pipeline".)

    # Initialize serializer
    serializer = MLflowSerializer(mlflow_tracking_uri=mlflow_tracking_uri)

    # Get old model and age
    _, _, old_model_age = train_pipeline_step_load_model(pj, serializer)

    # Check old model age and continue yes/no
    if (old_model_age < MAXIMUM_MODEL_AGE) and check_old_model_age:
        context.perf_meter.checkpoint(
            f"Old model is younger than {MAXIMUM_MODEL_AGE} days, skip training"
        )
        if pj.save_train_forecasts:
            raise SkipSaveTrainingForecasts
        return

    # Define start and end of the training input data
    if datetime_end is None:
        datetime_end = datetime.utcnow()
    if datetime_start is None:
        datetime_start = datetime_end - timedelta(days=TRAINING_PERIOD_DAYS)

    # Get training input data from database
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    context.perf_meter.checkpoint("Retrieved timeseries input")

    # Excecute the model training pipeline
    try:
        data_sets = train_model_pipeline(
            pj,
            input_data,
            check_old_model_age=check_old_model_age,
            mlflow_tracking_uri=mlflow_tracking_uri,
            artifact_folder=artifact_folder,
        )

        if data_sets:
            context.perf_meter.checkpoint("Model trained")
        else:
            context.perf_meter.checkpoint("Model not trained")

        if pj.save_train_forecasts:
            if data_sets is None:
                raise RuntimeError("Forecasts were not retrieved")
            if not hasattr(context.database, "write_train_forecasts"):
                raise RuntimeError(
                    "Database connector does dot support 'write_train_forecasts' while "
                    "'save_train_forecasts option was activated.'"
                )
            context.database.write_train_forecasts(pj, data_sets)
            context.logger.debug(f"Saved Forecasts from trained model on datasets")
    except SkipSaveTrainingForecasts:
        context.logger.debug(f"Skip saving forecasts")
    except InputDataOngoingZeroFlatlinerError:
        if (
            context.config.known_zero_flatliners
            and pj.id in context.config.known_zero_flatliners
        ):
            context.logger.info(
                "No model was trained for this known zero flatliner. No model needs to be trained either, since the fallback forecasts are sufficient."
            )
            return
        else:
            raise InputDataOngoingZeroFlatlinerError(
                'All recent load measurements are zero. Check the load profile of this pid as well as related/neighbouring prediction jobs. Afterwards, consider adding this pid to the "known_zero_flatliners" app_setting and possibly removing other pids from the same app_setting.'
            )


def main(model_type=None, config=None, database=None):
    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a config object and/or database connection object. These"
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
