# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""optimize_hyper_params.py

This module contains the CRON job that is periodically executed to optimize the
hyperparameters for the prognosis models.

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python optimize_hyperparameters.py

"""
from datetime import datetime, timedelta
from pathlib import Path

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import MLModelType
from openstef.model.serializer import MLflowSerializer
from openstef.monitoring import teams
from openstef.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

MAX_AGE_HYPER_PARAMS_DAYS = 31
DEFAULT_TRAINING_PERIOD_DAYS = 121


def optimize_hyperparameters_task(
    pj: PredictionJobDataClass, context: TaskContext
) -> None:
    """Optimize hyperparameters task.

    Expected prediction job keys: "id", "model", "lat", "lon", "name", "description"
    Only used for logging: "name", "description"

    Args:
        pj (PredictionJobDataClass): Prediction job
        context (TaskContext): Task context
    """
    # Get the paths for storing model and reports from the config manager
    mlflow_tracking_uri = context.config.paths.mlflow_tracking_uri
    artifact_folder = context.config.paths.artifact_folder

    # Determine if we need to optimize hyperparams
    # retrieve last model age where hyperparameters were optimized
    mlflow_serializer = MLflowSerializer(mlflow_tracking_uri=mlflow_tracking_uri)
    hyper_params_age = mlflow_serializer.get_model_age(
        experiment_name=str(pj["id"]), hyperparameter_optimization_only=True
    )

    if hyper_params_age < MAX_AGE_HYPER_PARAMS_DAYS:
        context.logger.warning(
            "Skip hyperparameter optimization",
            pid=pj["id"],
            hyper_params_age=hyper_params_age,
            max_age=MAX_AGE_HYPER_PARAMS_DAYS,
        )
        return

    datetime_start = datetime.utcnow() - timedelta(days=DEFAULT_TRAINING_PERIOD_DAYS)
    datetime_end = datetime.utcnow()

    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    # Optimize hyperparams
    hyperparameters = optimize_hyperparameters_pipeline(
        pj,
        input_data,
        mlflow_tracking_uri=mlflow_tracking_uri,
        artifact_folder=artifact_folder,
    )

    # Sent message to Teams
    title = (
        f'Optimized hyperparameters for prediction job {pj["name"]} {pj["description"]}'
    )

    teams.post_teams(teams.format_message(title=title, params=hyperparameters))


def main(config=None, database=None):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specify a configmanager and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        model_type = [ml.value for ml in MLModelType]

        PredictionJobLoop(context, model_type=model_type).map(
            optimize_hyperparameters_task, context
        )


if __name__ == "__main__":
    main()
