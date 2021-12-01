# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
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

from openstef_dbc.services.prediction_job import PredictionJobDataClass

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
    # Folder where to store models
    trained_models_folder = Path(context.config.paths.trained_models_folder)

    # Determine if we need to optimize hyperparams

    # retrieve last model age where hyperparameters were optimized
    hyper_params_age = MLflowSerializer(trained_models_folder).get_model_age(
        pj["id"], hyperparameter_optimization_only=True
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
        trained_models_folder=trained_models_folder,
    )

    # Sent message to Teams
    title = (
        f'Optimized hyperparameters for prediction job {pj["name"]} {pj["description"]}'
    )

    teams.post_teams(teams.format_message(title=title, params=hyperparameters))


def main():
    taskname = Path(__file__).name.replace(".py", "")

    with TaskContext(taskname) as context:
        model_type = [ml.value for ml in MLModelType]

        PredictionJobLoop(context, model_type=model_type).map(
            optimize_hyperparameters_task, context
        )


if __name__ == "__main__":
    main()
