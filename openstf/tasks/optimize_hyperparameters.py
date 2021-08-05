# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
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

from openstf.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext
from openstf.monitoring import teams
from openstf.enums import MLModelType

MAX_AGE_HYPER_PARAMS_DAYS = 31
DEFAULT_TRAINING_PERIOD_DAYS = 91


def optimize_hyperparameters_task(pj: dict, context: TaskContext) -> None:
    """Optimize hyperparameters task.

    Expected prediction job keys: "id", "model", "lat", "lon", "name", "description"
    Only used for logging: "name", "description"

    Args:
        pj (dict): Prediction job
        context (TaskContext): Task context
    """

    # Determine if we need to optimize hyperparams
    datetime_last_optimized = context.database.get_hyper_params_last_optimized(pj)
    if datetime_last_optimized is not None:
        last_optimized_days = (datetime.utcnow() - datetime_last_optimized).days
        if last_optimized_days < MAX_AGE_HYPER_PARAMS_DAYS:
            context.logger.warning(
                "Skip hyperparameter optimization",
                pid=pj["id"],
                last_optimized_days=last_optimized_days,
                max_age=MAX_AGE_HYPER_PARAMS_DAYS,
            )
            return

    # Get input data (usese "id" and "model")
    current_hyperparams = context.database.get_hyper_params(pj)

    datetime_start = datetime.utcnow() - timedelta(
        days=int(
            current_hyperparams.get(
                "training_period_days", DEFAULT_TRAINING_PERIOD_DAYS
            )
        )
    )
    datetime_end = datetime.utcnow()

    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    # Optimize hyperparams
    hyperparameters = optimize_hyperparameters_pipeline(pj, input_data)

    context.database.write_hyper_params(pj, hyperparameters)

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
