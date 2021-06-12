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
from datetime import datetime

from ktpbase.database import DataBase

from openstf.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext
from openstf.monitoring import teams

MAX_AGE_HYPER_PARAMS_DAYS = 31


def optimize_hyperparameters_task(pj: dict, context: TaskContext) -> None:

    db = DataBase.get_instance()

    if _last_optimimization_too_long_ago(pj) is not True:
        context.logger.info("Hyperparameters not old enough to optimize again")
        return

    hyperparameters = optimize_hyperparameters_pipeline(pj)

    db.write_hyper_params(pj, hyperparameters)

    # Sent message to Teams
    title = f'Optimized hyperparameters for prediction job {pj["name"]} {pj["description"]}'

    teams.post_teams(teams.format_message(title=title, params=hyperparameters))

def _last_optimimization_too_long_ago(pj):
    db = DataBase.get_instance()
    # Get data of last stored hyper parameters
    previous_optimization_datetime = db.get_hyper_params_last_optimized(pj)

    days_ago = (datetime.utcnow() - previous_optimization_datetime).days

    return days_ago > MAX_AGE_HYPER_PARAMS_DAYS


def main():
    with TaskContext("optimize_hyperparameters") as context:
        model_type = ["xgb", "xgb_quantile", "lgb"]

        PredictionJobLoop(context, model_type=model_type).map(
            optimize_hyperparameters_task, context
        )


if __name__ == "__main__":
    main()
