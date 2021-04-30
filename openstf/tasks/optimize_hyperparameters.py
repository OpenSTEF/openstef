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

        $ python optimize_hyper_params.py

"""
from openstf.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def main():
    with TaskContext("optimize_hyperparameters") as context:
        model_type = ["xgb", "xgb_quantile", "lgb"]

        PredictionJobLoop(context, model_type=model_type).map(
            optimize_hyperparameters_pipeline
        )


if __name__ == "__main__":
    main()
