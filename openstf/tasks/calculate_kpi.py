# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""calculate_kpi.py
This module contains the CRON job that is periodically executed to calculate key
performance indicators (KPIs) and save them to the database.
This code assumes prognoses are available from the persistent storage. If these are not
available run create_forecast.py to train all models.

The folowing tasks are caried out:
  1: Calculate the KPI for a given pid. Ignore SplitEnergy
  2: Create figures
  3: Write KPI to database

Example:
    This module is meant to be called directly from a CRON job.
    Alternatively this code can be run directly by running::
        $ python calculate_kpi.py
Attributes:
"""
# Import builtins
from datetime import datetime, timedelta

# Import project modules
from openstf.model.performance import calc_kpi_for_specific_pid
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext

# Thresholds for retraining and optimizing
THRESHOLD_RETRAINING = 0.25
THRESHOLD_OPTIMIZING = 0.50


def check_kpi_pj(pj, context, start_time, end_time):
    kpis = calc_kpi_for_specific_pid(pj["id"], start_time=start_time, end_time=end_time)
    # Write KPI's to database
    context.database.write_kpi(pj, kpis)

    # Add pid to the list of pids that should be retrained or optimized if
    # performance is insufficient
    if kpis["47.0h"]["rMAE"] > THRESHOLD_RETRAINING:
        context.logger.warning(
            "Need to retrain model, retraining threshold rMAE 47h exceeded",
            t_ahead="47.0h",
            rMAE=kpis["47.0h"]["rMAE"],
            retraining_threshold=THRESHOLD_RETRAINING,
        )
        function_name = "train_specific_model"
        context.logger.info("Adding tracy job", function=function_name)
        context.database.ktp_api.add_tracy_job(pj["id"], function=function_name)

    if kpis["47.0h"]["rMAE"] > THRESHOLD_OPTIMIZING:
        context.logger.warning(
            "Need to optimize hyperparameters, optimizing threshold rMAE 47h exceeded",
            t_ahead="47.0h",
            rMAE=kpis["47.0h"]["rMAE"],
            optimizing_threshold=THRESHOLD_OPTIMIZING,
        )
        function_name = "optimize_hyperparameters_for_specific_pid"
        context.logger.info("Adding tracy job", function=function_name)
        context.database.ktp_api.add_tracy_job(pj["id"], function=function_name)


def main():
    with TaskContext(__file__) as context:
        model_type = ["xgb", "lgb"]
        # Set start and end time
        start_time = datetime.date(datetime.utcnow()) - timedelta(days=1)
        end_time = datetime.date(datetime.utcnow())

        if datetime.utcnow().weekday() in [0, 1, 2, 3, 4]:
            PredictionJobLoop(context, model_type=model_type).map(
                check_kpi_pj,
                context,
                start_time=start_time,
                end_time=end_time,
            )


if __name__ == "__main__":
    main()
