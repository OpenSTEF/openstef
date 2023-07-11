# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module should be executed once every day.

For all prediction_jobs, it will create a 'basecase' forecast which is less accurate, but (almost) always available.
For now, it uses the load a week earlier.
Missing datapoints are interpolated.

Example:
    This module is meant to be called directly from a CRON job. A description of the
    CRON job can be found in the /k8s/CronJobs folder.

    Alternatively this code can be run directly by running:

        $ python create_basecase_forecast.py

"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.enums import PipelineType
from openstef.pipeline.create_basecase_forecast import create_basecase_forecast_pipeline
from openstef.tasks.utils.predictionjobloop import PredictionJobLoop
from openstef.tasks.utils.taskcontext import TaskContext

T_BEHIND_DAYS: int = 15
T_AHEAD_DAYS: int = 14


def create_basecase_forecast_task(
    pj: PredictionJobDataClass, context: TaskContext
) -> None:
    """Top level task that creates a basecase forecast.

    On this task level all database and context manager dependencies are resolved.

    Args:
        pj: Prediction job
        context: Contect object that holds a config manager and a database connection

    """
    # Check pipeline types
    if PipelineType.FORECAST not in pj.pipelines_to_run:
        context.logger.info(
            "Skip this PredictionJob because forecast pipeline is not specified in the pj."
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

    # Make basecase forecast using the corresponding pipeline
    basecase_forecast = create_basecase_forecast_pipeline(pj, input_data)

    # Do not store basecase forecasts for moments within next 48 hours.
    # Those should be updated by regular forecast process.
    basecase_forecast = basecase_forecast.loc[
        basecase_forecast.index
        > (pd.to_datetime(datetime.utcnow(), utc=True) + timedelta(hours=48)),
        :,
    ]

    # Write basecase forecast to the database
    context.database.write_forecast(basecase_forecast, t_ahead_series=True)


def main(config: object = None, database: object = None):
    taskname = Path(__file__).name.replace(".py", "")

    if database is None or config is None:
        raise RuntimeError(
            "Please specifiy a config object and/or database connection object. These"
            " can be found in the openstef-dbc package."
        )

    with TaskContext(taskname, config, database) as context:
        model_type = ["xgb", "xgb_quantile", "lgb"]

        PredictionJobLoop(context, model_type=model_type).map(
            create_basecase_forecast_task, context
        )


if __name__ == "__main__":
    main()
