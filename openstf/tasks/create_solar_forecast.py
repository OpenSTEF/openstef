# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""create_solar_forecast
This module contains the CRON job that is periodically executed to make
prognoses of solar features that are usefull for splitting the load in solar and
wind contributions.
Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::
        $ python create_solar_forecast
Attributes:

"""
from datetime import datetime

import numpy as np

from openstf.model.general import fides
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def make_solar_predicion_pj(pj, context):
    """Make a solar prediction for a spcecific prediction job.

    Args:
        pj: (dict) prediction job
    """
    context.logger.info("Get solar input data from database")
    # pvdata is only stored in the prd database
    solar_input = context.database.get_solar_input(
        (pj["lat"], pj["lon"]),
        pj["horizon_minutes"],
        pj["resolution_minutes"],
        radius=pj["radius"],
        sid=pj["sid"],
    )

    if len(solar_input) == 0:
        raise ValueError("Empty solar input")

    context.logger.info("Make solar prediction using Fides")
    power = fides(
        solar_input[["aggregated", "radiation"]].rename(
            columns=dict(radiation="insolation", aggregated="load")
        )
    )

    # if the forecast is for a region, output should be scaled to peak power
    if (pj["radius"] != 0) and (not np.isnan(pj["peak_power"])):
        power = pj["peak_power"] / max(solar_input.aggregated) * power
    context.logger.info("Store solar prediction in database")
    power["pid"] = pj["id"]
    power["type"] = "solar"
    power["algtype"] = "Fides"
    power["customer"] = pj["name"]
    power["description"] = pj["description"]
    context.database.write_forecast(power)


def main():
    with TaskContext(__file__) as context:
        context.logger.info("Querying wind prediction jobs from database")
        prediction_jobs = context.database.get_prediction_jobs_solar()
        num_prediction_jobs = len(prediction_jobs)

        # only make customer = Provincie once an hour
        utc_now_minute = datetime.utcnow().minute
        if utc_now_minute >= 15:
            prediction_jobs = [
                pj for pj in prediction_jobs if str(pj["name"]).startswith("Provincie")
            ]
            num_removed_jobs = num_prediction_jobs - len(prediction_jobs)
            num_prediction_jobs = len(prediction_jobs)
            context.logger.info(
                "Remove 'Provincie' solar predictions",
                num_removed_jobs=num_removed_jobs,
                num_prediction_jobs=num_prediction_jobs,
            )

        PredictionJobLoop(context, prediction_jobs=prediction_jobs).map(
            make_solar_predicion_pj, context
        )


if __name__ == "__main__":
    main()
