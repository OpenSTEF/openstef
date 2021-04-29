# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""create_forecast.py

This module contains the CRON job that is periodically executed to make prognoses and
save them in to the database.
This code assumes trained models are available from the persistent storage. If these
are not available run model_train.py to train all models.
To provide the prognoses the folowing steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Apply features
  3. Load model
  4. Make prediction
  5. Write prediction to the database
  6. Send Teams message if something goes wrong

Example:
    This module is meant to be called directly from a CRON job.
    Alternatively this code can be run directly by running::

        $ python create_forecast.py

Attributes:


"""
from openstf.pipeline.create_forecast import create_forecast_pipeline
from openstf.enums import ForecastType
from openstf.tasks.utils.utils import check_status_change, update_status_change
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def main():
    with TaskContext("create_forecast") as context:
        model_type = ["xgb", "xgb_quantile", "lgb"]

        # status file callback after every iteration
        # TODO change implementation to a database one
        def callback(pj, successful):
            status_id = "Pred {}, {}".format(pj["name"], pj["description"])
            status_code = 0 if successful else 2

            if check_status_change(status_id, status_code):
                context.logger.warning("Status changed", status_code=status_code)

                update_status_change(status_id, status_code)

        PredictionJobLoop(
            context,
            model_type=model_type,
            on_end_callback=callback,
            # Debug specific pid
            # prediction_jobs=[{'id':282}],
        ).map(create_forecast_pipeline, forecast_type=ForecastType.DEMAND)


if __name__ == "__main__":
    main()
