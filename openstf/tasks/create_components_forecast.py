# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""create_components_forecast.py

This module contains the CRON job that is periodically executed to make
the components prognoses and save them in the database.

This code assumes trained models are available from the persistent storage.
If these are not available run model_train.py to train all models.
To provide the prognoses the following steps are carried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Apply features
  3. Load model
  4. Make component prediction
  5. Write prediction to the database
  6. Send Teams message if something goes wrong

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python create_components_forecast.py

Attributes:


"""
from openstf.pipeline.create_forecast import make_components_prediction
from openstf.tasks.utils.utils import check_status_change, update_status_change
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def create_components_forecast_pj(pj, context):
    if pj["train_components"] == 0:
        context.logger.info(
            "Skip prediction job", train_components=pj["train_components"]
        )
        return

    # Make forecast for the demand, wind and pv components
    make_components_prediction(pj)


def main():
    with TaskContext("create_components_forecast") as context:

        # status file callback after every iteration
        # TODO change implementation to a database one
        def callback(pj, successful):
            status_id = "Pred {}, {}".format(pj["name"], pj["description"])
            status_code = 0 if successful else 2

            if check_status_change(status_id, status_code):
                context.logger.warning("Status changed", status_code=status_code)

                update_status_change(status_id, status_code)

        model_type = ["xgb", "xgb_quantile", "lgb"]

        PredictionJobLoop(
            context,
            model_type=model_type,
            on_end_callback=callback,
        ).map(create_components_forecast_pj, context)


if __name__ == "__main__":
    main()
