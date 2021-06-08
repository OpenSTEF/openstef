# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""model_train.py

This module contains the CRON job that is periodically executed to retrain the
prognosis models. For this the folowing steps are caried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Apply features
  3. Train and Test the new model
  4. Check if new model performs better than the old model
  5. Store the model if it performs better
  6. Send slack message to inform the users

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python model_train.py

"""
from pathlib import Path
from datetime import datetime, timedelta

from openstf.pipeline.train_model_sklearn import train_model_pipeline
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext

TRAINING_PERIOD_DAYS = 120

def train_model_task(pj, context):

    pj["hyper_params"] = {"training_period_days": TRAINING_PERIOD_DAYS, "featureset_name": "D"}
    pj["hyper_params"].update(context.database.get_hyper_params(pj))
    pj['feature_names'] = context.database.get_featureset(pj["hyper_params"]["featureset_name"])

    datetime_start = datetime.utcnow() - timedelta(
        days=int(pj["hyper_params"]["training_period_days"])
    )
    datetime_end = datetime.utcnow()

    # Get data from database
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    trained_models_folder = Path("C:\\repos\\icarus - demand - live\\trained_models")
    save_figures_folder = Path("C:\\Data\\icarus\\visuals\\trained_models") / str(pj["id"])
    # trained_models_folder = Path(context.config.paths.trained_models_folder)
    # save_figures_folder = Path(context.config.paths.webroot) / pj["id"]

    train_model_pipeline(pj, input_data, check_old_model_age=True,
                         trained_models_folder=trained_models_folder,
                         save_figures_folder=save_figures_folder)


def main():
    with TaskContext("train_model") as context:
        model_type = ["xgb", "xgb_quantile", "lgb"]

        PredictionJobLoop(context, model_type=model_type).map(
            train_model_task, context
        )


if __name__ == "__main__":
    main()
