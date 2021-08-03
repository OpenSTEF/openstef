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

from openstf.pipeline.train_model import (
    train_model_pipeline,
    MAXIMUM_MODEL_AGE,
    get_model_age,
)
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext
from openstf.enums import MLModelType

TRAINING_PERIOD_DAYS: int = 120
DEFAULT_CHECK_MODEL_AGE: bool = True


def train_model_task(
    pj: dict, context: TaskContext, check_old_model_age: bool = DEFAULT_CHECK_MODEL_AGE
) -> None:
    """Train model task.

    Top level task that trains a new model and makes sure the beast available model is
    stored. On this task level all database and context manager dependencies are resolved.

    Expected prediction job keys:  "id", "model", "lat", "lon", "name"

    Args:
        pj (dict): Prediction job
        context (TaskContext): Contect object that holds a config manager and a
            database connection.
        check_old_model_age (bool): check if model is too young to be retrained
    """

    # TODO Update get_prediction job in openstf_dbc such that hyperparams are already included in the prediciton jobs
    # Include hyperparameter information in the prediction job
    pj["hyper_params"] = {
        "training_period_days": TRAINING_PERIOD_DAYS,
        "featureset_name": "D",
    }
    pj["hyper_params"].update(context.database.get_hyper_params(pj))
    pj["feature_names"] = context.database.get_featureset(
        pj["hyper_params"]["featureset_name"]
    )

    # Get the paths for storing model and reports from the config manager
    trained_models_folder = Path(context.config.paths.trained_models_folder)
    save_figures_folder = trained_models_folder / str(pj["id"])

    # If required, let's check the old model age before retrieving all the input data
    if check_old_model_age:
        old_model_age = get_model_age(trained_models_folder, pj.get("id"))
        context.logger.debug(f"Old model age: {old_model_age}")
        if old_model_age < MAXIMUM_MODEL_AGE:
            # Old model is new enough. Skip this pj
            context.logger.info(
                f"Old model was new enough, skipping ({old_model_age}<{MAXIMUM_MODEL_AGE})",
                pid=pj.get("id"),
            )
            return

    context.perf_meter.checkpoint("Added metadata to PredictionJob")

    # Define start and end of the training input data
    datetime_start = datetime.utcnow() - timedelta(
        days=int(pj["hyper_params"]["training_period_days"])
    )
    datetime_end = datetime.utcnow()

    # Get training input data from database
    input_data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    context.perf_meter.checkpoint("Retrieved timeseries input")

    # Excecute the model training pipeline
    train_model_pipeline(
        pj,
        input_data,
        check_old_model_age=False,  # Old model age is already checked
        trained_models_folder=trained_models_folder,
        save_figures_folder=save_figures_folder,
    )

    context.perf_meter.checkpoint("Model trained")


def main(model_type=None):
    if model_type is None:
        model_type = [ml.value for ml in MLModelType]

    taskname = Path(__file__).name.replace(".py", "")
    with TaskContext(taskname) as context:
        PredictionJobLoop(context, model_type=model_type).map(train_model_task, context)


if __name__ == "__main__":
    main()
