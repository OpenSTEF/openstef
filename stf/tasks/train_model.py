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
from stf.model.train import train_model_for_specific_pj
from stf.tasks.utils.predictionjobloop import PredictionJobLoop
from stf.tasks.utils.taskcontext import TaskContext


def main():
    with TaskContext(__file__) as context:
        model_type = ["lgb", "xgb_quantile"]

        PredictionJobLoop(context, model_type=model_type).map(
            train_model_for_specific_pj, context
        )


if __name__ == "__main__":
    main()
