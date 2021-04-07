# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-
"""run_validationrobot.py

This module contains the CRON job that is periodically executed to detect
non zero stationsflatliners.

To provide the detection the folowing steps are caried out:
  1. Get historic training data (TDCV, Load, Weather and APX price data)
  2. Set threshold
  3. Detect flatliners
  4. Send slack message if at least one flatliner is found

Example:
    This module is meant to be called directly from a CRON job.

    Alternatively this code can be run directly by running::

        $ python run_validationrobot.py

Attributes:


"""
# Import builtins
from datetime import datetime, timedelta

from openstf.pipeline.run_validation_robot import validation_robot_pj
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def main():
    with TaskContext(__file__) as context:
        datetime_start = datetime.utcnow() - timedelta(days=7)
        datetime_end = datetime.utcnow()

        PredictionJobLoop(context).map(
            validation_robot_pj,
            context,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
        )


if __name__ == "__main__":
    main()
