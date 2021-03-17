# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""split_forecast_into_components.py

This module contains the CRON job that is periodically executed to make
prognoses of solar features. These features are usefull for splitting the load
in solar and wind contributions.
This is achieved by carrying out the folowing steps:
  1. Get the wind and solar reference data for the specific location of the
     customer
  2. Get the TDCV (Typical Domestic Consumption Values) data
  3. Fit a linear combination of above time series to the historic load data to
     determine the contributions of each energy source.
  4. Write the resulting coeficients to the SQL database.

Example:
    This module is meant to be called directly from a CRON job. A description of
    the CRON job can be found in the /k8s/CronJobs folder.
    Alternatively this code can be run directly by running::

        $ python split_forecast_into_components.py

Attributes:


"""
from openstf.model.split_energy import split_energy
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext


def main():
    with TaskContext(__file__) as context:
        model_type = ["xgb", "lgb"]

        PredictionJobLoop(
            context,
            model_type=model_type,
        ).map(lambda pj: split_energy(pj["id"]))


if __name__ == "__main__":
    main()
