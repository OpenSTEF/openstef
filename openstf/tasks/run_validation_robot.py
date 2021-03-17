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
from pathlib import Path

import cufflinks
import pandas as pd

# Import third-party
import plotly

# Import project modules
from openstf.model.validation_robot import nonzero_flatliner, zero_flatliner
from openstf.monitoring.teams import post_teams
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext

cufflinks.go_offline()
colors = {"Flatliner": "rgba(255, 0, 0, 0.3)", "Zero_value": "rgba(0, 255, 0, 0.3)"}


def validation_robot_pj(pj, context, datetime_start, datetime_end):
    """Run validation robot for specific prediction job.

    Args:
        pj: (dict) prediction job
        datetime_start: (datetime.datetime) start date
        datetime_end: (datetime.datetime) end date

    Returns:
        None

    """
    # Get load data
    load_data = context.database.get_load_pid(
        pj["id"], datetime_start, datetime_end, aggregated=False
    )
    if len(load_data) == 0:
        context.logger.warning("No load data found, skipping")
        return

    # Detect flatliners and zero-values
    load_errors = pd.DataFrame()

    flatliner_moments = nonzero_flatliner(load_data, threshold=4)
    zero_value_moments = zero_flatliner(load_data, threshold=4)

    # Combine single errors to singel result dataframe
    # Act if flatliner occurs
    if flatliner_moments is not None:
        num_flatliners = len(flatliner_moments)
        context.logger.warning(
            "Flatliner detected",
            prediction_name=pj["name"],
            num_flatliners=num_flatliners,
        )

        flatliner_moments["type"] = "Flatliner"
        load_errors = load_errors.append(flatliner_moments)
    # Same for zero-values
    if zero_value_moments is not None:
        num_zero_values = len(zero_value_moments)
        context.logger.warning(
            "Zero-value detected",
            prediction_name=pj["name"],
            num_zero_values=num_zero_values,
        )
        zero_value_moments["type"] = "Zero_value"
        load_errors = load_errors.append(zero_value_moments)

    num_load_errors = len(load_errors)

    # If some error was found, proceed to make a nice image and send
    # a message to Teams
    if num_load_errors > 0:
        context.logger.warning(
            "Total load errors detected", num_load_errors=num_load_errors
        )
        # Make nice image
        plot_url = _create_load_error_plot(pj, load_data, load_errors, context.config)
        context.logger.info("Created a load error plot", url=plot_url)
        # Send teams error message
        _send_teams_error_message(pj, num_load_errors, plot_url)


def _create_load_error_plot(pj, load_data, load_errors, config):
    start = load_errors["from_time"]
    end = load_errors["to_time"]
    check_type = load_errors["type"]
    vspanlist = [
        {
            "x0": from_time,
            "x1": to_time,
            "color": colors[check_type],
            "fill": True,
            "opacity": 0.4,
        }
        for from_time, to_time, check_type in zip(start, end, check_type)
    ]
    fig = load_data.iplot(vspan=vspanlist, asFigure=True)

    # Define image save location
    # NOTE should this also be moved to the config?
    image_name = "stations_detection.html"

    base_path = config.paths.trained_models
    base_url = config.dashboard.trained_models_url

    filename = Path(f'{base_path}/{pj["id"]}/{image_name}')

    web_link = f'{base_url}/{pj["id"]}/{image_name}'

    plotly.offline.plot(
        fig,
        show_link=False,
        filename=str(filename),
        auto_open=False,
    )

    return web_link


def _send_teams_error_message(pj, num_load_errors, buttonurl):
    # Prepare  message
    msg = {
        "fallback": "Load data error(s) detected {0}: {1} timeperiods".format(
            pj["name"], num_load_errors
        ),
        "title": "Load data error(s) detected {0} {1}".format(pj["name"], pj["id"]),
        "links": dict(buttontext="Robot Result", buttonurl=buttonurl),
        "text": (
            "Found {0} moments where there is a load error at pid {1}!".format(
                num_load_errors, pj["id"]
            )
        ),
        "color": "#764FA5",
    }
    post_teams(msg)


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
