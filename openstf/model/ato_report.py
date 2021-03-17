# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from pathlib import Path

from ktpbase.database import DataBase
from ktpbase.config.config import ConfigManager


def make_report_pj(pj):
    """Makes a csv report of realized load and forecasted load for a prediction job.
    The csv is saved in the icarus visuals folder.

    Args:
        pj (dict): Prediction job
    """
    # Intitialize databse
    db = DataBase()
    config = ConfigManager.get_instance()

    output_dir = Path(config.paths.webroot) / "ato_reports"
    output_file = "report_{}.csv".format(pj["name"])

    # Define start and end of realized date
    start = (datetime.utcnow() - timedelta(days=7)).replace(
        tzinfo=None, minute=0, hour=0
    )

    end_realized = datetime.utcnow()

    # Define end of forecast period
    end_forecast = (end_realized + timedelta(days=2)).replace(minute=0, hour=0)

    # Get realized load
    load = db.get_load_pid(pj["id"], start, end_realized)
    # Get forecast
    forecast = db.get_predicted_load_tahead(pj, start, end_forecast, t_ahead="24H")

    # Merge forecast and load into one dataframe
    forecast = forecast[["forecast_24.0h"]]
    forecast = forecast.rename(columns={"forecast_24.0h": "day_ahead_forecast"})
    forecast = forecast.dropna()
    load = load.merge(forecast, how="outer", left_index=True, right_index=True)

    # Convert timezone from utc to CET as end users expect this time zone
    load.index = load.index.tz_convert("CET")
    load.index.name = "datetime"
    # Rename to make extra clear this is realised
    load = load.rename(columns={"load": "realised_load"})

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to csv
    load.to_csv(output_dir / output_file)  # can join path elements with / operator

    return load
