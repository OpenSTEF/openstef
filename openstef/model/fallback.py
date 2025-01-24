# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, UTC

import pandas as pd


def generate_fallback(
    forecast_input: pd.DataFrame,
    load: pd.DataFrame,
    fallback_strategy: str = "extreme_day",
) -> pd.DataFrame:
    """Make a fall back forecast, Set the value of the forecast 'quality' column to 'substituted'.

    Currently only fallback_strategy=extreme day is implemented which return historic profile of most extreme day.

    Args:
        forecast_input : dataframe desired for the forecast
        load: index=datetime, columns=['load']
        fallback_strategy: strategy to determine fallback. options:
            - extreme_day: use daily profile of most extreme day
    Returns:
        Fallback forecast DataFrame with columns; 'forecast', 'quality'

    Raises:
        ValueError if len(load) == 0
        NotImplementedError if fallback_strategy != 'extreme_day'

    """
    # Check if load is completely empty
    if len(load.dropna()) == 0:
        raise ValueError("No historic load data available")

    if fallback_strategy != "extreme_day":
        raise NotImplementedError(
            f'fallback_strategy should be "extreme_day", received:{fallback_strategy}'
        )

    if fallback_strategy == "extreme_day":
        # Execute this fallback strategy
        # Find most extreme historic day and merge it by time-of-day to the requested moments

        # Find most extreme historic day (do not count today as it is incomplete)
        day_with_highest_load_date = (
            load[load.index < datetime.now(tz=UTC)].idxmax().load.date()
        )
        # generate datetime range of the day with the highest load
        from_datetime = pd.Timestamp(day_with_highest_load_date, tz=load.index.tz)
        till_datetime = from_datetime + pd.Timedelta("1 days")

        # slice load dataframe, only rows for the day with the highest load
        highest_daily_loadprofile = load.loc[
            (load.index >= from_datetime) & (load.index < till_datetime)
        ]

        # Match moments by time-of-day
        highest_daily_loadprofile.loc[:, "time"] = highest_daily_loadprofile.index.time
        forecast = pd.DataFrame(index=forecast_input.index)
        forecast["time"] = forecast.index.time
        forecast = (
            forecast.reset_index()
            .merge(
                highest_daily_loadprofile, left_on="time", right_on="time", how="outer"
            )
            .set_index("index")
        )

    # Rename so column is called forecast
    forecast = forecast[["load"]].rename(columns=dict(load="forecast"))

    # Add a column quality.
    forecast["quality"] = "substituted"

    return forecast
