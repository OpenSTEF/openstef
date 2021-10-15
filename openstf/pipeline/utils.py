# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from typing import Tuple


def generate_forecast_datetime_range(
    resolution_minutes: int, horizon_minutes: int
) -> Tuple[datetime, datetime]:
    # get current date and time UTC
    datetime_utc = datetime.utcnow()
    # Datetime range for time interval to be predicted
    forecast_start = datetime_utc - timedelta(minutes=resolution_minutes)
    forecast_end = datetime_utc + timedelta(minutes=horizon_minutes)

    return forecast_start, forecast_end
