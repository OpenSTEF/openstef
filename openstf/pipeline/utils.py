# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
import pandas as pd
from typing import Tuple


def generate_forecast_datetime_range(
    forecast_data: pd.DataFrame,
) -> Tuple[datetime, datetime]:
    """Generate start and end forecast time based on forecast data."""
    # Forecast start is based on when the target column has null values
    forecast_null_values = forecast_data.iloc[:, 0].isnull()
    forecast_start_dt64 = forecast_data.index[forecast_null_values].values[0]
    forecast_start_datetime = pd.Timestamp(forecast_start_dt64).to_pydatetime()

    # Forecast end is based on last datetime of given forecast data
    forecast_end_dt64 = forecast_data.index.values[-1]
    forecast_end_datetime = pd.Timestamp(forecast_end_dt64).to_pydatetime()
    return forecast_start_datetime, forecast_end_datetime
