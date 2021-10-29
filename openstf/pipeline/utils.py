# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
import pandas as pd
from typing import Tuple
import scipy.ndimage.measurements as mnts


def generate_forecast_datetime_range(
    forecast_data: pd.DataFrame,
) -> Tuple[datetime, datetime]:
    """Generate forecast range based on last cluster of null values in forecast data."""
    # Cluster null/nan values of target column to find last cluster with null/nan values
    labeled, clusters = mnts.label(forecast_data.iloc[:, 0].isnull().values)
    if clusters == 0:
        raise ValueError(
            "Forecast target column must have null values to indicate "
            "when forecast starts and ends."
        )
    # Get first datetime index of last cluster with null/nan values
    forecast_start_dt = forecast_data.loc[labeled == clusters].index[0].to_pydatetime()

    # Forecast end is based on last datetime of given forecast data
    forecast_end_dt = forecast_data.index[-1].to_pydatetime()
    return forecast_start_dt, forecast_end_dt
