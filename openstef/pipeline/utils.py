# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime

import pandas as pd
import scipy.ndimage as mnts


def generate_forecast_datetime_range(
    forecast_data: pd.DataFrame,
) -> tuple[datetime, datetime]:
    """Generate forecast range based on last cluster of null values in first target column of forecast data.

    Example:

        A forecast dataset with data between 2021-11-05 and 2021-11-19, and the
        target column 'load' as first column is given as input to this function. The first
        column 'load' has null values between 2021-11-17 04:00:00 and 2021-11-19 05:00:00.
        The null values at the end of the column indicate when forecasts are needed.
        Therefore this function sets starting time of forecasts as 2021-11-17 04:00:00 and
        end time of forecasts as 2021-11-19 05:00:00.

    Args:
        forecast_data: The forecast dataframe.

    Returns:
        Start and end datetimes of the forecast range.

    """
    # By labeling the True/False values (based on the isnull() statement) as clusters,
    # we find what True value belongs to what cluster and the amount of True clusters.
    label_clusters, n_clusters = mnts.label(forecast_data.iloc[:, 0].isnull().values)

    # If there are zero true clusters, it means the target column does not have nulls
    if n_clusters == 0:
        raise ValueError(
            "Forecast target column must have null values to indicate "
            "when forecast starts and ends."
        )

    # If there are multiple true clusters, we select the last cluster as forecast range
    forecast_range_data = forecast_data.loc[label_clusters == n_clusters]
    # We select first datetime index of last cluster
    forecast_start_dt = forecast_range_data.index[0].to_pydatetime()

    # Forecast end is based on last datetime of given forecast data
    forecast_end_dt = forecast_data.index[-1].to_pydatetime()
    return forecast_start_dt, forecast_end_dt
